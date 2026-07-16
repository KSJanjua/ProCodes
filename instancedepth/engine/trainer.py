"""Generic (phase-agnostic) training loop: optimizer/scheduler construction,
mixed precision, gradient clipping, checkpointing, and TensorBoard logging.

Optimizer/scheduler choice (AdamW, backbone-vs-head param-group LR split,
polynomial decay) is adapted from Depth Anything V2's own training code
(``metric_depth/train.py``) -- the paper states only the initial LR
(1e-5) and iteration count (55k) for Phase 1's "Global Depth Range
Pretraining" stage (Sec. 4.3); everything else here is that reference
repo's convention, not a paper fact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from instancedepth.configs.config import OptimConfig
from instancedepth.utils.checkpoint import load_checkpoint, save_checkpoint
from instancedepth.utils.manifest import RunManifest

log = logging.getLogger("instancedepth.engine.trainer")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig, backbone_module_name: str = "backbone") -> torch.optim.Optimizer:
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith(backbone_module_name) else head_params).append(p)
    groups = [
        {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": head_params, "lr": cfg.lr * cfg.head_lr_mult},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimConfig) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(it: int) -> float:
        if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
            return it / max(cfg.warmup_iters, 1)
        progress = min(it / max(cfg.total_iters, 1), 1.0)
        return (1.0 - progress) ** cfg.poly_power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Owns the training loop; model/loss/dataloader construction lives in
    ``train_hdi.py`` so this class stays phase-agnostic and reusable by a
    future Phase 2/3 trainer."""

    def __init__(
        self,
        cfg: Any,
        model: torch.nn.Module,
        compute_loss: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Dict[str, torch.Tensor]],
        train_loader: DataLoader,
        run_dir: Path,
        eval_fn: Optional[Callable[[torch.nn.Module], Dict[str, float]]] = None,
        device: Optional[torch.device] = None,
        build_optimizer_fn: Optional[Callable[[torch.nn.Module, Any], torch.optim.Optimizer]] = None,
    ) -> None:
        """``cfg`` is duck-typed (any dataclass exposing the same ``.optim.*``
        fields as ``OptimConfig``) so this trainer is reusable across phases
        with different config trees (Phase 1's ``HDIConfig``, Phase 2's
        ``Phase2Config``, ...) -- see ``instancedepth/engine/train_phase2.py``
        for an example passing a custom ``build_optimizer_fn`` because Phase
        2's backbone lives at a different attribute path than Phase 1's
        (name-prefix-based param-group splitting isn't one-size-fits-all
        across differently-structured models)."""
        self.cfg = cfg
        self.model = model
        self.compute_loss = compute_loss
        self.train_loader = train_loader
        self.run_dir = run_dir
        self.eval_fn = eval_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        optimizer_fn = build_optimizer_fn or build_optimizer
        self.optimizer = optimizer_fn(self.model, cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.optim)

        precision = cfg.optim.precision
        assert precision in _PRECISION_DTYPE, f"optim.precision must be one of {list(_PRECISION_DTYPE)}, got {precision}"
        self.autocast_dtype = _PRECISION_DTYPE[precision]
        self.use_autocast = precision != "fp32"
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and self.device.type == "cuda"))

        self.writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        self.iteration = 0
        self.best_metric: Optional[float] = None

    # ------------------------------------------------------------------ #
    def _autocast_ctx(self):
        if not self.use_autocast:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        if getattr(self.model, "cfg", None) is not None and getattr(self.model.cfg.backbone, "freeze", False):
            pass  # DINOv2Backbone.train() already re-freezes itself to eval() internally

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_ctx():
            losses = self.compute_loss(self.model, batch, self.device)
            total = losses["total"]

        # A batch can yield a loss with no path to any trainable parameter --
        # e.g. Phase 3 with a frozen depth branch (freeze_phase1) on a batch
        # with no valid occlusion pairs, where the criterion falls back to
        # ``base_depth.sum()*0`` whose only grad path was the (now frozen)
        # Phase-1 branch. Nothing to learn from it: skip backward/step rather
        # than let ``backward()`` raise "does not require grad". The LR
        # schedule still advances so the run stays on its iteration budget.
        if total.requires_grad:
            if self.scaler.is_enabled():
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)
                self.optimizer.step()
        self.scheduler.step()

        return {k: float(v.detach().item()) for k, v in losses.items()}

    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        manifest = RunManifest.build(self.cfg, repo_root=Path(__file__).resolve().parents[2])
        manifest.save(self.run_dir / "manifest.json")
        log.info("Run manifest written to %s (experiment_id=%s)", self.run_dir / "manifest.json", manifest.experiment_id)

        data_iter = iter(self.train_loader)
        while self.iteration < self.cfg.optim.total_iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss_dict = self.train_step(batch)
            self.iteration += 1

            if self.iteration % self.cfg.optim.log_every == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"train/{k}", v, self.iteration)
                # Log each param group's LR by index -- robust to optimizers
                # with a single group (e.g. a frozen-backbone stage where the
                # empty group is dropped: the video temporal stage, or Phase 3
                # with freeze_phase1) as well as the usual backbone+head pair.
                groups = self.optimizer.param_groups
                names = ["lr_backbone", "lr_head"] if len(groups) == 2 \
                    else [f"lr_group{i}" for i in range(len(groups))]
                for name, g in zip(names, groups):
                    self.writer.add_scalar(f"train/{name}", g["lr"], self.iteration)
                log.info("iter %d/%d  total=%.4f", self.iteration, self.cfg.optim.total_iters, loss_dict["total"])

            if self.iteration % self.cfg.optim.ckpt_every == 0:
                save_checkpoint(self.run_dir / "latest.pth", self.model, self.optimizer, self.scheduler,
                                 self.iteration, self.best_metric)
                save_checkpoint(self.run_dir / f"ckpt_{self.iteration:07d}.pth", self.model, self.optimizer,
                                 self.scheduler, self.iteration, self.best_metric)
                log.info("checkpoint written at iter %d", self.iteration)

            if self.eval_fn is not None and self.iteration % self.cfg.optim.eval_every == 0:
                metrics = self.eval_fn(self.model)
                for k, v in metrics.items():
                    self.writer.add_scalar(f"eval/{k}", v, self.iteration)
                primary = metrics.get("abs_rel")
                if primary is not None and (self.best_metric is None or primary < self.best_metric):
                    self.best_metric = primary
                    save_checkpoint(self.run_dir / "best.pth", self.model, self.optimizer, self.scheduler,
                                     self.iteration, self.best_metric)
                    log.info("new best checkpoint at iter %d (abs_rel=%.4f)", self.iteration, primary)

        save_checkpoint(self.run_dir / "latest.pth", self.model, self.optimizer, self.scheduler,
                         self.iteration, self.best_metric)
        log.info("training complete at iter %d", self.iteration)

    def resume(self, path: Path) -> None:
        ckpt = load_checkpoint(path, self.model, self.optimizer, self.scheduler, map_location=str(self.device))
        self.iteration = ckpt["iteration"]
        self.best_metric = ckpt.get("best_metric")
        log.info("resumed from %s at iter %d", path, self.iteration)
