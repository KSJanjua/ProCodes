"""Fine-tune the full pretrained Depth-Anything-V2 (encoder + DPT head) as
Phase 1 — the AbsRel lever (videodepth/docs/DESIGN.md).

Point BOTH paths at the same official checkpoint file:
  * ``backbone.checkpoint_path`` — the DINOv2 wrapper reads the encoder
    ('pretrained.*' keys, auto-detected);
  * ``dav2_checkpoint``          — this stage reads the DPT head
    ('depth_head.*' keys).
Recommended file: the *metric hypersim* fine-tune
(depth_anything_v2_metric_hypersim_vitl.pth — indoor range, sigmoid head
matches). The relative checkpoint also loads (the activation has no params)
but starts from disparity semantics.

Usage:
    python -m videodepth.engine.train_dav2 \\
        --config videodepth/configs/dav2_full.yaml \\
        --override dav2_checkpoint=/path/to/depth_anything_v2_metric_hypersim_vitl.pth \\
                   backbone.checkpoint_path=/path/to/depth_anything_v2_metric_hypersim_vitl.pth

    # evaluate (streaming: abs_rel + TAE in one pass):
    python -m videodepth.engine.evaluate_video ... — or:
    python -m videodepth.engine.train_dav2 --evaluate \\
        --config videodepth/configs/dav2_full.yaml \\
        --checkpoint runs/dav2_full/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid
from instancedepth.engine.trainer import Trainer
from instancedepth.losses.hdi_losses import GradientMatchingLoss, SigLogLoss
from instancedepth.utils.seed import seed_everything
from videodepth.configs.config import DAV2Config
from videodepth.engine.evaluate_video import evaluate_streaming
from videodepth.models.dav2_dpt import DAV2MetricModel

log = logging.getLogger("videodepth.engine.train_dav2")


def build_dataloader(cfg: DAV2Config, split: str) -> DataLoader:
    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=cfg.data.annotations_root,
        split=split,
        image_size=cfg.data.image_size,
        max_depth=cfg.data.max_depth,
        min_instance_px=cfg.data.min_instance_px,
        hflip_prob=cfg.data.hflip_prob if split == "train" else 0.0,
        color_jitter=cfg.data.color_jitter if split == "train" else 0.0,
    ))
    return DataLoader(ds, batch_size=cfg.optim.batch_size, shuffle=(split == "train"),
                      num_workers=cfg.optim.num_workers, collate_fn=collate_gid,
                      drop_last=(split == "train"), pin_memory=True)


def build_dav2_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """Encoder at lr*encoder_lr_mult, DPT head at lr — the standard
    foundation-model fine-tune split."""
    enc, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith("backbone") else head).append(p)
    log.info("DAv2 param groups: encoder=%.1fM (lr=%.1e), head=%.1fM (lr=%.1e)",
             sum(p.numel() for p in enc) / 1e6, cfg.lr * cfg.encoder_lr_mult,
             sum(p.numel() for p in head) / 1e6, cfg.lr)
    return torch.optim.AdamW(
        [{"params": enc, "lr": cfg.lr * cfg.encoder_lr_mult},
         {"params": head, "lr": cfg.lr}],
        lr=cfg.lr, weight_decay=cfg.weight_decay)


def make_compute_loss(cfg: DAV2Config):
    silog = SigLogLoss(cfg.silog_lambda)
    grad_match = GradientMatchingLoss()

    def compute_loss(model: torch.nn.Module, batch: Dict[str, Any],
                     device: torch.device) -> Dict[str, torch.Tensor]:
        image = batch["image"].to(device, non_blocking=True)
        gt = batch["depth"].to(device, non_blocking=True)
        pred = model(image)
        mask = gt > 0
        losses = {"silog": silog(pred, gt, mask)}
        if cfg.gradient_matching_weight > 0:
            losses["gradient_matching"] = \
                cfg.gradient_matching_weight * grad_match(pred, gt, mask)
        losses["total"] = sum(losses.values())
        return losses

    return compute_loss


def make_eval_fn(cfg: DAV2Config):
    def eval_fn(model: torch.nn.Module) -> Dict[str, float]:
        device = next(model.parameters()).device
        return evaluate_streaming(model, cfg, device,
                                  max_frames=cfg.eval_max_frames,
                                  precision=cfg.optim.precision)
    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--evaluate", action="store_true",
                    help="full streaming eval of --checkpoint instead of training")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = DAV2Config.from_yaml_with_overrides(args.config, args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    if args.evaluate:
        from instancedepth.utils.checkpoint import load_checkpoint
        assert args.checkpoint, "--evaluate requires --checkpoint"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.dav2_checkpoint = None   # the run checkpoint supersedes the init
        model = DAV2MetricModel.from_config(cfg).to(device)
        load_checkpoint(Path(args.checkpoint), model, map_location=str(device),
                        restore_rng=False)
        metrics = evaluate_streaming(model, cfg, device, split=args.split,
                                     precision=cfg.optim.precision)
        log.info("DAv2 streaming eval (%s):\n%s", args.split, json.dumps(metrics, indent=2))
        out = Path(cfg.run_root) / cfg.run_name / f"eval_streaming_{args.split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        log.info("Wrote %s", out)
        return

    model = DAV2MetricModel.from_config(cfg)
    trainer = Trainer(
        cfg=cfg,
        model=model,
        compute_loss=make_compute_loss(cfg),
        train_loader=build_dataloader(cfg, "train"),
        run_dir=Path(cfg.run_root) / cfg.run_name,
        eval_fn=make_eval_fn(cfg),
        build_optimizer_fn=build_dav2_optimizer,
    )
    if args.resume:
        trainer.resume(Path(args.resume))
    trainer.fit()


if __name__ == "__main__":
    main()
