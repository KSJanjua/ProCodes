"""Stage-2 temporal fine-tuning for Phase 1.

Loads a trained per-frame Phase-1 checkpoint (the stage-1 product), attaches
the zero-initialized TemporalAligner, and trains on short ordered clips with
full BPTT and per-clip state resets. Default = stage 2a: spatial model frozen,
only the temporal module learns -- cleanest attribution and cheapest memory
(no gradients flow through the backbone at all, since neither its inputs nor
its parameters require grad).

Usage (server, project root):

    python -m instancedepth.engine.train_hdi_temporal \\
        --config instancedepth/configs/hdi_temporal.yaml

    # smoke test:
    python -m instancedepth.engine.train_hdi_temporal \\
        --config instancedepth/configs/hdi_temporal.yaml \\
        --override optim.total_iters=20 optim.batch_size=1 optim.log_every=2 \\
                   optim.ckpt_every=10 optim.eval_every=10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.config import HDIConfig
from instancedepth.data.clip_dataset import ClipDatasetConfig, GIDClipDataset, collate_clips
from instancedepth.engine.trainer import Trainer
from instancedepth.losses.hdi_losses import HDILoss
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.utils.seed import seed_everything

log = logging.getLogger("instancedepth.engine.train_hdi_temporal")


def load_spatial_init(model: HolisticDepthModel, checkpoint_path: str) -> None:
    """Non-strict load of the stage-1 per-frame checkpoint: every key except
    the (new, zero-init) temporal aligners must match exactly."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in spatial init checkpoint: {unexpected[:5]}")
    stray = [k for k in missing if not k.startswith("temporal_aligners")]
    if stray:
        raise RuntimeError(f"spatial init checkpoint is missing non-temporal keys: {stray[:5]}")
    log.info("loaded spatial init from %s (%d fresh temporal keys)", checkpoint_path, len(missing))


def freeze_spatial(model: HolisticDepthModel) -> None:
    n_frozen = n_train = 0
    for name, p in model.named_parameters():
        train = name.startswith("temporal_aligners")
        p.requires_grad_(train)
        (n_train, n_frozen) = (n_train + p.numel(), n_frozen) if train else (n_train, n_frozen + p.numel())
    log.info("stage 2a freeze: %.2fM trainable (temporal) / %.1fM frozen (spatial)",
             n_train / 1e6, n_frozen / 1e6)


def build_clip_loader(cfg: HDIConfig, split: str) -> DataLoader:
    ds = GIDClipDataset(ClipDatasetConfig(
        annotations_root=cfg.data.annotations_root,
        split=split,
        image_size=cfg.data.image_size,
        max_depth=cfg.bins.max_depth,
        clip_len=cfg.temporal.clip_len,
        strides=cfg.temporal.clip_strides,
        hflip_prob=cfg.data.hflip_prob if split == "train" else 0.0,
    ))
    log.info("%s clip dataset: %d clips (len %d, strides %s)",
             split, len(ds), cfg.temporal.clip_len, cfg.temporal.clip_strides)
    return DataLoader(ds, batch_size=cfg.optim.batch_size, shuffle=(split == "train"),
                      num_workers=cfg.optim.num_workers, collate_fn=collate_clips,
                      drop_last=(split == "train"), pin_memory=True)


def make_compute_loss(loss_module: HDILoss):
    """Per-frame Phase-1 loss applied to every clip frame, averaged; state
    reset per clip, kept (with its autograd graph) across the clip's frames
    -- full truncated BPTT with truncation = clip length."""

    def compute_loss(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        images = batch["images"].to(device, non_blocking=True)   # (B,T,3,H,W)
        depths = batch["depths"].to(device, non_blocking=True)   # (B,T,1,H,W)
        T = images.shape[1]

        model.reset_temporal_state()
        acc: Dict[str, torch.Tensor] = {}
        for t in range(T):
            out = model(images[:, t])
            losses = loss_module(out.depth_final, out.seg_final, out.depth_levels,
                                 out.seg_levels, depths[:, t])
            for k, v in losses.items():
                acc[k] = acc.get(k, 0.0) + v / T
        return acc

    return compute_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = HDIConfig.from_yaml_with_overrides(args.config, args.override)
    assert cfg.temporal.enabled, "this entry point requires temporal.enabled: true"
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    model = HolisticDepthModel(cfg)
    if cfg.temporal.init_checkpoint:
        load_spatial_init(model, cfg.temporal.init_checkpoint)
    else:
        log.warning("temporal.init_checkpoint unset -- spatial model starts UNTRAINED "
                    "(smoke test only; a real run needs the stage-1 checkpoint)")
    if cfg.temporal.freeze_spatial:
        freeze_spatial(model)

    loss_module = HDILoss(cfg.loss, rd=cfg.bins.rd, max_depth=cfg.bins.max_depth, camera=cfg.camera)

    from instancedepth.engine.evaluate_hdi import make_eval_fn

    run_dir = Path(cfg.run_root) / cfg.run_name
    trainer = Trainer(
        cfg=cfg,
        model=model,
        compute_loss=make_compute_loss(loss_module),
        train_loader=build_clip_loader(cfg, "train"),
        run_dir=run_dir,
        eval_fn=make_eval_fn(cfg),   # per-frame reset-mode eval: the must-not-regress signal
    )
    if args.resume:
        trainer.resume(Path(args.resume))
    trainer.fit()


if __name__ == "__main__":
    main()
