"""Temporal-stage training: trained Phase-1 spatial model (frozen) + streaming
temporal stabilizer, on motion-weighted clips, with the temporal loss the
first attempt was missing.

The three root causes of the earlier null result (docs/AUDIT_2026.md §1.1) and
their fixes here:

  1. **No temporal loss** -> ``TemporalGradientMatchingLoss`` (VDA's loss
     without VDA's heavy clip-attention architecture) is the training signal.
  2. **No motion in training clips** -> motion-weighted sampling over strides
     spanning up to ~100 frames (data/motion_clips.py).
  3. **Selection blind to temporal quality** -> best.pth chosen on a
     STREAMING eval score: abs_rel + tae_weight · TAE.

Usage (server, project root):

    python -m videodepth.engine.train_video \\
        --config videodepth/configs/video_temporal.yaml \\
        --override init_checkpoint=runs/hdi_enhanced/best.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from instancedepth.data.clip_dataset import ClipDatasetConfig
from instancedepth.engine.trainer import Trainer
from instancedepth.losses.hdi_losses import SigLogLoss
from instancedepth.utils.seed import seed_everything
from videodepth.configs.config import VideoConfig
from videodepth.data.motion_clips import build_motion_clip_loader
from videodepth.engine.evaluate_video import evaluate_streaming
from videodepth.losses.temporal_losses import TemporalGradientMatchingLoss
from videodepth.models.video_model import VideoDepthModel

log = logging.getLogger("videodepth.engine.train_video")


def build_video_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """Single param group over trainable params only — with freeze_spatial
    that is exactly the temporal stabilizer. (The Trainer's default builder
    needs backbone/head lr-mult fields this stage doesn't have.)"""
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    log.info("video optimizer: %.2fM trainable params (lr=%.1e)", n / 1e6, cfg.lr)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def make_compute_loss(cfg: VideoConfig):
    """spatial SigLog per frame (keeps single-frame accuracy pinned) +
    temporal gradient matching over the clip (the anti-flicker signal)."""
    silog = SigLogLoss()
    tgm = TemporalGradientMatchingLoss(order=cfg.loss.tgm_order,
                                       log_space=cfg.loss.tgm_log_space)

    def compute_loss(model: torch.nn.Module, batch: Dict[str, Any],
                     device: torch.device) -> Dict[str, torch.Tensor]:
        images = batch["images"].to(device, non_blocking=True)   # (B,T,3,H,W)
        depths = batch["depths"].to(device, non_blocking=True)   # (B,T,1,H,W)
        preds = model.forward_clip(images)                       # (B,T,1,H,W)

        T = preds.shape[1]
        spatial = sum(silog(preds[:, t], depths[:, t], depths[:, t] > 0)
                      for t in range(T)) / T
        temporal = cfg.loss.temporal_weight * tgm(preds, depths)
        return {"spatial_silog": spatial,
                "temporal_gradient_matching": temporal,
                "total": spatial + temporal}

    return compute_loss


def make_eval_fn(cfg: VideoConfig, hdi_cfg):
    """Streaming (sequence-ordered, stateful) periodic eval; the selection
    scalar 'abs_rel' the Trainer minimises is abs_rel + tae_weight·TAE, so
    checkpoint choice finally sees what the temporal head does."""

    def eval_fn(model: torch.nn.Module) -> Dict[str, float]:
        device = next(model.parameters()).device
        m = evaluate_streaming(model, hdi_cfg, device,
                               max_frames=cfg.eval.max_frames)
        m["abs_rel_frame"] = m["abs_rel"]
        m["abs_rel"] = m["abs_rel_frame"] + cfg.eval.tae_weight * m["temporal_alignment_error"]
        return m

    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = VideoConfig.from_yaml_with_overrides(args.config, args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    from instancedepth.configs.config import HDIConfig
    hdi_cfg = HDIConfig.from_yaml(cfg.hdi_config)

    model = VideoDepthModel.from_config(cfg)

    run_dir = Path(cfg.run_root) / cfg.run_name
    base_cfg = ClipDatasetConfig(
        annotations_root=hdi_cfg.data.annotations_root,
        split="train",
        image_size=hdi_cfg.data.image_size,
        max_depth=hdi_cfg.bins.max_depth,
        clip_len=cfg.clips.clip_len,
        strides=cfg.clips.strides,
        hflip_prob=hdi_cfg.data.hflip_prob,
    )
    train_loader = build_motion_clip_loader(
        base_cfg, cfg.clips,
        batch_size=cfg.optim.batch_size, num_workers=cfg.optim.num_workers,
        cache_path=run_dir / cfg.clips.motion_cache,
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        compute_loss=make_compute_loss(cfg),
        train_loader=train_loader,
        run_dir=run_dir,
        eval_fn=make_eval_fn(cfg, hdi_cfg),
        build_optimizer_fn=build_video_optimizer,
    )
    if args.resume:
        trainer.resume(Path(args.resume))
    trainer.fit()


if __name__ == "__main__":
    main()
