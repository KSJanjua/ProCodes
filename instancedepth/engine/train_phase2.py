"""Phase 2 (Instance Depth Layer Prediction) training entry point.

Usage (on the Backend.AI server, from the project root):

    python -m instancedepth.engine.train_phase2 \\
        --config instancedepth/configs/phase2_mask2former.yaml \\
        --run-name my_first_phase2_run

    # smoke test:
    python -m instancedepth.engine.train_phase2 \\
        --config instancedepth/configs/phase2_mask2former.yaml \\
        --override optim.total_iters=50 optim.batch_size=1 optim.log_every=5 \\
                   optim.ckpt_every=25 optim.eval_every=25 optim.warmup_iters=0

    # resume:
    python -m instancedepth.engine.train_phase2 \\
        --config instancedepth/configs/phase2_mask2former.yaml \\
        --resume runs/phase2_mask2former/latest.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.phase2_config import Phase2Config
from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid
from instancedepth.engine.trainer import Trainer, build_scheduler
from instancedepth.models.phase2.criterion import Phase2Criterion
from instancedepth.models.phase2.matcher import Phase2HungarianMatcher
from instancedepth.models.phase2.model import Phase2Model
from instancedepth.utils.seed import seed_everything

log = logging.getLogger("instancedepth.engine.train_phase2")

# Substring identifying the Swin backbone's parameters inside
# Mask2FormerForUniversalSegmentation's module tree, per HF's
# Mask2FormerModel naming convention (pixel_level_module wraps the
# backbone encoder + the pixel decoder; transformer_module is the
# masked-attention transformer decoder). **Expected, not yet confirmed by
# execution** -- scripts/verify_mask2former_api.py's output should be
# cross-checked against this; if wrong, the backbone LR multiplier
# (optim.backbone_lr_mult) silently applies to zero parameters instead of
# the actual backbone, which the sanity log below is meant to catch.
_BACKBONE_NAME_SUBSTRING = "pixel_level_module.encoder"


def build_phase2_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if _BACKBONE_NAME_SUBSTRING in name else other_params).append(p)

    n_backbone = sum(p.numel() for p in backbone_params)
    n_other = sum(p.numel() for p in other_params)
    log.info(
        "Phase 2 param groups: backbone (%s) = %d params (%.1fM), other = %d params (%.1fM). "
        "If 'backbone' looks like 0 or the wrong order of magnitude, "
        "_BACKBONE_NAME_SUBSTRING in train_phase2.py needs updating -- "
        "check scripts/verify_mask2former_api.py's output.",
        _BACKBONE_NAME_SUBSTRING, len(backbone_params), n_backbone / 1e6, len(other_params), n_other / 1e6,
    )

    groups = [
        {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": other_params, "lr": cfg.lr},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_dataloader(cfg: Phase2Config, split: str) -> DataLoader:
    ds_cfg = GIDDatasetConfig(
        annotations_root=cfg.data.annotations_root,
        split=split,
        image_size=cfg.data.image_size,
        min_instance_px=cfg.data.min_instance_px,
        hflip_prob=cfg.data.hflip_prob if split == "train" else 0.0,
        color_jitter=cfg.data.color_jitter if split == "train" else 0.0,
        size_divisor=32,   # Swin-L Mask2Former stride (not DINOv2/14's 14) -- 736x1280 is 32x23 by 32x40
    )
    dataset = GIDInstanceDepthDataset(ds_cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.optim.num_workers,
        collate_fn=collate_gid,
        drop_last=(split == "train"),
        pin_memory=True,
    )


def make_compute_loss(matcher: Phase2HungarianMatcher, criterion: Phase2Criterion):
    def compute_loss(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        image = batch["image"].to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items() if k != "track_ids"}
            for t in batch["targets"]
        ]

        out = model(image)
        indices = matcher(out.class_logits, out.mask_logits, out.depth_layers, targets)
        return criterion(out.class_logits, out.mask_logits, out.depth_layers, targets, indices)

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

    cfg = Phase2Config.from_yaml_with_overrides(args.config, args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    run_dir = Path(cfg.run_root) / cfg.run_name
    log.info("run_dir=%s", run_dir)

    model = Phase2Model(
        checkpoint=cfg.model.checkpoint, checkpoint_dir=cfg.model.checkpoint_dir,
        allow_hub_download=cfg.model.allow_hub_download, num_classes=cfg.model.num_classes,
    )
    matcher = Phase2HungarianMatcher(
        cost_class=cfg.matcher.cost_class, cost_mask=cfg.matcher.cost_mask,
        cost_dice=cfg.matcher.cost_dice, cost_depth=cfg.matcher.cost_depth,
        num_points=cfg.matcher.num_points,
    )
    criterion = Phase2Criterion(
        num_classes=cfg.model.num_classes, eos_coef=cfg.loss.eos_coef,
        oversample_ratio=cfg.loss.oversample_ratio, importance_sample_ratio=cfg.loss.importance_sample_ratio,
        weight_class=cfg.loss.weight_class, weight_mask=cfg.loss.weight_mask,
        weight_dice=cfg.loss.weight_dice, weight_depth=cfg.loss.weight_depth,
    )
    train_loader = build_dataloader(cfg, split="train")

    from instancedepth.engine.evaluate_phase2 import make_eval_fn

    trainer = Trainer(
        cfg=cfg,
        model=model,
        compute_loss=make_compute_loss(matcher, criterion),
        train_loader=train_loader,
        run_dir=run_dir,
        eval_fn=make_eval_fn(cfg),
        build_optimizer_fn=build_phase2_optimizer,
    )
    if args.resume:
        trainer.resume(Path(args.resume))

    trainer.fit()


if __name__ == "__main__":
    main()
