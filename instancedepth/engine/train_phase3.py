"""Phase 3 (Occlusion-Aware Depth Refinement) training entry point.

Reproduces paper Sec. 4.3 "Occlusion-Aware Joint Refinement": the instance
decoder (Phase 2) is frozen; the depth encoder+decoder (Phase 1) are
fine-tuned at 1e-6; the new Phi_o head is trained. 25k iterations.

Usage (project root, Backend.AI server):

    python -m instancedepth.engine.train_phase3 \\
        --config instancedepth/configs/phase3.yaml \\
        --override phase1_checkpoint=runs/hdi_faithful/best.pth \\
                   phase2_checkpoint=runs/phase2_mask2former/best.pth \\
        --run-name phase3_first

    # smoke test (tiny):
    python -m instancedepth.engine.train_phase3 \\
        --config instancedepth/configs/phase3.yaml \\
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

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid
from instancedepth.data.occlusion_index import occlusion_frame_indices
from instancedepth.engine.trainer import Trainer
from instancedepth.losses.phase3_losses import Phase3Criterion
from instancedepth.models.phase2.matcher import Phase2HungarianMatcher
from instancedepth.models.phase3.model import Phase3Model
from instancedepth.models.phase3.targets import build_dense_gt_rois
from instancedepth.utils.seed import seed_everything

log = logging.getLogger("instancedepth.engine.train_phase3")


def build_phase3_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """Two groups: the fine-tuned Phase-1 depth branch at ``lr`` (paper 1e-6),
    the fresh Phi_o head at ``lr * head_lr_mult``. Phase 2 is frozen (its
    params have requires_grad=False and are skipped)."""
    depth_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("phase1"):
            depth_params.append(p)
        elif name.startswith("relation_head"):
            head_params.append(p)
        else:
            # Any trainable param not in phase1/relation_head is unexpected
            # (phase2 is frozen); surface it rather than silently grouping it.
            log.warning("unexpected trainable param outside phase1/relation_head: %s", name)
            head_params.append(p)

    n_depth = sum(p.numel() for p in depth_params)
    n_head = sum(p.numel() for p in head_params)
    log.info("Phase 3 param groups: depth-branch=%.1fM (lr=%.1e), Phi_o=%.1fM (lr=%.1e)",
             n_depth / 1e6, cfg.lr, n_head / 1e6, cfg.lr * cfg.head_lr_mult)

    # When Phase 1 is frozen (freeze_phase1: true) the depth group is empty;
    # dropping it keeps a single, cleanly-logged group instead of a dead one.
    groups = []
    if depth_params:
        groups.append({"params": depth_params, "lr": cfg.lr})          # group 0 -> logged as lr_backbone
    else:
        log.info("Phase 1 frozen -> optimizing only the Phi_o relation head")
    groups.append({"params": head_params, "lr": cfg.lr * cfg.head_lr_mult})   # -> logged as lr_head
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_dataloader(cfg: Phase3Config, split: str) -> DataLoader:
    ds_cfg = GIDDatasetConfig(
        annotations_root=cfg.data.annotations_root,
        split=split,
        image_size=cfg.data.image_size,
        max_depth=cfg.data.max_depth,
        min_instance_px=cfg.data.min_instance_px,
        hflip_prob=cfg.data.hflip_prob if split == "train" else 0.0,
        color_jitter=cfg.data.color_jitter if split == "train" else 0.0,
        size_divisor=cfg.data.size_divisor,   # 32 (Swin-L); depth branch resizes internally
    )
    dataset = GIDInstanceDepthDataset(ds_cfg)
    if split == "train" and cfg.data.occlusion_only:
        # Bias training toward frames that can actually form occlusion pairs
        #. Eval (split=="test") is never filtered -- dense
        # metrics must cover the whole test split.
        from torch.utils.data import Subset
        occ = occlusion_frame_indices(dataset, max_depth=cfg.data.max_depth)
        if not occ:
            raise RuntimeError(
                "data.occlusion_only=True but no frames have >=2 overlapping "
                "instances -- check the annotations / lower the criterion.")
        log.info("occlusion_only: training on %d / %d frames", len(occ), len(dataset))
        dataset = Subset(dataset, occ)
    return DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.optim.num_workers,
        collate_fn=collate_gid,
        drop_last=(split == "train"),
        pin_memory=True,
    )


def make_compute_loss(cfg: Phase3Config, matcher: Phase2HungarianMatcher, criterion: Phase3Criterion):
    out_hw = tuple(cfg.head.roi_size)
    sr = cfg.head.roi_sampling_ratio

    def compute_loss(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        criterion.to(device)
        image = batch["image"].to(device, non_blocking=True)
        gt_depth = batch["depth"].to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items() if k != "track_ids"}
            for t in batch["targets"]
        ]

        output, aux = model(image)
        p2 = aux["p2"]
        # Hungarian match frozen Phase-2 predictions to GT (for supervision only).
        indices = matcher(p2.class_logits, p2.mask_logits, p2.depth_layers, targets)
        refine_targets = build_dense_gt_rois(aux["pairs"], indices, targets, gt_depth, out_hw, sr)
        losses = criterion(output, refine_targets, gt_depth)
        # Diagnostics (logged by Trainer, not part of `total`): how much
        # supervision this step actually had. num_valid_pairs == 0 explains a
        # total==0 step.
        losses["num_pairs"] = gt_depth.new_tensor(float(len(aux["pairs"])))
        losses["num_valid_pairs"] = refine_targets.pair_valid.sum().float()
        return losses

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

    cfg = Phase3Config.from_yaml_with_overrides(args.config, args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    run_dir = Path(cfg.run_root) / cfg.run_name
    log.info("run_dir=%s", run_dir)

    if not cfg.phase1_checkpoint or not cfg.phase2_checkpoint:
        log.warning("phase1_checkpoint / phase2_checkpoint not both set -- this is a smoke-test "
                    "configuration; a real run needs both trained checkpoints.")

    model = Phase3Model(cfg)
    matcher = Phase2HungarianMatcher(
        cost_class=cfg.phase2.matcher.cost_class, cost_mask=cfg.phase2.matcher.cost_mask,
        cost_dice=cfg.phase2.matcher.cost_dice, cost_depth=cfg.phase2.matcher.cost_depth,
        num_points=cfg.phase2.matcher.num_points,
    )
    criterion = Phase3Criterion(cfg.loss)
    train_loader = build_dataloader(cfg, split="train")

    from instancedepth.engine.evaluate_phase3 import make_eval_fn

    trainer = Trainer(
        cfg=cfg,
        model=model,
        compute_loss=make_compute_loss(cfg, matcher, criterion),
        train_loader=train_loader,
        run_dir=run_dir,
        eval_fn=make_eval_fn(cfg),
        build_optimizer_fn=build_phase3_optimizer,
    )
    if args.resume:
        trainer.resume(Path(args.resume))

    trainer.fit()


if __name__ == "__main__":
    main()
