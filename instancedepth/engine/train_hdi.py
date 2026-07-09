"""Phase 1 (Holistic Depth Initialization) training entry point.

Usage (on the Backend.AI server, from the project root):

    python -m instancedepth.engine.train_hdi \\
        --config instancedepth/configs/hdi.yaml \\
        --run-name my_first_run

    # CLI dotlist overrides, e.g. to shrink the run for a smoke test:
    python -m instancedepth.engine.train_hdi \\
        --config instancedepth/configs/hdi.yaml \\
        --override optim.total_iters=200 optim.batch_size=2 optim.log_every=10

    # resume:
    python -m instancedepth.engine.train_hdi \\
        --config instancedepth/configs/hdi.yaml --resume runs/hdi_faithful/latest.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.config import HDIConfig
from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid
from instancedepth.engine.trainer import Trainer
from instancedepth.losses.hdi_losses import HDILoss
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.utils.seed import seed_everything

log = logging.getLogger("instancedepth.engine.train_hdi")


def build_dataloader(cfg: HDIConfig, split: str) -> DataLoader:
    ds_cfg = GIDDatasetConfig(
        annotations_root=cfg.data.annotations_root,
        split=split,
        image_size=cfg.data.image_size,
        min_instance_px=cfg.data.min_instance_px,
        hflip_prob=cfg.data.hflip_prob if split == "train" else 0.0,
        color_jitter=cfg.data.color_jitter if split == "train" else 0.0,
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


def make_compute_loss(loss_module: HDILoss):
    def compute_loss(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        image = batch["image"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        output = model(image)
        return loss_module(
            depth_final=output.depth_final,
            seg_final=output.seg_final,
            depth_levels=output.depth_levels,
            seg_levels=output.seg_levels,
            gt_depth=depth_gt,
        )

    return compute_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="e.g. instancedepth/configs/hdi.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="dotlist overrides, e.g. optim.total_iters=200")
    ap.add_argument("--run-name", default=None, help="overrides cfg.run_name if set")
    ap.add_argument("--resume", default=None, help="path to a checkpoint (e.g. runs/hdi_faithful/latest.pth)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = HDIConfig.from_yaml_with_overrides(args.config, args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    run_dir = Path(cfg.run_root) / cfg.run_name
    log.info("run_dir=%s", run_dir)

    model = HolisticDepthModel(cfg)
    loss_module = HDILoss(cfg.loss, rd=cfg.bins.rd, max_depth=cfg.bins.max_depth, camera=cfg.camera)
    train_loader = build_dataloader(cfg, split="train")

    # local import: instancedepth.engine.evaluate_hdi imports build_dataloader
    # from this module, so importing it back at module load time would be
    # circular -- deferring to call time avoids that.
    from instancedepth.engine.evaluate_hdi import make_eval_fn

    trainer = Trainer(
        cfg=cfg,
        model=model,
        compute_loss=make_compute_loss(loss_module),
        train_loader=train_loader,
        run_dir=run_dir,
        eval_fn=make_eval_fn(cfg),
    )
    if args.resume:
        trainer.resume(Path(args.resume))

    trainer.fit()


if __name__ == "__main__":
    main()
