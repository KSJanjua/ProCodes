"""Phase-3 training with the improved (bounded pair-attention) relation head.

Identical to ``instancedepth.engine.train_phase3`` — same data, matcher,
criterion, optimizer, eval — with ``Phase3VideoModel`` substituted, whose
``BoundedPairAttentionHead`` is interface-compatible with the original Φo.
Pair it with ``freeze_phase1: true`` so the dense
base cannot drift and the improved head is the only variable.

Usage:
    python -m videodepth.engine.train_phase3_video \\
        --config instancedepth/configs/phase3_current.yaml \\
        --override phase1_checkpoint=runs/hdi_enhanced/best.pth \\
                   phase2_checkpoint=runs/phase2_run/best.pth \\
        --run-name phase3_video --max-corr 0.15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.engine.train_phase3 import (
    build_dataloader, build_phase3_optimizer, make_compute_loss,
)
from instancedepth.engine.trainer import Trainer
from instancedepth.losses.phase3_losses import Phase3Criterion
from instancedepth.models.phase2.matcher import Phase2HungarianMatcher
from instancedepth.utils.seed import seed_everything
from videodepth.models.phase3_video import Phase3VideoModel

log = logging.getLogger("videodepth.engine.train_phase3_video")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-corr", type=float, default=0.15,
                    help="hard cap on the multiplicative depth correction (±fraction)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = Phase3Config.from_yaml_with_overrides(args.config, args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    seed_everything(cfg.seed)

    if not cfg.freeze_phase1:
        log.warning("freeze_phase1 is FALSE — the dense base can drift "
                    "(0.078 -> 0.139 abs_rel in recorded runs)")

    model = Phase3VideoModel(cfg, max_corr=args.max_corr)
    matcher = Phase2HungarianMatcher(
        cost_class=cfg.phase2.matcher.cost_class, cost_mask=cfg.phase2.matcher.cost_mask,
        cost_dice=cfg.phase2.matcher.cost_dice, cost_depth=cfg.phase2.matcher.cost_depth,
        num_points=cfg.phase2.matcher.num_points,
    )
    criterion = Phase3Criterion(cfg.loss)
    train_loader = build_dataloader(cfg, split="train")

    from instancedepth.engine.evaluate_phase3 import make_eval_fn

    run_dir = Path(cfg.run_root) / cfg.run_name
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
