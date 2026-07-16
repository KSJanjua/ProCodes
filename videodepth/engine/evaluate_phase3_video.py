"""Phase-3 evaluation for the video-aware model (bounded pair-attention head).

Mirror of ``instancedepth.engine.evaluate_phase3`` with one substitution:
the model is ``Phase3VideoModel`` (``BoundedPairAttentionHead``), so a
checkpoint trained by ``videodepth.engine.train_phase3_video`` loads into the
matching head. The metric logic itself is reused unchanged from
``evaluate_phase3.evaluate`` (overall/occ, refined/base), so the numbers are
directly comparable to the stock Phase-3 eval.

``--max-corr`` must match training (it does not change parameter shapes, only
the forward correction bound, but keeping it consistent keeps eval faithful).

Usage:
    python -m videodepth.engine.evaluate_phase3_video \\
        --config videodepth/configs/phase3_dav2_p2run.yaml \\
        --checkpoint runs/phase3_video_dav2/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.engine.evaluate_phase3 import evaluate
from instancedepth.engine.train_phase3 import build_dataloader
from instancedepth.utils.checkpoint import load_checkpoint
from videodepth.models.phase3_video import Phase3VideoModel

log = logging.getLogger("videodepth.engine.evaluate_phase3_video")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--max-corr", type=float, default=0.15,
                    help="must match the value used at training (arch-invariant, "
                         "affects the correction bound in forward)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = Phase3Config.from_yaml_with_overrides(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The full Phase-3 checkpoint supersedes the base-phase init weights; null
    # them so construction never needs (possibly absent) init files, and the
    # two "smoke test only" warnings from Phase3Model are expected/benign here.
    cfg.phase1_checkpoint = None
    cfg.phase2_checkpoint = None
    model = Phase3VideoModel(cfg, max_corr=args.max_corr).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)

    loader = build_dataloader(cfg, split=args.split)
    result = evaluate(model, loader, device, args.max_batches, precision=cfg.optim.precision)

    log.info("Phase 3 (video head) eval (%s, %s):\n%s", args.split, args.checkpoint,
             json.dumps(result, indent=2))
    out_path = Path(cfg.run_root) / cfg.run_name / f"eval_phase3_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
