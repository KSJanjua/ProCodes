"""Phase 1 evaluation: paper Table 2/3 metrics (RMS/REL/RMSlog/Log10/
sigma1-3) on the held-out test split, plus optional disparity diagnostics
(plan SS8 -- never affects training, purely informational).

Usage (on the Backend.AI server):

    python -m instancedepth.engine.evaluate_hdi \\
        --config instancedepth/configs/hdi.yaml \\
        --checkpoint runs/hdi_faithful/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.config import HDIConfig
from instancedepth.engine.train_hdi import build_dataloader
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.metrics import compute_depth_metrics, compute_disparity_diagnostics

log = logging.getLogger("instancedepth.engine.evaluate_hdi")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@torch.no_grad()
def evaluate(
    model: HolisticDepthModel,
    loader: DataLoader,
    device: torch.device,
    cfg: HDIConfig,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    disp_totals: Dict[str, float] = {}
    n_batches = 0
    n_disp_batches = 0

    # Match Trainer.train_step's precision exactly (trainer.py's
    # _autocast_ctx) -- without this, the forward pass here runs in full
    # fp32 while training runs it in bf16/fp16, roughly doubling this
    # model's activation memory for the same batch size/resolution. That
    # mismatch is what OOM'd on the very first periodic eval call even
    # though training itself had already run thousands of steps cleanly.
    precision = cfg.optim.precision
    use_autocast = precision != "fp32" and device.type == "cuda"

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        with torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            out = model(image)
        mask = gt_depth > 0

        m = compute_depth_metrics(out.depth_final, gt_depth, mask)
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        disp_m = compute_disparity_diagnostics(out.depth_final, gt_depth, mask, cfg.camera)
        if disp_m is not None:
            for k, v in disp_m.items():
                disp_totals[k] = disp_totals.get(k, 0.0) + v
            n_disp_batches += 1

    result = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if n_disp_batches:
        result.update({k: v / n_disp_batches for k, v in disp_totals.items()})
    else:
        log.info("Camera intrinsics not set (instancedepth/configs/*.yaml -> camera.*) -- "
                  "skipping disparity diagnostics (this never blocks the standard metrics above).")
    return result


def make_eval_fn(cfg: HDIConfig, max_batches: int = 50):
    """Factory used by ``Trainer`` for periodic in-training eval (a cheap
    subset, not the full held-out set -- see plan SS16 'eval cadence')."""
    val_loader = build_dataloader(cfg, split="test")

    def eval_fn(model: torch.nn.Module) -> Dict[str, float]:
        device = next(model.parameters()).device
        return evaluate(model, val_loader, device, cfg, max_batches=max_batches)

    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-batches", type=int, default=None, help="default: evaluate the full split")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = HDIConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HolisticDepthModel(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)

    loader = build_dataloader(cfg, split=args.split)
    metrics = evaluate(model, loader, device, cfg, max_batches=args.max_batches)

    log.info("Evaluation results (%s split, %s):\n%s", args.split, args.checkpoint, json.dumps(metrics, indent=2))
    out_path = Path(cfg.run_root) / cfg.run_name / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
