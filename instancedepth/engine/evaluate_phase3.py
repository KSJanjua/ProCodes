"""Phase 3 evaluation. Dense depth metrics (paper Table 2/3
protocol -- REL, RMS, RMSlog, Log10, sigma1-3) on the **refined** map vs GT,
reported three ways so the Table-4 ablation logic (H vs H+I) is reproducible:

  * overall_refined / overall_base : whole-frame, all frames.
  * occ_refined     / occ_base     : instance-region pixels, on the
    occlusion slice (frames with >=2 overlapping GT instances) -- the
    decisive signal for whether the refinement helps where it should.

``base`` = Phase-1 depth (pre-refinement); ``refined`` = composited output.
The gap between them is Phase 3's contribution.

Usage:

    python -m instancedepth.engine.evaluate_phase3 \\
        --config instancedepth/configs/phase3.yaml \\
        --checkpoint runs/phase3_refine/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.engine.train_phase3 import build_dataloader
from instancedepth.models.phase3.model import Phase3Model
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.metrics import compute_depth_metrics
from instancedepth.utils.phase2_metrics import has_overlapping_instances

log = logging.getLogger("instancedepth.engine.evaluate_phase3")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_METRIC_KEYS = ("abs_rel", "rms", "rms_log", "log10", "sigma1", "sigma2", "sigma3")


def _mean_metrics(frames: List[Dict[str, float]]) -> Dict[str, float]:
    if not frames:
        return {k: float("nan") for k in _METRIC_KEYS} | {"num_frames": 0}
    out = {}
    for k in _METRIC_KEYS:
        vals = [f[k] for f in frames if f[k] == f[k]]   # drop NaNs
        out[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    out["num_frames"] = len(frames)
    return out


@torch.no_grad()
def evaluate(
    model: Phase3Model,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    precision: str = "fp32",
) -> Dict[str, Dict[str, float]]:
    model.eval()
    use_autocast = precision != "fp32" and device.type == "cuda"
    overall_ref, overall_base, occ_ref, occ_base = [], [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        with torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            output, _ = model(image)
        refined = output.refined_depth.float()
        base = output.base_depth.float()

        for b in range(image.shape[0]):
            gt = gt_depth[b:b + 1]
            valid = gt > 0
            overall_ref.append(compute_depth_metrics(refined[b:b + 1], gt, valid))
            overall_base.append(compute_depth_metrics(base[b:b + 1], gt, valid))

            tgt = batch["targets"][b]
            gt_masks = tgt["masks"].bool().to(device)
            if gt_masks.shape[0] >= 2 and has_overlapping_instances(gt_masks):
                inst = gt_masks.any(0)[None, None]               # (1,1,H,W) union of GT instances
                inst_valid = valid & inst
                occ_ref.append(compute_depth_metrics(refined[b:b + 1], gt, inst_valid))
                occ_base.append(compute_depth_metrics(base[b:b + 1], gt, inst_valid))

    return dict(
        overall_refined=_mean_metrics(overall_ref),
        overall_base=_mean_metrics(overall_base),
        occ_refined=_mean_metrics(occ_ref),
        occ_base=_mean_metrics(occ_base),
    )


def make_eval_fn(cfg: Phase3Config, max_batches: int = 50):
    val_loader = build_dataloader(cfg, split="test")

    def eval_fn(model: torch.nn.Module) -> Dict[str, float]:
        device = next(model.parameters()).device
        result = evaluate(model, val_loader, device, max_batches=max_batches, precision=cfg.optim.precision)
        flat: Dict[str, float] = {}
        for group, metrics in result.items():
            flat.update({f"{group}_{k}": v for k, v in metrics.items()})
        # Trainer selects "best" on abs_rel (lower better): the refined whole-frame REL.
        flat["abs_rel"] = result["overall_refined"]["abs_rel"]
        return flat

    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = Phase3Config.from_yaml_with_overrides(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Phase3Model(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)

    loader = build_dataloader(cfg, split=args.split)
    result = evaluate(model, loader, device, args.max_batches, precision=cfg.optim.precision)

    log.info("Phase 3 eval (%s, %s):\n%s", args.split, args.checkpoint, json.dumps(result, indent=2))
    out_path = Path(cfg.run_root) / cfg.run_name / f"eval_phase3_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
