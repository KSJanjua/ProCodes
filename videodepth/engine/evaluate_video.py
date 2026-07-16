"""Streaming video evaluation: sequence-ordered, stateful, per-frame accuracy
+ temporal consistency in one pass.

Metrics:
  * abs_rel / rms / sigma1-3 …    — standard dense metrics (paper Table 2/3)
  * temporal_alignment_error (TAE) — mean |Δpred − Δgt|: the flicker measure
    (0 = prediction tracks GT's motion perfectly; needs no optical flow since
    this data has per-frame metric GT)
  * gt/pred_temporal_delta         — TAE's context (see AUDIT_2026.md §1.1)
  * flicker_ratio = pred_delta/gt_delta — >1 means the prediction moves more
    than the scene does: pure flicker.

Works with any model exposing ``reset_temporal_state()`` and returning either
a depth tensor or an object with ``.depth_final`` — i.e. both the new
VideoDepthModel and the original per-frame HolisticDepthModel (its numbers
are the baseline the temporal head must beat).

Usage:
    python -m videodepth.engine.evaluate_video \\
        --config videodepth/configs/video_temporal.yaml \\
        --checkpoint runs/video_temporal/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from instancedepth.utils.metrics import compute_depth_metrics, temporal_alignment_error

log = logging.getLogger("videodepth.engine.evaluate_video")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _depth_of(out) -> torch.Tensor:
    return out if torch.is_tensor(out) else out.depth_final


@torch.no_grad()
def evaluate_streaming(model: torch.nn.Module, hdi_cfg, device: torch.device,
                       split: str = "test", max_frames: Optional[int] = None,
                       precision: str = "bf16") -> Dict[str, float]:
    from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset

    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=hdi_cfg.data.annotations_root, split=split,
        image_size=hdi_cfg.data.image_size,
        min_instance_px=hdi_cfg.data.min_instance_px,
        hflip_prob=0.0, color_jitter=0.0,
    ))
    model.eval()
    use_autocast = precision != "fp32" and device.type == "cuda"

    totals: Dict[str, float] = {}
    n = 0
    tae_sum = gt_d_sum = pred_d_sum = 0.0
    tae_n = 0
    prev_seq, prev_pred, prev_gt = None, None, None

    limit = len(ds) if max_frames is None else min(len(ds), max_frames)
    for i in range(limit):
        sample = ds[i]
        seq = sample["meta"]["sequence"]
        if seq != prev_seq:
            if hasattr(model, "reset_temporal_state"):
                model.reset_temporal_state()
            prev_seq, prev_pred, prev_gt = seq, None, None

        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["depth"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type,
                            dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            pred = _depth_of(model(image)).float()
        mask = gt > 0

        for k, v in compute_depth_metrics(pred, gt, mask).items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

        if prev_pred is not None:
            both = mask & (prev_gt > 0)
            tae = temporal_alignment_error(pred, prev_pred, gt, prev_gt, both)
            if tae == tae:
                tae_sum += tae
                tae_n += 1
                gt_d_sum += float((gt - prev_gt).abs()[both].mean())
                pred_d_sum += float((pred - prev_pred).abs()[both].mean())
        prev_pred, prev_gt = pred, gt

    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["temporal_alignment_error"] = tae_sum / max(tae_n, 1)
    out["gt_temporal_delta"] = gt_d_sum / max(tae_n, 1)
    out["pred_temporal_delta"] = pred_d_sum / max(tae_n, 1)
    out["flicker_ratio"] = out["pred_temporal_delta"] / max(out["gt_temporal_delta"], 1e-9)
    out["num_frames"] = n
    return out


def main() -> None:
    from instancedepth.configs.config import HDIConfig
    from instancedepth.utils.checkpoint import load_checkpoint
    from videodepth.configs.config import VideoConfig
    from videodepth.models.video_model import VideoDepthModel

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = VideoConfig.from_yaml_with_overrides(args.config, args.override)
    hdi_cfg = HDIConfig.from_yaml(cfg.hdi_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.init_checkpoint = None   # the full video checkpoint supersedes the init
    model = VideoDepthModel.from_config(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)

    metrics = evaluate_streaming(model, hdi_cfg, device, split=args.split,
                                 max_frames=args.max_frames,
                                 precision=cfg.optim.precision)
    log.info("Streaming eval (%s, %s):\n%s", args.split, args.checkpoint,
             json.dumps(metrics, indent=2))
    out_path = Path(cfg.run_root) / cfg.run_name / f"eval_streaming_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
