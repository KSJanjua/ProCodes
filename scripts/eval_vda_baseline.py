"""Zero-shot metric-depth baseline: Video Depth Anything (metric, ViT-L) on the
GID test split, scored with THIS repo's metrics so the numbers sit directly
beside the InstanceDepth results.

VDA is an external model with no loader in this repo. This bridge runs VDA's own
inference over each test sequence *in frame order* (exercising its temporal
head), resizes the metric-depth output to the GID eval resolution, and
accumulates the SAME per-frame + temporal metrics as
videodepth/engine/evaluate_video.py -- identical GT / mask / resolution to the
models you compare against.

Run from the repo root (needs an external Video-Depth-Anything clone; VDA is
not a dependency of this repo):

    python -m scripts.eval_vda_baseline \
        --vda-repo   /path/to/Video-Depth-Anything \
        --vda-ckpt   /path/to/metric_video_depth_anything_vitl.pth \
        --hdi-config instancedepth/configs/hdi_dav2.yaml \
        --split test

--hdi-config only supplies the dataset (annotations_root / image_size /
max_depth / min_instance_px), so use the SAME config whose test numbers you are
comparing against -- that guarantees byte-identical GT and masks.

--median-align rescales each frame's prediction by median(gt)/median(pred) over
valid pixels. That is the scale-invariant number, NOT a metric one; report it
separately (an off-domain "metric" model often needs it, so it is offered).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("eval_vda_baseline")

# ISOLATED VDA SURFACE -------------------------------------------------------
# The only VDA-version-specific code. If your clone differs, reconcile HERE:
# the import path, the class name, the metric config, and the inference call.
VITL_CFG = dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])


def load_vda(vda_repo: str, vda_ckpt: str, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(Path(vda_repo)))
    from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

    model = VideoDepthAnything(**VITL_CFG)
    sd = torch.load(vda_ckpt, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    model.load_state_dict(sd, strict=True)
    log.info("loaded VDA metric ViT-L from %s", vda_ckpt)
    return model.to(device).eval()


@torch.no_grad()
def vda_depth_for_clip(model: torch.nn.Module, frames_rgb: np.ndarray,
                       input_size: int, device: torch.device) -> np.ndarray:
    """frames_rgb (T,H,W,3) uint8 RGB -> (T,H,W) float32 metric depth (meters)."""
    depths, _ = model.infer_video_depth(
        frames_rgb, target_fps=-1, input_size=input_size, device=str(device))
    depths = np.asarray(depths, dtype=np.float32)
    return depths[..., 0] if depths.ndim == 4 else depths
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vda-repo", required=True, help="local clone of Video-Depth-Anything")
    ap.add_argument("--vda-ckpt", required=True, help="metric_video_depth_anything_vitl.pth")
    ap.add_argument("--hdi-config", required=True,
                    help="the config whose test numbers you compare against")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--input-size", type=int, default=518, help="VDA network input size")
    ap.add_argument("--median-align", action="store_true",
                    help="per-frame median rescale (scale-invariant, NOT metric)")
    ap.add_argument("--max-sequences", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--out", default=None, help="output JSON (default: results/...)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from instancedepth.configs.config import HDIConfig
    from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset
    from instancedepth.utils.metrics import compute_depth_metrics, temporal_alignment_error

    hdi = HDIConfig.from_yaml(args.hdi_config)
    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=hdi.data.annotations_root, split=args.split,
        image_size=hdi.data.image_size, max_depth=hdi.bins.max_depth,
        min_instance_px=hdi.data.min_instance_px, hflip_prob=0.0, color_jitter=0.0,
    ))
    H, W = hdi.data.image_size

    # ds.index is already sequence-grouped and frame-sorted.
    seqs: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, (man, fkey) in enumerate(ds.index):
        seqs.setdefault(man["sequence"], []).append(i)
    seq_items = list(seqs.items())
    if args.max_sequences:
        seq_items = seq_items[:args.max_sequences]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vda(args.vda_repo, args.vda_ckpt, device)

    totals: Dict[str, float] = {}
    n = 0
    tae_sum = gt_d_sum = pred_d_sum = 0.0
    tae_n = 0

    for si, (seq, idxs) in enumerate(seq_items):
        frames = np.stack([
            cv2.resize(ds._load_rgb(ds.index[i][0]["frames"][ds.index[i][1]]["rgb"]),
                       (W, H), interpolation=cv2.INTER_LINEAR)
            for i in idxs
        ])
        pred_seq = vda_depth_for_clip(model, frames, args.input_size, device)  # (T,H,W)

        if si == 0:
            # sanity: a metric model emits meters (~0.5-10 here); disparity would
            # be O(1e-3..1). If this looks like disparity, the ckpt is not metric.
            log.info("sanity median predicted depth (seq 0): %.3f", float(np.median(pred_seq)))

        prev_pred = prev_gt = None
        for t, i in enumerate(idxs):
            gt = ds[i]["depth"].to(device)                                   # (1,H,W)
            pred = torch.from_numpy(pred_seq[t])[None].to(device).float()    # (1,H,W)
            if pred.shape[-2:] != (H, W):
                pred = F.interpolate(pred[None], size=(H, W),
                                     mode="bilinear", align_corners=False)[0]
            mask = gt > 0
            if args.median_align and mask.any():
                pred = pred * (gt[mask].median() / pred[mask].clamp_min(1e-6).median())

            for k, v in compute_depth_metrics(pred, gt, mask).items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

            if prev_pred is not None:
                both = mask & (prev_gt > 0)
                tae = temporal_alignment_error(pred, prev_pred, gt, prev_gt, both)
                if tae == tae:                          # skip NaN (no shared valid px)
                    tae_sum += tae
                    tae_n += 1
                    gt_d_sum += float((gt - prev_gt).abs()[both].mean())
                    pred_d_sum += float((pred - prev_pred).abs()[both].mean())
            prev_pred, prev_gt = pred, gt

        log.info("[%d/%d] %s: %d frames (running frames=%d)",
                 si + 1, len(seq_items), seq, len(idxs), n)

    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["temporal_alignment_error"] = tae_sum / max(tae_n, 1)
    out["gt_temporal_delta"] = gt_d_sum / max(tae_n, 1)
    out["pred_temporal_delta"] = pred_d_sum / max(tae_n, 1)
    out["flicker_ratio"] = out["pred_temporal_delta"] / max(out["gt_temporal_delta"], 1e-9)
    out["num_frames"] = n
    out["num_sequences"] = len(seq_items)
    result = {
        "model": "video_depth_anything_metric_vitl",
        "eval_mode": "median-aligned (scale-invariant)" if args.median_align else "metric (as-is)",
        "checkpoint": args.vda_ckpt,
        "hdi_config": args.hdi_config,
        "split": args.split,
        "image_size": [H, W],
        "input_size": args.input_size,
        "metrics": out,
    }

    suffix = "_aligned" if args.median_align else ""
    out_path = Path(args.out) if args.out else Path("results") / f"vda_metric_baseline_{args.split}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("\n%s\nWrote %s", json.dumps(out, indent=2), out_path)


if __name__ == "__main__":
    main()
