"""Dataset diagnostics for Phase 1, run once before any training.

Two independent checks:

1. Depth-range statistics over the *full* (not just tracked-instance) GT
   depth maps in the training split -- informs whether MAX_d/rd (currently
   inherited from instancedepth/configs/gid_custom.yaml) are still
   reasonable once real full-scene depth is examined, not just the
   instance-level histogram already in meta.json.
2. A spot-check that `left_filled` (16-bit PNG) and `left_filled_np`
   (float32 .npy) decode to the same depth for a sample of real frames --
   `GIDInstanceDepthDataset`/`depth_io.py` prefer the .npy source
   (`prefer_npy=True`); this check turns that preference from a reasoned
   assumption into a verified fact (or surfaces a real discrepancy) before
   it silently affects training.

Usage (on the machine where the annotated dataset actually lives):

    python -m scripts.precompute_bins \\
        --annotations-root gid_custom \\
        --max-depth 10.0 \\
        --sample-frames 200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger("precompute_bins")


def _iter_sample_frames(annotations_root: Path, split: str, n: int) -> List[Tuple[dict, str]]:
    seq_ids = [s for s in (annotations_root / f"{split}.txt").read_text().splitlines() if s.strip()]
    out: List[Tuple[dict, str]] = []
    rng = np.random.default_rng(2026)
    rng.shuffle(seq_ids)
    for sid in seq_ids:
        man_path = annotations_root / sid / "annotations.json"
        if not man_path.exists():
            continue
        with open(man_path) as f:
            man = json.load(f)
        frame_keys = list(man["frames"].keys())
        rng.shuffle(frame_keys)
        for fk in frame_keys:
            out.append((man, fk))
            if len(out) >= n:
                return out
    return out


def _load_depth_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _load_depth_png(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"failed to read {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def spot_check_npy_vs_png(samples: List[Tuple[dict, str]], depth_scale_key: str = "depth_scale_to_m") -> dict:
    """Compare npy vs png decoded depth (both scaled to meters) on real
    frames where both files exist. Returns summary stats; does not raise --
    the caller decides whether the discrepancy is acceptable."""
    diffs = []
    checked = 0
    for man, fk in samples:
        frame = man["frames"][fk]
        if not (frame.get("depth_npy") and frame.get("depth_png")):
            continue
        scale = man.get(depth_scale_key, 1.0)
        d_npy = _load_depth_npy(frame["depth_npy"]) * scale
        d_png = _load_depth_png(frame["depth_png"]) * scale
        if d_npy.shape != d_png.shape:
            d_png = cv2.resize(d_png, (d_npy.shape[1], d_npy.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = (d_npy > 0) & (d_png > 0)
        if valid.sum() == 0:
            continue
        diffs.append(np.abs(d_npy[valid] - d_png[valid]))
        checked += 1
    if not diffs:
        return dict(checked_frames=0, note="no frame had both depth_npy and depth_png -- nothing to compare")
    all_diffs = np.concatenate(diffs)
    return dict(
        checked_frames=checked,
        mean_abs_diff_m=float(all_diffs.mean()),
        p95_abs_diff_m=float(np.percentile(all_diffs, 95)),
        max_abs_diff_m=float(all_diffs.max()),
    )


def depth_histogram(samples: List[Tuple[dict, str]], max_depth: float, prefer_npy: bool = True) -> dict:
    """Full-scene (not just tracked-instance) depth histogram, bucketed by
    meter, over the sampled frames."""
    buckets = np.zeros(int(np.ceil(max_depth)) + 1, dtype=np.int64)
    n_pixels = 0
    for man, fk in samples:
        frame = man["frames"][fk]
        scale = man.get("depth_scale_to_m", 1.0)
        if prefer_npy and frame.get("depth_npy"):
            d = _load_depth_npy(frame["depth_npy"]) * scale
        elif frame.get("depth_png"):
            d = _load_depth_png(frame["depth_png"]) * scale
        else:
            continue
        valid = d[(d > 0) & (d <= max_depth) & np.isfinite(d)]
        idx = np.clip(valid.astype(np.int64), 0, len(buckets) - 1)
        np.add.at(buckets, idx, 1)
        n_pixels += valid.size
    return dict(
        total_valid_pixels=int(n_pixels),
        per_meter_bucket=buckets.tolist(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations-root", required=True, help="output.out_root of the data engine, e.g. gid_custom")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--sample-frames", type=int, default=200)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    root = Path(args.annotations_root)
    samples = _iter_sample_frames(root, args.split, args.sample_frames)
    log.info("Sampled %d frames from split=%s under %s", len(samples), args.split, root)

    log.info("Running npy-vs-png depth spot-check ...")
    consistency = spot_check_npy_vs_png(samples)
    log.info("npy-vs-png consistency: %s", json.dumps(consistency, indent=2))
    if consistency.get("checked_frames", 0) and consistency["max_abs_diff_m"] > 0.05:
        log.warning(
            "max_abs_diff_m=%.4f exceeds 5cm -- .npy and .png sources disagree "
            "more than expected. Investigate before trusting prefer_npy=True "
            "(instancedepth/data_engine/config.py DepthConfig).",
            consistency["max_abs_diff_m"],
        )

    log.info("Computing full-scene depth histogram (max_depth=%.1f) ...", args.max_depth)
    hist = depth_histogram(samples, args.max_depth)
    log.info("total_valid_pixels=%d", hist["total_valid_pixels"])
    for m, count in enumerate(hist["per_meter_bucket"]):
        if count:
            log.info("  [%d-%dm): %d px (%.2f%%)", m, m + 1, count, 100 * count / max(hist["total_valid_pixels"], 1))

    out = dict(consistency=consistency, histogram=hist, args=vars(args))
    out_path = root / "precompute_bins_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
