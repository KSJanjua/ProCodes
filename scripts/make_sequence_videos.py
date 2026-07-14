"""Turn held-out test sequences into side-by-side comparison videos:

    [ RGB | (optional GT depth) | predicted depth | (optional instances) ]

one video per sequence, for qualitative evaluation. Works with either the
Phase-1 (holistic) or the Phase-3 (occlusion-refined) model.

The instances panel overlays each instance mask on the RGB frame with its
depth layer (metres) printed at the mask centroid, drawn far-to-near so
occlusion ordering reads correctly (same convention as visualize_phase2.py).
``--instances pred`` uses the frozen Phase-2 branch's predictions (Phase 3
only -- Phase 1 is holistic and predicts no instances); ``--instances gt``
uses the dataset's id-map + GT depth layers; ``auto`` (default) picks pred
for Phase 3 and gt for Phase 1; ``none`` disables the panel.

Usage (server, project root):

    # Phase 3 refined depth + predicted instances:
    python -m scripts.make_sequence_videos \\
        --phase 3 --config instancedepth/configs/phase3_current.yaml \\
        --checkpoint runs/phase3_current/best.pth \\
        --out-dir videos/phase3_current --include-gt --limit-seq 4

    # Phase 1 + GT instances:
    python -m scripts.make_sequence_videos \\
        --phase 1 --config instancedepth/configs/hdi_enhanced.yaml \\
        --checkpoint runs/hdi_enhanced/best.pth \\
        --out-dir videos/hdi_enhanced --include-gt

Depth colorization: TURBO, near = warm, far = cool, invalid GT = black, fixed
to [0, max_depth] so colors are comparable across frames and sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from instancedepth.data.gid_dataset import GIDInstanceDepthDataset
from instancedepth.predict import build_scene_predictor
from instancedepth.utils.viz import (
    colorize_depth, draw_instances_with_depth, open_video_writer, put_label,
)

log = logging.getLogger("scripts.make_sequence_videos")


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read rgb {path}")
    return img   # BGR


def _gt_instances(frame: dict, hw: Tuple[int, int]) -> Tuple[List[np.ndarray], List[float]]:
    """Manifest frame -> (masks, depth layers) from the id-map annotation.
    Instances without a valid GT depth layer are skipped (no meaningful
    label to print)."""
    id_map = cv2.imread(frame["object_mask"], cv2.IMREAD_UNCHANGED)
    if id_map is None:
        return [], []
    if id_map.shape[:2] != hw:
        id_map = cv2.resize(id_map, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    masks, depths = [], []
    for inst in frame.get("instances", []):
        d = float(inst.get("depth_layer_m", 0.0))
        if d <= 0.0:
            continue
        m = id_map == inst["track_id"]
        if m.any():
            masks.append(m)
            depths.append(d)
    return masks, depths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 3], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--annotations-root", default=None,
                    help="default: the config's data.annotations_root")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--include-gt", action="store_true", help="add a GT depth middle panel")
    ap.add_argument("--instances", choices=["auto", "pred", "gt", "none"], default="auto",
                    help="instance-mask panel source: pred (Phase-3's instance branch), "
                         "gt (dataset id-map), auto (pred for phase 3, gt for phase 1), none")
    ap.add_argument("--inst-score-thresh", type=float, default=0.5,
                    help="category-confidence cut for predicted instances (viz-oriented, "
                         "looser than Phase 3's 0.9 candidate filter)")
    ap.add_argument("--limit-seq", type=int, default=None, help="only the first N sequences")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    inst_mode = args.instances
    if inst_mode == "auto":
        inst_mode = "pred" if args.phase == 3 else "gt"
    if inst_mode == "pred" and args.phase == 1:
        log.warning("--instances pred is unavailable for phase 1 (holistic-only); using GT instances")
        inst_mode = "gt"

    predict, max_depth = build_scene_predictor(
        args.phase, args.config, args.checkpoint, args.override,
        inst_score_thresh=args.inst_score_thresh,
    )

    if args.annotations_root is None:
        import yaml
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        args.annotations_root = raw.get("data", {}).get("annotations_root", "gid_custom")

    ann_root = Path(args.annotations_root)
    seq_ids = [s for s in (ann_root / f"{args.split}.txt").read_text().splitlines() if s.strip()]
    if args.limit_seq:
        seq_ids = seq_ids[: args.limit_seq]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for si, sid in enumerate(seq_ids):
        with open(ann_root / sid / "annotations.json") as f:
            man = json.load(f)
        frame_keys = sorted(man["frames"].keys())[:: args.stride]
        log.info("[%d/%d] %s (%d frames)", si + 1, len(seq_ids), sid, len(frame_keys))

        writer = None
        actual_path = None
        for fi, fk in enumerate(frame_keys):
            frame = man["frames"][fk]
            bgr = _load_rgb(frame["rgb"])
            H, W = bgr.shape[:2]

            pred = predict(bgr)
            depth = pred["depth"]
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            panels = [put_label(bgr, "RGB")]
            if args.include_gt:
                gt = GIDInstanceDepthDataset._load_depth(frame, man["depth_scale_to_m"])
                if gt.shape != (H, W):
                    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
                panels.append(put_label(colorize_depth(gt, max_depth), "GT depth"))
            label = "Phase-3 refined" if args.phase == 3 else "Phase-1 depth"
            panels.append(put_label(colorize_depth(depth, max_depth), label))
            if inst_mode != "none":
                if inst_mode == "pred":
                    masks, layer_depths = pred["masks"], pred["mask_depths"]
                    inst_label = f"instances (pred Dep, {len(masks)})"
                else:
                    masks, layer_depths = _gt_instances(frame, (H, W))
                    inst_label = f"instances (GT, {len(masks)})"
                panels.append(put_label(draw_instances_with_depth(bgr, masks, layer_depths), inst_label))

            canvas = np.hstack(panels)
            if writer is None:
                writer, actual_path = open_video_writer(
                    out_dir / sid.replace("/", "__"), args.fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if (fi + 1) % 100 == 0:
                log.info("  %d/%d frames", fi + 1, len(frame_keys))
        if writer is not None:
            writer.release()
            log.info("  wrote %s", actual_path)

    log.info("Done: %d sequence videos in %s", len(seq_ids), out_dir)


if __name__ == "__main__":
    main()
