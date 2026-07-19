"""Turn held-out test sequences into side-by-side comparison videos:

    [ RGB | (GT depth) | (predicted depth) | (pred instances) | (GT instances) ]

one video per sequence, for qualitative evaluation. Works with the Phase-1
(holistic depth), Phase-2 (instance branch) or Phase-3 (occlusion-refined
depth) model.

The instances panel overlays each instance mask on the RGB frame with its
depth layer (metres) printed at the mask centroid, drawn far-to-near so
occlusion ordering reads correctly (same convention as visualize_phase2.py).
``--instances pred`` uses the model's own instance branch (Phase 2/3 only --
Phase 1 is holistic and predicts none); ``gt`` uses the dataset's id-map + GT
depth layers; ``both`` draws the two panels side by side (the pred-vs-GT
comparison); ``auto`` (default) picks pred for Phase 2/3 and gt for Phase 1.

Phase 2 predicts no dense depth, so its videos carry no prediction-depth
panel. Note its Dep_i comes from an MLP on Mask2Former query embeddings and
never reads Phase 1, so Phase-2 output is identical regardless of which
Phase-1 checkpoint exists -- it cannot show a temporal-module difference.

Usage (server, project root):

    # Phase 3 refined depth + predicted instances:
    python -m scripts.make_sequence_videos \\
        --phase 3 --config instancedepth/configs/phase3_current.yaml \\
        --checkpoint runs/phase3_current/best.pth \\
        --out-dir videos/phase3_current --include-gt --limit-seq 4

    # Phase 2 instance branch, predicted vs GT instances side by side:
    python -m scripts.make_sequence_videos \\
        --phase 2 --config instancedepth/configs/phase2_mask2former.yaml \\
        --checkpoint runs/phase2_run/best.pth \\
        --out-dir videos/phase2 --include-gt --instances both

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


def _gt_instances(frame: dict, hw: Tuple[int, int]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """Manifest frame -> (masks, depth layers, track_ids) from the id-map.

    track_ids are returned so the overlay can colour each person consistently
    across the whole sequence -- the data engine enforces identity consistency,
    so GT colours are genuinely stable (unlike predictions, which have no
    temporal tracking). Instances without a valid GT depth layer are skipped
    (no meaningful label to print)."""
    id_map = cv2.imread(frame["object_mask"], cv2.IMREAD_UNCHANGED)
    if id_map is None:
        return [], [], []
    if id_map.shape[:2] != hw:
        id_map = cv2.resize(id_map, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    masks, depths, ids = [], [], []
    for inst in frame.get("instances", []):
        d = float(inst.get("depth_layer_m", 0.0))
        if d <= 0.0:
            continue
        m = id_map == inst["track_id"]
        if m.any():
            masks.append(m)
            depths.append(d)
            ids.append(int(inst["track_id"]))
    return masks, depths, ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], required=True,
                    help="1 = holistic depth, 2 = instance branch only (no dense depth panel), "
                         "3 = occlusion-refined depth")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--annotations-root", default=None,
                    help="default: the config's data.annotations_root")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--include-gt", action="store_true", help="add a GT depth middle panel")
    ap.add_argument("--depth-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="colorize depth over this fixed metric window (metres), e.g. "
                         "--depth-range 0 5 for a shallow scene, instead of the full 0-max_depth "
                         "range (which reads as one flat colour when the scene is much shallower). "
                         "Applies to both GT and prediction panels; visualization only.")
    ap.add_argument("--far-black", type=float, default=None, metavar="METRES",
                    help="render depth >= this many metres BLACK in both GT and prediction "
                         "panels (matching how GT looks beyond the sensor range). Default: "
                         "max_depth when no --depth-range is given, disabled with one -- pass "
                         "e.g. --far-black 10 to force it in either mode.")
    ap.add_argument("--instances", choices=["auto", "pred", "gt", "both", "none"], default="auto",
                    help="instance-mask panel source: pred (the instance branch's masks + Dep_i), "
                         "gt (dataset id-map + GT depth layers), both (two panels, side by side -- "
                         "the pred-vs-GT comparison), auto (pred for phase 2/3, gt for phase 1), none")
    ap.add_argument("--inst-score-thresh", type=float, default=0.5,
                    help="category-confidence cut for predicted instances (viz-oriented, "
                         "looser than Phase 3's 0.9 candidate filter)")
    ap.add_argument("--track-instances", action="store_true",
                    help="stabilize predicted-instance identities across frames with a lightweight "
                         "IoU tracker (persistent colours, suppresses one-frame flicker, bridges brief "
                         "dropouts). Reduces the 'colours keep changing' churn; GT instances are already "
                         "stable via track_id so this only affects predictions.")
    ap.add_argument("--no-contours", action="store_true",
                    help="don't outline instance masks in the overlay panel (cleaner video; "
                         "the translucent fill still shows each mask)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize the final stacked canvas by this factor before writing "
                         "(e.g. 0.5 halves file size; also helps picky encoders with very wide frames)")
    ap.add_argument("--limit-seq", type=int, default=None, help="only the first N sequences")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    inst_mode = args.instances
    if inst_mode == "auto":
        inst_mode = "pred" if args.phase in (2, 3) else "gt"
    if inst_mode in ("pred", "both") and args.phase == 1:
        log.warning("phase 1 is holistic-only and predicts no instances; using GT instances")
        inst_mode = "gt"
    if args.phase == 2 and inst_mode == "none":
        raise SystemExit("--phase 2 with --instances none would render nothing but RGB "
                         "(Phase 2 predicts no dense depth)")

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
    if args.track_instances and inst_mode in ("pred", "both"):
        from videodepth.models.query_tracker import QueryInstanceTracker
        tracker = QueryInstanceTracker()
    else:
        tracker = None

    for si, sid in enumerate(seq_ids):
        with open(ann_root / sid / "annotations.json") as f:
            man = json.load(f)
        frame_keys = sorted(man["frames"].keys())[:: args.stride]
        log.info("[%d/%d] %s (%d frames)", si + 1, len(seq_ids), sid, len(frame_keys))
        if hasattr(predict, "reset"):
            predict.reset()   # temporal models: fresh memory per sequence
        if tracker is not None:
            tracker.reset()   # fresh instance identities per sequence

        writer = None
        actual_path = None
        for fi, fk in enumerate(frame_keys):
            frame = man["frames"][fk]
            bgr = _load_rgb(frame["rgb"])
            H, W = bgr.shape[:2]

            pred = predict(bgr)
            panels = [put_label(bgr, "RGB")]
            # Fixed colour window: explicit --depth-range, else the full metric
            # range (GT-comparable). lo/hi and far_thresh chosen so both panels
            # share one mapping. Default far behaviour: black beyond max_depth
            # (GT-comparable) unless a custom window is given (clamp, don't
            # black); --far-black overrides the threshold in either mode.
            c_lo, c_hi = (args.depth_range if args.depth_range else (0.0, max_depth))
            c_far = args.far_black if args.far_black is not None else \
                (None if args.depth_range else max_depth)
            if args.include_gt:
                gt = GIDInstanceDepthDataset._load_depth(frame, man["depth_scale_to_m"])
                if gt.shape != (H, W):
                    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
                panels.append(put_label(colorize_depth(gt, c_hi, far_thresh=c_far, min_depth=c_lo), "GT depth"))

            # Phase 2 predicts instances only -- no dense-depth panel to draw.
            if pred["depth"] is not None:
                depth = pred["depth"]
                if depth.shape != (H, W):
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
                label = "Phase-3 refined" if args.phase == 3 else "Phase-1 depth"
                panels.append(put_label(colorize_depth(depth, c_hi, far_thresh=c_far, min_depth=c_lo), label))

            if inst_mode in ("pred", "both"):
                pm, pd, pids = pred["masks"], pred["mask_depths"], pred.get("mask_ids")
                if tracker is not None:                      # stabilize identities across frames
                    pm, pd, pids = tracker.update(pm, pd, pred.get("mask_embeds") or None)
                panels.append(put_label(
                    draw_instances_with_depth(bgr, pm, pd, ids=pids,
                                              draw_contour=not args.no_contours),
                    f"instances (pred Dep_i, {len(pm)})"))
            if inst_mode in ("gt", "both"):
                gt_masks, gt_layers, gt_ids = _gt_instances(frame, (H, W))
                panels.append(put_label(
                    draw_instances_with_depth(bgr, gt_masks, gt_layers, ids=gt_ids,
                                              draw_contour=not args.no_contours),
                    f"instances (GT, {len(gt_masks)})"))

            canvas = np.hstack(panels)
            if args.scale != 1.0:
                # even dimensions -- some encoders reject odd sizes
                nw = max(int(canvas.shape[1] * args.scale) // 2 * 2, 2)
                nh = max(int(canvas.shape[0] * args.scale) // 2 * 2, 2)
                canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)
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
