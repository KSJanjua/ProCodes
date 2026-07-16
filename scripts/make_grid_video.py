"""Per-sequence 2x3 grid comparison video:

    row 1:  RGB | Phase-1 depth | GT depth
    row 2:  RGB | predicted instances (Phase 2) | GT instances

one video per test sequence. Phase 1 is holistic (no instances) and Phase 2
predicts no dense depth, so the two models are run together -- Phase 1 fills
the depth panel, Phase 2 the predicted-instance panel. Ground truth (depth +
id-map instances) comes from the annotations. Use hdi_temporal for the
Phase-1 args to see the temporal model's depth.

Usage (server, project root):

    python -m scripts.make_grid_video \\
        --phase1-config instancedepth/configs/hdi_enhanced.yaml \\
        --phase1-checkpoint runs/hdi_enhanced/best.pth \\
        --phase2-config instancedepth/configs/phase2_mask2former.yaml \\
        --phase2-checkpoint runs/phase2_run/best.pth \\
        --out-dir videos/grid --limit-seq 4 --track-instances

Every panel is colorized / drawn exactly as make_sequence_videos does
(shared helpers), so a grid cell matches its single-strip counterpart.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from instancedepth.data.gid_dataset import GIDInstanceDepthDataset
from instancedepth.predict import build_scene_predictor
from instancedepth.utils.viz import (
    MaskTracker, colorize_depth, draw_instances_with_depth, open_video_writer, put_label,
)
# reuse the GT-instance reader / rgb loader rather than duplicate them
from scripts.make_sequence_videos import _gt_instances, _load_rgb

log = logging.getLogger("scripts.make_grid_video")


def compose_grid(rgb: np.ndarray, p1_depth: np.ndarray, gt_depth: np.ndarray,
                 pred_inst: np.ndarray, gt_inst: np.ndarray) -> np.ndarray:
    """Assemble the labelled 2x3 grid. All five inputs are BGR uint8 at the
    same (H,W); RGB is repeated as the left column of each row for anchoring.
    Returns a (2H, 3W, 3) canvas."""
    top = np.hstack([put_label(rgb, "RGB"),
                     put_label(p1_depth, "Phase-1 depth"),
                     put_label(gt_depth, "GT depth")])
    bottom = np.hstack([put_label(rgb, "RGB"),
                        put_label(pred_inst, "pred instances (Dep_i)"),
                        put_label(gt_inst, "GT instances")])
    return np.vstack([top, bottom])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-config", required=True)
    ap.add_argument("--phase1-checkpoint", required=True)
    ap.add_argument("--phase1-override", nargs="*", default=[])
    ap.add_argument("--phase2-config", required=True)
    ap.add_argument("--phase2-checkpoint", required=True)
    ap.add_argument("--phase2-override", nargs="*", default=[])
    ap.add_argument("--annotations-root", default=None,
                    help="default: the Phase-1 config's data.annotations_root")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--inst-score-thresh", type=float, default=0.5,
                    help="category-confidence cut for predicted instances")
    ap.add_argument("--track-instances", action="store_true",
                    help="stabilize predicted-instance identities/colours across frames (IoU tracker)")
    ap.add_argument("--no-contours", action="store_true", help="don't outline instance masks")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="resize each cell by this factor before assembly (default 0.5 -- the grid "
                         "is 2x3 full frames, which is large; halving keeps files manageable)")
    ap.add_argument("--limit-seq", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    depth_predict, max_depth = build_scene_predictor(
        1, args.phase1_config, args.phase1_checkpoint, args.phase1_override)
    inst_predict, _ = build_scene_predictor(
        2, args.phase2_config, args.phase2_checkpoint, args.phase2_override,
        inst_score_thresh=args.inst_score_thresh)

    if args.annotations_root is None:
        import yaml
        with open(args.phase1_config) as f:
            args.annotations_root = (yaml.safe_load(f).get("data", {})
                                     .get("annotations_root", "gid_custom"))
    ann_root = Path(args.annotations_root)
    seq_ids = [s for s in (ann_root / f"{args.split}.txt").read_text().splitlines() if s.strip()]
    if args.limit_seq:
        seq_ids = seq_ids[: args.limit_seq]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = MaskTracker() if args.track_instances else None

    def _resize(img: np.ndarray, hw) -> np.ndarray:
        return img if img.shape[:2] == hw else cv2.resize(img, (hw[1], hw[0]),
                                                          interpolation=cv2.INTER_LINEAR)

    for si, sid in enumerate(seq_ids):
        with open(ann_root / sid / "annotations.json") as f:
            man = json.load(f)
        frame_keys = sorted(man["frames"].keys())[:: args.stride]
        log.info("[%d/%d] %s (%d frames)", si + 1, len(seq_ids), sid, len(frame_keys))
        depth_predict.reset()                    # temporal Phase-1: fresh memory per sequence
        if tracker is not None:
            tracker.reset()

        writer = actual_path = None
        for fi, fk in enumerate(frame_keys):
            frame = man["frames"][fk]
            bgr = _load_rgb(frame["rgb"])
            H, W = bgr.shape[:2]

            # --- top row: Phase-1 depth + GT depth ---
            depth = _resize(depth_predict(bgr)["depth"], (H, W))
            gt = GIDInstanceDepthDataset._load_depth(frame, man["depth_scale_to_m"])
            gt = gt if gt.shape == (H, W) else cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            p1_panel = colorize_depth(depth, max_depth, far_thresh=max_depth)
            gtd_panel = colorize_depth(gt, max_depth, far_thresh=max_depth)

            # --- bottom row: predicted + GT instances ---
            ip = inst_predict(bgr)
            pm, pd, pids = ip["masks"], ip["mask_depths"], ip.get("mask_ids")
            if tracker is not None:
                pm, pd, pids = tracker.update(pm, pd)
            pred_panel = draw_instances_with_depth(bgr, pm, pd, ids=pids,
                                                   draw_contour=not args.no_contours)
            gm, gd, gids = _gt_instances(frame, (H, W))
            gt_panel = draw_instances_with_depth(bgr, gm, gd, ids=gids,
                                                 draw_contour=not args.no_contours)

            grid = compose_grid(bgr, p1_panel, gtd_panel, pred_panel, gt_panel)
            if args.scale != 1.0:
                nw = max(int(grid.shape[1] * args.scale) // 2 * 2, 2)
                nh = max(int(grid.shape[0] * args.scale) // 2 * 2, 2)
                grid = cv2.resize(grid, (nw, nh), interpolation=cv2.INTER_AREA)

            if writer is None:
                writer, actual_path = open_video_writer(
                    out_dir / sid.replace("/", "__"), args.fps, (grid.shape[1], grid.shape[0]))
            writer.write(grid)
            if (fi + 1) % 100 == 0:
                log.info("  %d/%d frames", fi + 1, len(frame_keys))
        if writer is not None:
            writer.release()
            log.info("  wrote %s", actual_path)

    log.info("Done: %d grid videos in %s", len(seq_ids), out_dir)


if __name__ == "__main__":
    main()
