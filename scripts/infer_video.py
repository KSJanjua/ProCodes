"""Run the full inference pipeline on an ARBITRARY real-world video (e.g.
downloaded from the internet) and produce a side-by-side comparison video:

    [ original RGB | predicted depth ]

Works with either the Phase-1 (holistic) or Phase-3 (occlusion-refined) model.
This is the generalization check: the model was trained on a single indoor
RGB-D sensor with metric depth in (0, 10] m, so out-of-domain scenes (outdoor,
long sightlines) will saturate the metric range -- use
``--normalize percentile`` to colorize by each frame's own 2-98 percentile
range instead of the fixed metric range when inspecting such footage.

Usage (server, project root):

    python -m scripts.infer_video \\
        --phase 3 --config instancedepth/configs/phase3_current.yaml \\
        --checkpoint runs/phase3_current/best.pth \\
        --video downloads/street_dance.mp4 \\
        --out videos/street_dance_phase3

Frames are resized (plain resize, matching the dataset pipeline's own
convention) to the model's training resolution for inference, and the depth
map is resized back to the source resolution for display.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from instancedepth.utils.viz import colorize_depth, open_video_writer, put_label

log = logging.getLogger("scripts.infer_video")


def build_predict(phase: int, config: str, checkpoint: str, overrides):
    if phase == 1:
        from instancedepth.configs.config import HDIConfig
        from instancedepth.models.hdi.inference import HDIInferencer
        cfg = HDIConfig.from_yaml_with_overrides(config, overrides)
        inf = HDIInferencer(cfg, checkpoint)
        return (lambda bgr: inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                .depth_final[0, 0].float().cpu().numpy()), cfg.bins.max_depth

    from instancedepth.configs.phase3_config import Phase3Config
    from instancedepth.models.phase3.inference import Phase3Inferencer
    cfg = Phase3Config.from_yaml_with_overrides(config, overrides)
    inf = Phase3Inferencer(cfg, checkpoint)
    return (lambda bgr: inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["refined"]), cfg.data.max_depth


def colorize(depth: np.ndarray, mode: str, max_depth: float) -> np.ndarray:
    if mode == "metric":
        return colorize_depth(depth, max_depth)
    lo, hi = np.percentile(depth, [2, 98])
    span = max(hi - lo, 1e-6)
    return colorize_depth(np.clip(depth - lo, 0, None) + 1e-3, span)   # per-frame relative


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 3], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--video", required=True, help="input video file")
    ap.add_argument("--out", required=True, help="output path (extension chosen by codec)")
    ap.add_argument("--normalize", choices=["metric", "percentile"], default="metric",
                    help="depth colorization range: fixed metric [0,max_depth] or per-frame percentile")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    predict, max_depth = build_predict(args.phase, args.config, args.checkpoint, args.override)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"failed to open video {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = src_fps / max(args.stride, 1)
    log.info("input: %s (%.2f fps, %d frames); output fps %.2f", args.video, src_fps, total, out_fps)

    writer = None
    actual_path = None
    idx = written = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        idx += 1

        H, W = bgr.shape[:2]
        depth = predict(bgr)
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        label = "Phase-3 refined" if args.phase == 3 else "Phase-1 depth"
        canvas = np.hstack([put_label(bgr, "RGB"),
                            put_label(colorize(depth, args.normalize, max_depth), label)])
        if writer is None:
            writer, actual_path = open_video_writer(Path(args.out), out_fps,
                                                    (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        written += 1
        if written % 50 == 0:
            log.info("  %d frames written (source frame %d/%d)", written, idx, total)
        if args.max_frames and written >= args.max_frames:
            break

    cap.release()
    if writer is None:
        raise RuntimeError("no frames were processed -- check the input video / --stride")
    writer.release()
    log.info("Wrote %d frames to %s", written, actual_path)


if __name__ == "__main__":
    main()
