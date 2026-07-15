"""Run the full inference pipeline on an ARBITRARY real-world video (e.g.
downloaded from the internet) and produce a side-by-side comparison video:

    [ original RGB | predicted depth | instance overlay ]

The instance panel (Phase 3 only; suppress with --no-instances) shows the
instance branch's predicted masks over the original frame, each labelled with
its predicted depth layer Dep_i in metres.

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

Input handling is robust to OpenCV builds without FFmpeg decoding (the same
builds whose VideoWriter needs the PNG fallback): ``--video`` accepts a video
file (decoded by cv2 when possible, else extracted via the system ``ffmpeg``
binary) or a DIRECTORY of image frames (the universal escape hatch -- extract
frames yourself with any tool and point --video at the folder).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from instancedepth.predict import build_scene_predictor
from instancedepth.utils.viz import (
    colorize_depth, draw_instances_with_depth, open_video_writer, put_label,
)

log = logging.getLogger("scripts.infer_video")

_FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def colorize(depth: np.ndarray, mode: str, max_depth: float) -> np.ndarray:
    if mode == "metric":
        return colorize_depth(depth, max_depth)
    lo, hi = np.percentile(depth, [2, 98])
    span = max(hi - lo, 1e-6)
    return colorize_depth(np.clip(depth - lo, 0, None) + 1e-3, span)   # per-frame relative


def _iter_image_dir(files) -> Iterator[np.ndarray]:
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("skipping unreadable frame %s", f)
            continue
        yield img


def _probe_fps(video: Path, fallback: float) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return fallback
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except Exception:
        return fallback


def open_frame_source(video: str, fps_fallback: float) -> Tuple[Iterator[np.ndarray], float, Optional[int]]:
    """Return (BGR frame iterator, fps, total frame count or None) for any of:
    a directory of image frames; a video cv2 can decode; a video cv2 CANNOT
    decode (FFmpeg-less OpenCV build) but the system ffmpeg binary can, in
    which case frames are extracted to a temporary directory first."""
    p = Path(video)
    if not p.exists():
        raise FileNotFoundError(f"input '{video}' does not exist (cwd: {Path.cwd()})")

    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in _FRAME_EXTS)
        if not files:
            raise IOError(f"'{video}' is a directory but contains no image frames ({sorted(_FRAME_EXTS)})")
        log.info("frame-directory input: %d images from %s (fps=%g)", len(files), p, fps_fallback)
        return _iter_image_dir(files), fps_fallback, len(files)

    cap = cv2.VideoCapture(str(p))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        def _iter_cap() -> Iterator[np.ndarray]:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()

        return _iter_cap(), fps, total
    cap.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise IOError(
            f"cv2.VideoCapture could not open '{video}' and no system ffmpeg binary is "
            "available -- this OpenCV build likely lacks FFmpeg decode support (check "
            "cv2.getBuildInformation()). Workarounds: extract frames with any tool and "
            "pass the DIRECTORY of images as --video, or install ffmpeg on this machine."
        )
    tmpdir = Path(tempfile.mkdtemp(prefix="infer_video_frames_"))
    log.warning("cv2 cannot decode '%s' -- extracting frames via system ffmpeg to %s", video, tmpdir)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(p), "-qscale:v", "2",
         str(tmpdir / "frame_%06d.jpg")],
        check=True,
    )
    files = sorted(tmpdir.glob("frame_*.jpg"))
    if not files:
        raise IOError(f"ffmpeg extracted no frames from '{video}'")
    fps = _probe_fps(p, fps_fallback)
    log.info("ffmpeg fallback: %d frames extracted (fps=%.2f)", len(files), fps)
    return _iter_image_dir(files), fps, len(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 3], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--video", required=True,
                    help="input video file, or a DIRECTORY of image frames")
    ap.add_argument("--out", required=True, help="output path (extension chosen by codec)")
    ap.add_argument("--normalize", choices=["metric", "percentile"], default="metric",
                    help="depth colorization range: fixed metric [0,max_depth] or per-frame percentile")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="fps fallback when the source doesn't carry one (frame directories)")
    ap.add_argument("--no-instances", action="store_true",
                    help="drop the instance-overlay panel (Phase 3; Phase 1 has no instance branch)")
    ap.add_argument("--inst-score-thresh", type=float, default=0.5,
                    help="category-confidence cut for the instance overlay")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    show_instances = args.phase == 3 and not args.no_instances
    predict, max_depth = build_scene_predictor(
        args.phase, args.config, args.checkpoint, args.override,
        inst_score_thresh=args.inst_score_thresh,
    )

    frames, src_fps, total = open_frame_source(args.video, fps_fallback=args.fps)
    out_fps = src_fps / max(args.stride, 1)
    log.info("input: %s (%.2f fps, %s frames); output fps %.2f",
             args.video, src_fps, total if total is not None else "?", out_fps)

    writer = None
    actual_path = None
    idx = written = 0
    for bgr in frames:
        if idx % args.stride != 0:
            idx += 1
            continue
        idx += 1

        H, W = bgr.shape[:2]
        pred = predict(bgr)
        depth = pred["depth"]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        label = "Phase-3 refined" if args.phase == 3 else "Phase-1 depth"
        panels = [put_label(bgr, "RGB"),
                  put_label(colorize(depth, args.normalize, max_depth), label)]
        if show_instances:
            overlay = draw_instances_with_depth(bgr, pred["masks"], pred["mask_depths"])
            panels.append(put_label(overlay, f"instances ({len(pred['masks'])}, Dep_i)"))
        canvas = np.hstack(panels)
        if writer is None:
            writer, actual_path = open_video_writer(Path(args.out), out_fps,
                                                    (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        written += 1
        if written % 50 == 0:
            log.info("  %d frames written (source frame %d/%s)", written, idx,
                     total if total is not None else "?")
        if args.max_frames and written >= args.max_frames:
            break

    if writer is None:
        raise RuntimeError("no frames were processed -- check the input video / --stride")
    writer.release()
    log.info("Wrote %d frames to %s", written, actual_path)


if __name__ == "__main__":
    main()
