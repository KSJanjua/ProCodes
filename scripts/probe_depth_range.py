"""Print a random video's actual predicted-depth range -- no video output,
just numbers -- so you know what to pass ``infer_video.py --depth-range``.

Answers "how do I know what --depth-range to use on a video I've never run
before": rather than guess, run the model on a handful of sampled frames and
report the real percentiles. For a one-off, ``infer_video.py --normalize
global`` already does this automatically per-video; use THIS tool when you
want the actual numbers -- to reuse a fixed window across several similar
clips, to sanity-check the model isn't saturating (predictions piling up at
0 or max_depth), or to pick the window by hand.

Usage:
    python -m scripts.probe_depth_range \\
        --phase 3 --config videodepth/configs/phase3_dav2_p2run.yaml \\
        --checkpoint runs/phase3_video_dav2/best.pth \\
        --video random_clip.mp4

    # then, in the printed suggestion's own words:
    python -m scripts.infer_video --phase 3 ... --depth-range <LO> <HI>
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List

import numpy as np

from instancedepth.predict import build_depth_predictor
from scripts.infer_video import open_frame_source

log = logging.getLogger("scripts.probe_depth_range")


def summarize_depth_samples(samples: List[np.ndarray], max_depth: float) -> Dict[str, float]:
    """Pure numeric core (no I/O): valid-pixel arrays from sampled frames ->
    percentiles, saturation fractions, and a suggested --depth-range window.
    Separated from main() so this logic is unit-testable without a model."""
    all_v = np.concatenate(samples)
    p1, p2, p50, p98, p99 = (float(x) for x in np.percentile(all_v, [1, 2, 50, 98, 99]))
    lo_sat = float((all_v <= 0.02 * max_depth).mean())
    hi_sat = float((all_v >= 0.98 * max_depth).mean())
    lo, hi = max(0.0, p2 - 0.2), p98 + 0.2   # small margin so near/far pixels aren't clipped
    return dict(n_pixels=int(all_v.size), min=float(all_v.min()), p1=p1, p2=p2,
               median=p50, p98=p98, p99=p99, max=float(all_v.max()),
               lo_saturation=lo_sat, hi_saturation=hi_sat,
               suggested_lo=lo, suggested_hi=hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 3], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--video", required=True, help="video file, or a DIRECTORY of image frames")
    ap.add_argument("--num-frames", type=int, default=12,
                    help="frames sampled evenly across the video (default 12 -- plenty "
                         "for a stable percentile estimate without decoding the whole clip)")
    ap.add_argument("--fps", type=float, default=30.0, help="fps fallback for frame-dir input")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    predict, max_depth = build_depth_predictor(args.phase, args.config, args.checkpoint, args.override)
    frames, _, total = open_frame_source(args.video, fps_fallback=args.fps)

    # Evenly-spaced sample indices over the (possibly unknown-length) stream:
    # if total is known, pick exact indices; otherwise just take every Nth
    # frame up to num_frames seen, which is "evenly spaced enough" without a
    # second pass over the source.
    stride = max(total // args.num_frames, 1) if total else 1
    samples = []
    for i, bgr in enumerate(frames):
        if i % stride == 0:
            d = predict(bgr)
            v = d[d > 0]
            if v.size:
                samples.append(v)
            if len(samples) >= args.num_frames:
                break

    if not samples:
        raise SystemExit(f"no valid (>0) depth pixels found in the sampled frames "
                         f"of '{args.video}' -- check the model/checkpoint match this video's domain")

    r = summarize_depth_samples(samples, max_depth)
    print(f"\nSampled {len(samples)} frames ({r['n_pixels']} valid pixels) from '{args.video}'")
    print(f"  min={r['min']:.2f}  p1={r['p1']:.2f}  p2={r['p2']:.2f}  median={r['median']:.2f}  "
         f"p98={r['p98']:.2f}  p99={r['p99']:.2f}  max={r['max']:.2f}  (metres)")
    if r["lo_saturation"] > 0.05 or r["hi_saturation"] > 0.05:
        print(f"  [!] {r['lo_saturation']:.0%} of pixels sit within 2% of 0, "
             f"{r['hi_saturation']:.0%} within 2% of max_depth ({max_depth} m) -- the model "
             f"may be SATURATING on this footage (out of its trained domain), not just "
             f"needing a tighter colour window.")

    print(f"\nSuggested:  --depth-range {r['suggested_lo']:.1f} {r['suggested_hi']:.1f}")
    print(f"(or skip the guesswork: --normalize global auto-estimates and freezes this per-video)")


if __name__ == "__main__":
    main()
