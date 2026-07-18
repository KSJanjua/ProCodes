"""Unit tests for the training-free temporal depth smoother
(videodepth/models/depth_smoother.py) -- the anti-flicker post-processing for
out-of-domain video. The contracts:

  * static-scene noise is strongly damped (the whole point);
  * moving content is NOT smeared -- pixels whose RGB changed follow the
    current prediction;
  * global scale pumping (whole-map ratio jumping frame to frame) is locked;
  * invalid (<=0) pixels are never hallucinated into valid depth;
  * strength=0 wiring is a no-op; reset() forgets the video.
"""

from __future__ import annotations

import numpy as np
import pytest

from videodepth.models.depth_smoother import TemporalDepthSmoother


def _static_frame(H=40, W=40, val=120):
    return np.full((H, W, 3), val, np.uint8)


def test_static_scene_flicker_is_strongly_damped():
    rng = np.random.default_rng(0)
    sm = TemporalDepthSmoother(strength=0.85, scale_stabilize=False)
    bgr = _static_frame()
    base = np.full((40, 40), 4.0, np.float32)

    raw_prev = out_prev = None
    raw_deltas, out_deltas = [], []
    for _ in range(20):
        noisy = base + rng.normal(0, 0.15, base.shape).astype(np.float32)   # frame-to-frame shimmer
        out = sm(noisy, bgr)
        if raw_prev is not None:
            raw_deltas.append(np.abs(noisy - raw_prev).mean())
            out_deltas.append(np.abs(out - out_prev).mean())
        raw_prev, out_prev = noisy, out
    # smoothed frame-to-frame change is a small fraction of the raw shimmer
    assert np.mean(out_deltas) < 0.35 * np.mean(raw_deltas)


def test_moving_content_follows_current_prediction():
    sm = TemporalDepthSmoother(strength=0.85, scale_stabilize=False)
    H = W = 40
    # frame 1: dark left / bright right; person at 2 m in the left half
    bgr1 = np.zeros((H, W, 3), np.uint8); bgr1[:, W // 2:] = 200
    d1 = np.full((H, W), 5.0, np.float32); d1[:, :W // 2] = 2.0
    sm(d1, bgr1)
    # frame 2: the bright/dark edge SWAPPED (big RGB change everywhere) and the
    # person moved to the right half at a new depth
    bgr2 = np.full((H, W, 3), 200, np.uint8); bgr2[:, W // 2:] = 0
    d2 = np.full((H, W), 5.0, np.float32); d2[:, W // 2:] = 2.5
    out = sm(d2, bgr2)
    # where RGB changed hugely, the output must be ~the current prediction
    assert abs(float(out[:, W // 2:].mean()) - 2.5) < 0.1
    assert abs(float(out[:, :W // 2].mean()) - 5.0) < 0.1


def test_global_scale_pumping_is_locked():
    sm = TemporalDepthSmoother(strength=0.0, scale_stabilize=True)   # isolate the scale lock
    bgr = _static_frame()
    base = np.linspace(2.0, 6.0, 1600, dtype=np.float32).reshape(40, 40)
    medians = []
    for t in range(12):
        scale = 1.2 if t % 2 else 1.0 / 1.2      # the map "pumps" +-20% every frame
        out = sm(base * scale, bgr)
        medians.append(float(np.median(out)))
    raw_swing = abs(1.2 - 1 / 1.2) * float(np.median(base))
    swings = np.abs(np.diff(medians[2:]))        # after the reference settles
    assert swings.max() < 0.35 * raw_swing       # pumping strongly suppressed


def test_invalid_pixels_never_hallucinated():
    sm = TemporalDepthSmoother(strength=0.9, scale_stabilize=False)
    bgr = _static_frame()
    d1 = np.full((20, 20), 3.0, np.float32)
    sm(d1, bgr)
    d2 = d1.copy(); d2[5, 5] = 0.0               # sensor/model hole this frame
    out = sm(d2, bgr)
    assert out[5, 5] == 0.0                      # stays invalid, not filled from memory


def test_first_frame_passthrough_and_reset():
    sm = TemporalDepthSmoother(strength=0.9, scale_stabilize=False)
    bgr = _static_frame()
    d = np.full((10, 10), 4.0, np.float32)
    assert np.allclose(sm(d, bgr), d)            # nothing to blend with yet
    sm(np.full((10, 10), 8.0, np.float32), bgr)  # builds history
    sm.reset()
    d3 = np.full((10, 10), 1.0, np.float32)
    assert np.allclose(sm(d3, bgr), d3)          # history forgotten


def test_resolution_change_is_safe():
    sm = TemporalDepthSmoother(strength=0.9)
    sm(np.full((20, 20), 4.0, np.float32), _static_frame(20, 20))
    out = sm(np.full((30, 30), 4.0, np.float32), _static_frame(30, 30))
    assert out.shape == (30, 30)                 # shape change -> fresh blend, no crash
