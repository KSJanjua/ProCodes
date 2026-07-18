"""Training-free temporal depth smoothing for out-of-domain video.

The trained temporal stabilizer lives inside the Phase-1 clip model and was
fit on THIS project's dataset; on foreign footage the base model itself is
out of domain and its per-frame predictions flicker in two distinct ways:

  1. **Global scale pumping** -- the whole map's scale jumps frame to frame
     (a metric model guessing scale on scenes it never saw). Visible as the
     entire video "breathing" between colours.
  2. **Per-pixel shimmer** -- static regions (walls, floor) re-estimated
     slightly differently every frame.

``TemporalDepthSmoother`` addresses both at inference, with no training and
no ground truth, using the RGB frames themselves as the change detector:

  * scale lock: each frame is rescaled so its median depth follows an EMA of
    the sequence median (ratio clamped, so a real scene change can still move
    the scale, just not instantaneously);
  * motion-gated EMA: per pixel, blend toward the previous smoothed depth
    with weight  strength * exp(-|Δgray| / motion_thresh)  -- static pixels
    (tiny |Δgray|) get heavy smoothing, moving pixels follow the current
    prediction almost untouched, so moving people don't smear.

Honest scope: this is a *visualization/demo* stabilizer -- it makes the
output video steady, it does not make the model more accurate on foreign
domains (that's the full-DAv2-prior work). On in-domain footage it is nearly
a no-op (predictions barely change frame to frame, so the EMA tracks them).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class TemporalDepthSmoother:
    """Streaming per-video state; call ``reset()`` at every new video.

    Parameters
    ----------
    strength : float in [0, 1)
        Maximum blend toward the previous smoothed depth (applies to fully
        static pixels). 0 disables smoothing entirely; 0.85 is a good demo
        default (static regions average over ~6-7 frames).
    motion_thresh : float
        Grayscale |Δ| (0-255 units) at which a pixel counts as "moving":
        blend weight falls off as exp(-Δ/motion_thresh), so pixels changing
        much more than this follow the current prediction.
    scale_stabilize : bool
        Enable the global median-scale lock.
    scale_momentum : float in [0, 1)
        EMA momentum of the median-depth scale reference.
    max_scale_step : float > 1
        Per-frame clamp on the correction ratio (both directions), so a
        genuine scene change can re-converge within a few frames instead of
        being pinned forever.
    """

    def __init__(self, strength: float = 0.85, motion_thresh: float = 12.0,
                 scale_stabilize: bool = True, scale_momentum: float = 0.9,
                 max_scale_step: float = 1.25) -> None:
        assert 0.0 <= strength < 1.0
        assert motion_thresh > 0 and 0.0 <= scale_momentum < 1.0 and max_scale_step > 1.0
        self.strength = strength
        self.motion_thresh = motion_thresh
        self.scale_stabilize = scale_stabilize
        self.scale_momentum = scale_momentum
        self.max_scale_step = max_scale_step
        self.reset()

    def reset(self) -> None:
        self._prev_depth: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._ema_median: Optional[float] = None

    # ------------------------------------------------------------------ #
    def _gray(self, bgr: np.ndarray, shape) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if g.shape != shape:
            g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return g

    def __call__(self, depth: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        """depth: (H,W) float metric prediction for this frame; bgr: the
        frame it came from (any resolution). Returns the smoothed depth."""
        d = np.asarray(depth, np.float32)
        gray = self._gray(bgr, d.shape)

        # ---- global scale lock -----------------------------------------
        if self.scale_stabilize:
            valid = d > 0
            if valid.any():
                med = float(np.median(d[valid]))
                if med > 1e-6:
                    if self._ema_median is None:
                        self._ema_median = med
                    ratio = self._ema_median / med
                    ratio = float(np.clip(ratio, 1.0 / self.max_scale_step, self.max_scale_step))
                    d = d * ratio
                    # update the reference with the CORRECTED frame's median, so
                    # a persistent scene change walks the reference over (clamp
                    # lets it move max_scale_step per frame) instead of fighting
                    # it forever
                    self._ema_median = (self.scale_momentum * self._ema_median
                                        + (1.0 - self.scale_momentum) * med * ratio)

        # ---- motion-gated per-pixel EMA --------------------------------
        if self._prev_depth is not None and self._prev_depth.shape == d.shape:
            diff = np.abs(gray - self._prev_gray)
            w = self.strength * np.exp(-diff / self.motion_thresh)   # 1=static, ->0 moving
            # never borrow from previously-invalid pixels
            w = np.where(self._prev_depth > 0, w, 0.0).astype(np.float32)
            out = w * self._prev_depth + (1.0 - w) * d
            # a currently-invalid pixel stays invalid (don't hallucinate depth)
            out = np.where(d > 0, out, d)
        else:
            out = d

        self._prev_depth = out
        self._prev_gray = gray
        return out
