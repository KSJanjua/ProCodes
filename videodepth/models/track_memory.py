"""Temporal amodal completion via per-track depth memory.

The paper's Phase 3 reasons about occlusion **within a single frame**: when a
person is heavily occluded, the only evidence for their depth is the sliver of
visible pixels plus the pair-relation prior. But this is *video* — the same
person (tracked identity) was fully visible a moment ago. This module
exploits that with a single visibility-modulated EMA:

    memory(t) = (1 − m·v_t) · memory(t−1) + m·v_t · layer_pred(t)
    stabilized_layer(t) = memory(t)

where ``v_t`` ∈ [0,1] is the instance's current *visibility* and ``m`` the
base momentum, plus a constant-velocity term (a person walking away keeps
receding while hidden). Fully visible ⇒ the observation folds in at full
momentum, so the output is a low-passed (flicker-free) estimate; fully
occluded ⇒ the update is damped to nothing and the output is the track's held
depth — the temporal analogue of amodal completion. It is **training-free**:
pure inference-time state, deployable on any existing Phase-2/Phase-3
checkpoint.

Visibility source: ``visibility_from_area`` (mask area vs. the track's own
area EMA — a shrinking mask means growing occlusion) works with any
segmenter; callers with better signals (GT overlap, pair IoU) can pass their
own ``v``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class _Track:
    layer: float                     # EMA of depth layer (m)
    velocity: float = 0.0            # EMA of frame-to-frame layer delta (m/frame)
    area: float = 0.0                # EMA of mask area (px) — visibility reference
    age: int = 0                     # frames since last update


class TrackDepthMemory:
    """Streaming per-track depth-layer memory. ``reset()`` at sequence starts.

    Parameters
    ----------
    momentum : EMA momentum for layer/area under FULL visibility (the
        effective momentum scales with visibility: an occluded observation
        barely moves the memory — its layer estimate is contaminated by the
        occluder).
    velocity_momentum : EMA momentum for the constant-velocity term.
    max_age : forget a track not seen for this many frames.
    """

    def __init__(self, momentum: float = 0.35, velocity_momentum: float = 0.2,
                 max_age: int = 30) -> None:
        assert 0.0 < momentum <= 1.0 and 0.0 <= velocity_momentum <= 1.0
        self.momentum = momentum
        self.velocity_momentum = velocity_momentum
        self.max_age = max_age
        self._tracks: Dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks = {}

    def step(self) -> None:
        """Advance one frame: age every track, coast it along its velocity,
        and forget stale ones. Call once per frame, before updates."""
        stale = []
        for tid, t in self._tracks.items():
            t.age += 1
            t.layer += t.velocity          # coast while unobserved
            if t.age > self.max_age:
                stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

    # ------------------------------------------------------------- update
    def update(self, track_id: int, layer: float, visibility: float,
               area: Optional[float] = None) -> float:
        """Fold this frame's observation in; returns the stabilized layer —
        the visibility-weighted EMA itself.

        visibility ∈ [0,1] scales the EMA step: 1 = fully visible (observation
        folded in at full momentum, output = smoothed estimate — this is what
        low-passes per-frame layer flicker), 0 = fully occluded (memory
        unchanged, output = the track's held depth — amodal completion). One
        formula covers both regimes."""
        v = min(max(float(visibility), 0.0), 1.0)
        t = self._tracks.get(track_id)
        if t is None:
            t = _Track(layer=float(layer), area=float(area or 0.0))
            self._tracks[track_id] = t
            return float(layer)

        m = self.momentum * v                       # occlusion damps the update
        new_layer = (1.0 - m) * t.layer + m * float(layer)
        delta = new_layer - t.layer
        t.velocity = (1.0 - self.velocity_momentum) * t.velocity \
            + self.velocity_momentum * delta
        t.layer = new_layer
        if area is not None:
            # area EMA updates faster when visible (same damping logic)
            t.area = (1.0 - m) * t.area + m * float(area)
        t.age = 0
        return t.layer

    # ------------------------------------------------------------- queries
    def visibility_from_area(self, track_id: int, area: float) -> float:
        """Visibility proxy: current mask area over the track's area EMA,
        clamped to [0,1]. A person half-hidden behind another shows ~half
        their usual area. New/unknown tracks -> 1.0 (no reference yet)."""
        t = self._tracks.get(track_id)
        if t is None or t.area <= 0:
            return 1.0
        return min(max(area / t.area, 0.0), 1.0)

    def get(self, track_id: int) -> Optional[float]:
        t = self._tracks.get(track_id)
        return None if t is None else t.layer

    def __len__(self) -> int:
        return len(self._tracks)
