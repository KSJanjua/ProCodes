"""Query-embedding instance tracking -- the Mask2Former-VIS identity concept,
without the video training.

**The concept** (Cheng et al., "Mask2Former for Video Instance Segmentation"):
in video Mask2Former a single query represents the SAME instance in every
frame of a clip, so identity consistency is a property of the query, not a
post-hoc association. The full video model buys that with clip-level
attention and video training -- heavy, and it would obsolete the trained
``phase2_run`` checkpoint.

**The adaptation** (proven by MinVIS, NeurIPS 2022: per-frame Mask2Former +
query matching beats video-trained VIS models): the per-frame model's final
transformer-decoder query embeddings are already temporally consistent --
the same person yields a nearby embedding in the next frame even though the
QUERY INDEX shuffles. So track identity by Hungarian-matching query
embeddings (cosine) across frames, with mask IoU as a secondary spatial cue.
Zero retraining; works on the existing checkpoint; and unlike mask-IoU-only
tracking it survives the hard cases:

  * two people crossing: embeddings follow appearance, IoU follows position;
  * occlusion gaps: an unseen track keeps its embedding, so the person
    RE-IDENTIFIES on return -- even many frames later, from a different
    position (mask IoU alone can never do this).

``QueryInstanceTracker.update(masks, depths, embeds)`` -> (masks, depths,
ids) with persistent ids; ``embeds=None`` degrades gracefully to IoU-only
matching, so it is a drop-in superset of ``instancedepth.utils.viz
.MaskTracker`` and the tools use it whenever embeddings are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class _Track:
    tid: int
    embed: Optional[np.ndarray]      # EMA of the query embedding (None in IoU-only mode)
    mask: np.ndarray                 # last seen mask
    depth: float
    hits: int = 1
    age: int = 0                     # frames since last match


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(a @ b / (na * nb))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


class QueryInstanceTracker:
    """Streaming per-video tracker; ``reset()`` at every new video.

    Parameters
    ----------
    w_embed / w_iou : cost weights. Embedding similarity dominates (the VIS
        cue); IoU breaks ties spatially. With ``embeds=None`` the cost is
        IoU-only automatically.
    min_score : minimum combined score to accept a match; below it a
        detection starts a new track instead of corrupting an old one.
    embed_momentum : EMA momentum of a track's embedding on match (drifts
        with appearance changes without forgetting the identity).
    min_hits : detections needed before a track is shown (suppresses
        one-frame false positives).
    max_age : frames a track survives unseen. Deliberately long (30): the
        embedding lets a person re-identify after a real occlusion, which is
        the whole point -- mask-IoU trackers must keep this short because
        they can only re-match by position.
    """

    def __init__(self, w_embed: float = 0.7, w_iou: float = 0.3,
                 min_score: float = 0.35, embed_momentum: float = 0.7,
                 min_hits: int = 2, max_age: int = 30) -> None:
        assert w_embed >= 0 and w_iou >= 0 and (w_embed + w_iou) > 0
        self.w_embed = w_embed
        self.w_iou = w_iou
        self.min_score = min_score
        self.embed_momentum = embed_momentum
        self.min_hits = min_hits
        self.max_age = max_age
        self.reset()

    def reset(self) -> None:
        self._tracks: List[_Track] = []
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def _score(self, t: _Track, mask: np.ndarray, embed: Optional[np.ndarray]) -> float:
        if embed is not None and t.embed is not None:
            sim01 = 0.5 * (1.0 + _cos(t.embed, embed))          # cosine -> [0,1]
            # IoU only counts for tracks seen RECENTLY; an aged track's last
            # position is stale, so its re-identification rides on the embed.
            iou = _iou(t.mask, mask) if t.age == 0 else 0.0
            return (self.w_embed * sim01 + self.w_iou * iou) / (self.w_embed + self.w_iou)
        return _iou(t.mask, mask)                                # IoU-only fallback

    def update(self, masks: Sequence[np.ndarray], depths: Sequence[float],
               embeds: Optional[Sequence[np.ndarray]] = None
               ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        masks = [np.asarray(m, bool) for m in masks]
        if embeds is not None:
            embeds = [np.asarray(e, np.float32).ravel() for e in embeds]
            assert len(embeds) == len(masks), "one embedding per mask"

        n_t, n_d = len(self._tracks), len(masks)
        matches: List[Tuple[int, int]] = []
        if n_t and n_d:
            score = np.zeros((n_t, n_d), np.float32)
            for ti, t in enumerate(self._tracks):
                for di in range(n_d):
                    score[ti, di] = self._score(t, masks[di],
                                                embeds[di] if embeds is not None else None)
            try:
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(-score)
                matches = [(int(r), int(c)) for r, c in zip(rows, cols)
                           if score[r, c] >= self.min_score]
            except Exception:                                     # scipy-less: greedy
                order = sorted(((score[r, c], r, c) for r in range(n_t) for c in range(n_d)),
                               reverse=True)
                used_t, used_d = set(), set()
                for sc, r, c in order:
                    if sc < self.min_score or r in used_t or c in used_d:
                        continue
                    used_t.add(r); used_d.add(c); matches.append((r, c))

        matched_t = {r for r, _ in matches}
        matched_d = {c for _, c in matches}

        for r, c in matches:
            t = self._tracks[r]
            t.mask = masks[c]
            t.depth = float(depths[c])
            if embeds is not None:
                e = embeds[c]
                t.embed = e if t.embed is None else \
                    (self.embed_momentum * t.embed + (1.0 - self.embed_momentum) * e)
            t.hits += 1
            t.age = 0

        for ti, t in enumerate(self._tracks):                     # age the unmatched
            if ti not in matched_t:
                t.age += 1

        for di in range(n_d):                                     # unmatched dets -> new tracks
            if di in matched_d:
                continue
            self._tracks.append(_Track(
                tid=self._next_id,
                embed=(embeds[di] if embeds is not None else None),
                mask=masks[di], depth=float(depths[di])))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        vis = [t for t in self._tracks if t.hits >= self.min_hits and t.age == 0]
        return ([t.mask for t in vis], [t.depth for t in vis], [t.tid for t in vis])
