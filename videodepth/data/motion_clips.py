"""Motion-aware clip sampling for temporal training.

Two dataset facts drive this design (user-confirmed):

  * Frame-to-frame motion is small; it becomes *visible* over spans of
    ~100 frames. A temporal module trained on adjacent-frame clips therefore
    sees almost no signal -> strides up to 24 make a clip_len=5 clip span up
    to 97 frames, so training clips actually contain motion.
  * Some sequences are only ~50 frames. Long-span clips can't fit there (the
    index construction skips them naturally), and ``min_seq_len`` drops such
    sequences from temporal training entirely.

On top of that, ``motion_weighting`` samples clips proportionally to their
measured GT motion (mean |Δlog-depth| between consecutive clip frames on a
subsampled grid, computed once and cached to JSON). High-motion clips — the
only ones that teach the stabilizer anything — dominate training; a weight
floor keeps static clips present so the module also learns to *do nothing*
when nothing moves (its most common inference-time state on this data).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from instancedepth.data.clip_dataset import ClipDatasetConfig, GIDClipDataset, collate_clips
from videodepth.configs.config import ClipConfig

log = logging.getLogger("videodepth.data.motion_clips")

_EPS = 1e-6
_GRID_STRIDE = 8   # motion scored on every 8th pixel — plenty for a scalar


def frame_motion_scores(depths: List[np.ndarray]) -> List[float]:
    """Per-frame motion score for one ordered sequence of GT depth maps:
    score[t] = mean |log d_t − log d_{t-1}| over pixels valid in both frames
    (score[0]=0). Log space matches the training losses; pixels where either
    frame is invalid (0) contribute nothing, so sensor holes never read as
    motion."""
    scores = [0.0]
    prev = depths[0]
    for d in depths[1:]:
        a, b = prev[::_GRID_STRIDE, ::_GRID_STRIDE], d[::_GRID_STRIDE, ::_GRID_STRIDE]
        # raw depth files may carry NaN/inf (the dataset sanitizes only in
        # __getitem__, and this scorer reads the raw arrays)
        valid = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
        if valid.any():
            la = np.log(np.maximum(a[valid], _EPS))
            lb = np.log(np.maximum(b[valid], _EPS))
            scores.append(float(np.abs(la - lb).mean()))
        else:
            scores.append(0.0)
        prev = d
    return scores


def clip_weights(index: List[Tuple[str, int, int]],
                 seq_scores: Dict[str, List[float]],
                 clip_len: int, floor: float) -> torch.Tensor:
    """Sampling weight per clip = mean motion of the frames it spans (frame
    t's score is its motion *from* t-1, so a clip (start, stride) accumulates
    the scores of the frames it actually steps across), normalized to mean 1
    over the index, then floored. Sequences missing from ``seq_scores`` get
    weight 1 (neutral) rather than being silently dropped."""
    raw = []
    for sid, start, stride in index:
        s = seq_scores.get(sid)
        if s is None:
            raw.append(float("nan"))
            continue
        span = [s[min(start + t * stride, len(s) - 1)] for t in range(1, clip_len)]
        raw.append(float(np.mean(span)) if span else 0.0)
    w = np.asarray(raw, np.float64)
    known = ~np.isnan(w)
    mean = w[known].mean() if known.any() and w[known].mean() > 0 else 1.0
    w = np.where(known, w / mean, 1.0)
    return torch.from_numpy(np.maximum(w, floor))


class MotionClipDataset(GIDClipDataset):
    """GIDClipDataset + min-sequence-length filtering + per-clip motion
    weights (``.weights``) ready for a WeightedRandomSampler."""

    def __init__(self, base_cfg: ClipDatasetConfig, clips: ClipConfig,
                 cache_path: Path | None = None) -> None:
        super().__init__(base_cfg)

        if clips.min_seq_len > 0:
            before = len(self.index)
            long_enough = {sid for sid, man in self._manifests.items()
                           if len(man["frames"]) >= clips.min_seq_len}
            self.index = [e for e in self.index if e[0] in long_enough]
            dropped = sorted(set(self._manifests) - long_enough)
            if dropped:
                log.info("min_seq_len=%d: dropped %d short sequences (%s…), %d -> %d clips",
                         clips.min_seq_len, len(dropped), dropped[0], before, len(self.index))

        if clips.motion_weighting:
            scores = self._load_or_compute_scores(cache_path)
            self.weights = clip_weights(self.index, scores, base_cfg.clip_len,
                                        clips.motion_floor)
        else:
            self.weights = torch.ones(len(self.index), dtype=torch.float64)

    # ------------------------------------------------------------- scores
    def _load_or_compute_scores(self, cache_path: Path | None) -> Dict[str, List[float]]:
        if cache_path is not None and cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            if set(cached) >= set(s for s, _, _ in self.index):
                log.info("motion scores: loaded cache %s", cache_path)
                return cached

        from instancedepth.data.gid_dataset import GIDInstanceDepthDataset
        scores: Dict[str, List[float]] = {}
        needed = sorted({sid for sid, _, _ in self.index})
        for sid in needed:
            man = self._manifests[sid]
            keys = sorted(man["frames"].keys())
            depths = [GIDInstanceDepthDataset._load_depth(man["frames"][k],
                                                          man["depth_scale_to_m"])
                      for k in keys]
            scores[sid] = frame_motion_scores(depths)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(scores, f)
            log.info("motion scores: computed %d sequences, cached to %s",
                     len(needed), cache_path)
        return scores


def build_motion_clip_loader(base_cfg: ClipDatasetConfig, clips: ClipConfig,
                             batch_size: int, num_workers: int,
                             cache_path: Path | None = None) -> DataLoader:
    ds = MotionClipDataset(base_cfg, clips, cache_path=cache_path)
    log.info("motion clip dataset: %d clips (len %d, strides %s, weighted=%s)",
             len(ds), base_cfg.clip_len, base_cfg.strides, clips.motion_weighting)
    sampler = WeightedRandomSampler(ds.weights, num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate_clips,
                      drop_last=True, pin_memory=True)
