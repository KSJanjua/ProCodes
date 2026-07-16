"""Video-aware Phase 3: bounded pair-attention Φo + streaming instance-depth
stabilization.

``Phase3VideoModel`` is ``instancedepth``'s Phase3Model with one substitution:
the MLP relation head is replaced by ``BoundedPairAttentionHead`` (same
interface — compositor, losses, matcher, trainer all run unchanged). Training
therefore reuses ``instancedepth.engine.train_phase3`` wholesale via
``videodepth.engine.train_phase3_video``.

``StreamingInstanceStabilizer`` is the inference-time video layer: it takes
each frame's predicted instance (masks, depth layers), assigns persistent
identities (Hungarian MaskTracker), and stabilizes each track's depth layer
through occlusions with ``TrackDepthMemory`` — temporal amodal completion,
zero training required, usable on any existing checkpoint.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.models.phase3.model import Phase3Model, _geom_channel_count
from instancedepth.utils.viz import MaskTracker
from videodepth.models.occlusion import BoundedPairAttentionHead
from videodepth.models.track_memory import TrackDepthMemory

log = logging.getLogger("videodepth.models.phase3_video")


class Phase3VideoModel(Phase3Model):
    """Phase3Model with the bounded pair-attention relation head."""

    def __init__(self, cfg: Phase3Config, max_corr: float = 0.15) -> None:
        super().__init__(cfg)
        feat_c = self._feat_channels
        per_member = feat_c + _geom_channel_count(cfg.head)
        self.relation_head = BoundedPairAttentionHead(
            per_member_channels=per_member,
            hidden_dim=cfg.head.hidden_dim,
            num_conv=cfg.head.num_conv,
            granularity=cfg.head.refine_granularity,
            max_corr=max_corr,
        )
        log.info("Phase3VideoModel: BoundedPairAttentionHead (max_corr=%.2f) "
                 "replaces the MLP Φo", max_corr)


class StreamingInstanceStabilizer:
    """Per-sequence streaming state: persistent IDs + occlusion-robust layers.

    update(masks, layers) -> (masks, stabilized_layers, ids); reset() at every
    sequence boundary. Purely observational — it never touches the dense
    depth map, only the reported per-instance layers and identities, so it can
    wrap ANY per-frame instance-depth pipeline (Phase 2 or Phase 3 output).
    """

    def __init__(self, iou_thresh: float = 0.3, min_hits: int = 2,
                 max_age: int = 5, memory_momentum: float = 0.35,
                 memory_max_age: int = 30) -> None:
        self.tracker = MaskTracker(iou_thresh=iou_thresh, min_hits=min_hits,
                                   max_age=max_age)
        self.memory = TrackDepthMemory(momentum=memory_momentum,
                                       max_age=memory_max_age)

    def reset(self) -> None:
        self.tracker.reset()
        self.memory.reset()

    def update(self, masks: Sequence[np.ndarray], layers: Sequence[float]
               ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        t_masks, t_layers, t_ids = self.tracker.update(masks, layers)
        self.memory.step()
        stabilized: List[float] = []
        for m, layer, tid in zip(t_masks, t_layers, t_ids):
            area = float(np.asarray(m, bool).sum())
            v = self.memory.visibility_from_area(tid, area)
            stabilized.append(self.memory.update(tid, float(layer), v, area=area))
        return t_masks, stabilized, t_ids
