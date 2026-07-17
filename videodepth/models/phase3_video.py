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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.models.hdi.inference import preprocess
from instancedepth.models.phase3.model import Phase3Model, _geom_channel_count
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.viz import MaskTracker
from videodepth.models.occlusion import BoundedPairAttentionHead
from videodepth.models.track_memory import TrackDepthMemory

log = logging.getLogger("videodepth.models.phase3_video")
_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


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


def is_bounded_relation_head_checkpoint(state_dict_keys: Iterable[str]) -> bool:
    """True if these Phase-3 state-dict keys belong to ``BoundedPairAttentionHead``
    (``relation_head.cross.*``, the pair-attention module) rather than the
    paper's MLP Φo (``relation_head.trunk.*``). Pure key inspection, no
    model construction -- lets ``instancedepth.predict`` pick the right
    inferencer for a checkpoint without importing this module unless it has
    to (instancedepth stays videodepth-independent for vanilla checkpoints)."""
    return any(k.startswith("relation_head.cross.") for k in state_dict_keys)


class Phase3VideoInferencer:
    """Mirrors ``instancedepth.models.phase3.inference.Phase3Inferencer``
    exactly (identical ``predict()`` contract and return shape), but builds
    ``Phase3VideoModel`` so a checkpoint trained by
    ``videodepth.engine.train_phase3_video`` loads into its own head instead
    of the paper's MLP Φo -- the two have different state-dict keys/shapes
    (see ``is_bounded_relation_head_checkpoint``), so using the wrong one
    raises on load (docs/AUDIT_2026.md: this exact failure hit
    ``scripts/infer_video.py`` before this class existed).

    ``max_corr`` is not shape-relevant (only bounds the forward correction),
    so it does not need to match training exactly to load, but should match
    it to reproduce the trained behaviour -- default 0.15 is
    ``train_phase3_video.py``'s own default.
    """

    def __init__(self, cfg: Phase3Config, checkpoint_path: str | Path,
                 device: Optional[torch.device] = None, max_corr: float = 0.15) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The full Phase-3 checkpoint supersedes the base-phase init weights;
        # null them so construction doesn't require (possibly absent) files.
        cfg.phase1_checkpoint = None
        cfg.phase2_checkpoint = None
        self.model = Phase3VideoModel(cfg, max_corr=max_corr).to(self.device)
        load_checkpoint(Path(checkpoint_path), self.model, map_location=str(self.device), restore_rng=False)
        log.info("loaded full Phase 3 (bounded pair-attention head) checkpoint from %s "
                 "(the two 'smoke test only' warnings above are expected here -- the full "
                 "checkpoint supersedes the base-phase init weights)", checkpoint_path)
        self.model.eval()
        precision = cfg.optim.precision
        self.reset_temporal_state = self.model.reset_temporal_state   # sequence-boundary hook
        self._autocast = precision != "fp32" and self.device.type == "cuda"
        self._dtype = _PRECISION_DTYPE[precision]

    @torch.no_grad()
    def predict(self, rgb_uint8_hwc: np.ndarray) -> Dict:
        """Identical contract to ``Phase3Inferencer.predict``: run Phase 1 ->
        Phase 2 -> Phase 3 on one RGB frame, return refined/base depth at the
        input frame's own resolution plus the raw model output/aux."""
        orig_h, orig_w = rgb_uint8_hwc.shape[:2]
        x = preprocess(rgb_uint8_hwc, tuple(self.cfg.data.image_size), self.device)
        with torch.autocast(device_type=self.device.type, dtype=self._dtype, enabled=self._autocast):
            output, aux = self.model(x)

        def to_native(t: torch.Tensor) -> np.ndarray:
            if t.shape[-2:] != (orig_h, orig_w):
                t = F.interpolate(t.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            return t[0, 0].float().cpu().numpy()

        return dict(
            refined=to_native(output.refined_depth),
            base=to_native(output.base_depth),
            output=output,
            aux=aux,
        )


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
