"""Unified depth-predictor factory used by the video/visualization scripts.

Returns a single ``predict(bgr_frame) -> (H,W) float32 metric depth`` closure
for either the Phase-1 holistic model or the full Phase-3 pipeline, so tools
that only need "RGB frame in, depth map out" (scripts/make_sequence_videos.py,
scripts/infer_video.py) don't each re-implement the phase dispatch.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np


def build_depth_predictor(
    phase: int, config: str, checkpoint: str, overrides: List[str] | None = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Load a trained model and return ``(predict, max_depth)``.

    ``predict`` takes a BGR uint8 frame and returns a float32 metric-depth map
    at the model's own output resolution (callers resize as needed).
    ``phase`` is 1 (holistic) or 3 (occlusion-refined). ``max_depth`` is the
    configured metric range, for colorization.
    """
    overrides = overrides or []
    if phase == 1:
        from instancedepth.configs.config import HDIConfig
        from instancedepth.models.hdi.inference import HDIInferencer

        cfg = HDIConfig.from_yaml_with_overrides(config, overrides)
        inf = HDIInferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> np.ndarray:
            out = inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return out.depth_final[0, 0].float().cpu().numpy()

        predict.reset = inf.reset_temporal_state   # sequence-boundary hook (no-op for per-frame models)
        return predict, cfg.bins.max_depth

    if phase == 3:
        from instancedepth.configs.phase3_config import Phase3Config
        from instancedepth.models.phase3.inference import Phase3Inferencer

        cfg = Phase3Config.from_yaml_with_overrides(config, overrides)
        inf = Phase3Inferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> np.ndarray:
            return inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["refined"]

        predict.reset = inf.reset_temporal_state
        return predict, cfg.data.max_depth

    raise ValueError(f"phase must be 1 or 3, got {phase}")


def build_scene_predictor(
    phase: int, config: str, checkpoint: str, overrides: List[str] | None = None,
    inst_score_thresh: float = 0.5, mask_binarize_thresh: float = 0.5,
) -> Tuple[Callable[[np.ndarray], dict], float]:
    """Like :func:`build_depth_predictor`, but the returned ``predict`` also
    exposes per-frame instance predictions where the model has them.

    ``predict(bgr)`` returns a dict:
      depth        (H,W) float32 metric depth at the input frame's resolution
      masks        list of (H,W) bool instance masks -- empty for Phase 1,
                   which is holistic-only (callers fall back to GT masks)
      mask_depths  list of float -- predicted depth layer Dep_i per mask

    ``inst_score_thresh`` is a visualization-oriented category-confidence cut
    (0.5, matching evaluate_phase2/visualize_phase2), deliberately looser than
    Phase 3's own 0.9 candidate filter so the overlay shows what the instance
    branch actually sees.
    """
    overrides = overrides or []
    if phase == 1:
        base_predict, max_depth = build_depth_predictor(1, config, checkpoint, overrides)

        def predict(bgr: np.ndarray) -> dict:
            return dict(depth=base_predict(bgr), masks=[], mask_depths=[])

        predict.reset = getattr(base_predict, "reset", lambda: None)
        return predict, max_depth

    if phase == 3:
        import torch

        from instancedepth.configs.phase3_config import Phase3Config
        from instancedepth.models.phase3.inference import Phase3Inferencer

        cfg = Phase3Config.from_yaml_with_overrides(config, overrides)
        inf = Phase3Inferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> dict:
            H, W = bgr.shape[:2]
            out = inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            p2 = out["aux"]["p2"]
            keep = torch.where(p2.scores()[0] > inst_score_thresh)[0]
            masks, deps = [], []
            for q in keep.tolist():
                m = (p2.mask_logits[0, q].sigmoid() > mask_binarize_thresh).float().cpu().numpy()
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                m = m.astype(bool)
                if m.any():
                    masks.append(m)
                    deps.append(float(p2.depth_layers[0, q]))
            return dict(depth=out["refined"], masks=masks, mask_depths=deps)

        predict.reset = inf.reset_temporal_state
        return predict, cfg.data.max_depth

    raise ValueError(f"phase must be 1 or 3, got {phase}")
