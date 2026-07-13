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

        return predict, cfg.bins.max_depth

    if phase == 3:
        from instancedepth.configs.phase3_config import Phase3Config
        from instancedepth.models.phase3.inference import Phase3Inferencer

        cfg = Phase3Config.from_yaml_with_overrides(config, overrides)
        inf = Phase3Inferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> np.ndarray:
            return inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["refined"]

        return predict, cfg.data.max_depth

    raise ValueError(f"phase must be 1 or 3, got {phase}")
