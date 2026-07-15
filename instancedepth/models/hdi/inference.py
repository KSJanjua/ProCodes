"""In-process inference API for Phase 1 -- the primary way
Phase 2/3 are meant to consume Holistic Depth Initialization, since
persisting ``feat_final`` (F_2) for the whole ~55k-frame dataset would be a
multi-hundred-GB on-disk cache. Load a checkpoint once, call ``.predict()``
per frame (or ``.predict_batch()``) as needed.

Output resolution note: the model operates internally at
``cfg.data.image_size`` (e.g. 728x1288), but the *raw* dataset
frames (and the masks/instances a future Phase 2 will produce) are at the
camera's native resolution (720x1280). To keep ``HolisticDepthOutput``
spatially aligned with whatever resolution the caller actually gives us,
``predict()`` resizes ``depth_final``/``seg_final`` back to the input
image's own resolution before returning; ``feat_final`` is left at its own
native (smaller) resolution with an updated ``feat_stride`` so ROI-align
-style consumers can still map between the two correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from instancedepth.configs.config import HDIConfig
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.models.hdi.output import HolisticDepthOutput
from instancedepth.utils.checkpoint import load_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(rgb_uint8_hwc: np.ndarray, target_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Matches ``GIDInstanceDepthDataset``'s own normalization exactly
    (instancedepth/data/gid_dataset.py) so inference-time preprocessing
    never silently diverges from training-time preprocessing."""
    import cv2

    h, w = target_hw
    resized = cv2.resize(rgb_uint8_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)
    return tensor.to(device)


class HDIInferencer:
    def __init__(self, cfg: HDIConfig, checkpoint_path: str | Path, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HolisticDepthModel(cfg).to(self.device)
        load_checkpoint(Path(checkpoint_path), self.model, map_location=str(self.device), restore_rng=False)
        self.model.eval()

    def reset_temporal_state(self) -> None:
        """Zero the temporal memory (no-op for per-frame models). Call at
        sequence boundaries when streaming a temporal model frame by frame."""
        if hasattr(self.model, "reset_temporal_state"):
            self.model.reset_temporal_state()

    @torch.no_grad()
    def predict(self, rgb_uint8_hwc: np.ndarray) -> HolisticDepthOutput:
        orig_h, orig_w = rgb_uint8_hwc.shape[:2]
        x = preprocess(rgb_uint8_hwc, self.cfg.data.image_size, self.device)
        out = self.model(x)

        if (orig_h, orig_w) != tuple(out.image_hw):
            depth_final = F.interpolate(out.depth_final, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            seg_final = F.interpolate(out.seg_final, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            out = HolisticDepthOutput(
                depth_final=depth_final,
                seg_final=seg_final,
                feat_final=out.feat_final,
                depth_levels=out.depth_levels,
                seg_levels=out.seg_levels,
                image_hw=(orig_h, orig_w),
                feat_hw=out.feat_hw,
            )
        return out

    @torch.no_grad()
    def predict_batch(self, rgb_batch_uint8_nhwc: np.ndarray) -> HolisticDepthOutput:
        """All frames must share the same original resolution (true within
        one video sequence)."""
        orig_h, orig_w = rgb_batch_uint8_nhwc.shape[1:3]
        tensors = [preprocess(f, self.cfg.data.image_size, self.device) for f in rgb_batch_uint8_nhwc]
        x = torch.cat(tensors, dim=0)
        out = self.model(x)
        if (orig_h, orig_w) != tuple(out.image_hw):
            depth_final = F.interpolate(out.depth_final, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            seg_final = F.interpolate(out.seg_final, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            out = HolisticDepthOutput(
                depth_final=depth_final,
                seg_final=seg_final,
                feat_final=out.feat_final,
                depth_levels=out.depth_levels,
                seg_levels=out.seg_levels,
                image_hw=(orig_h, orig_w),
                feat_hw=out.feat_hw,
            )
        return out
