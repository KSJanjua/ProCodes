"""In-process inference API for Phase 3 -- mirrors ``models/hdi/inference.py``'s
HDIInferencer pattern: load once, ``predict(rgb)`` per frame.

Checkpoint semantics: a Phase-3 run checkpoint (``runs/phase3_*/best.pth``)
contains the FULL Phase3Model state (fine-tuned Phase 1 + frozen Phase 2 +
Phi_o), so the config's ``phase1_checkpoint``/``phase2_checkpoint`` base paths
are only *initialization* sources that would immediately be overwritten. This
inferencer therefore nulls them before construction (the resulting
"smoke test only" warnings from Phase3Model are expected and benign here) and
then loads the full checkpoint. The Mask2Former COCO snapshot directory is
still required -- it defines the architecture.

Resolution: the model runs at ``cfg.data.image_size`` (the Phase-2 frame);
``predict`` resizes the refined/base depth maps back to the input frame's own
resolution, same convention as HDIInferencer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.models.hdi.inference import preprocess
from instancedepth.models.phase3.model import Phase3Model
from instancedepth.utils.checkpoint import load_checkpoint

log = logging.getLogger("instancedepth.models.phase3.inference")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Phase3Inferencer:
    def __init__(self, cfg: Phase3Config, checkpoint_path: str | Path,
                 device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The full Phase-3 checkpoint supersedes the base-phase init weights;
        # null them so construction doesn't require (possibly absent) files.
        cfg.phase1_checkpoint = None
        cfg.phase2_checkpoint = None
        self.model = Phase3Model(cfg).to(self.device)
        load_checkpoint(Path(checkpoint_path), self.model, map_location=str(self.device), restore_rng=False)
        log.info("loaded full Phase 3 checkpoint from %s (the two 'smoke test only' "
                 "warnings above are expected here -- the full checkpoint supersedes "
                 "the base-phase init weights)", checkpoint_path)
        self.model.eval()
        precision = cfg.optim.precision
        self.reset_temporal_state = self.model.reset_temporal_state   # sequence-boundary hook
        self._autocast = precision != "fp32" and self.device.type == "cuda"
        self._dtype = _PRECISION_DTYPE[precision]

    @torch.no_grad()
    def predict(self, rgb_uint8_hwc: np.ndarray) -> Dict:
        """Run the full Phase 1 -> Phase 2 -> Phase 3 pipeline on one RGB frame.

        Returns a dict:
          refined  (H,W) float32 np -- occlusion-refined metric depth, at the
                   input frame's own resolution
          base     (H,W) float32 np -- Phase-1 depth, same resolution
          output   RefinedDepthOutput at model resolution (ROI tensors intact)
          aux      model aux dict (Phase2Output ``p2``, ``pairs``, ...)
        """
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
