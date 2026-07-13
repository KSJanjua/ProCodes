"""RefinedDepthOutput -- the Phase 3 contract, final stage of the pipeline.

Mirrors ``models/hdi/output.py`` (HolisticDepthOutput) and
``models/phase2/output.py`` (Phase2Output): a dataclass with a
``contract_version`` string, deliberately over-exposing the intermediate
pair bookkeeping so downstream eval / visualization / 3D consumers have a
stable target.

The dense ``refined_depth`` is Phase 1's ``depth_final`` with each confident
occlusion pair's ROI corrections composited back in (nearest-depth-wins for
overlaps, matching ``data_engine/annotate.py::_flatten_id_map``). Regions
with no confident/paired instance are Phase-1 depth verbatim, so non-crowded
scenes degrade gracefully to Phase-1 quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

CONTRACT_VERSION = "1.0"


@dataclass
class RefinedDepthOutput:
    refined_depth: torch.Tensor            # (B,1,H,W) metric meters -- composited final map
    base_depth: torch.Tensor               # (B,1,H,W) Phase-1 depth_final, pre-refinement (for delta eval)
    image_hw: Tuple[int, int]              # (H, W) of refined_depth / base_depth

    # --- per-pair bookkeeping (flattened across the batch) --------------------
    pair_batch_index: torch.Tensor         # (P,) which image each pair belongs to
    pair_query_idx: torch.Tensor           # (P,2) [main, guest] query indices into the N Phase-2 queries
    pair_iou: torch.Tensor                 # (P,) main<->guest mask IoU (> overlap thresh)
    refined_layers: torch.Tensor           # (P,2) refined scalar depth of [main, guest] (mask-mean of D_hat)
    base_layers: torch.Tensor              # (P,2) pre-refinement scalar depth of [main, guest]

    # --- training-only tensors (None at inference) ---------------------------
    # D_hat / E_obj kept so the criterion can be computed outside the model and
    # so ablations can inspect the raw correction field.
    d_hat_roi: Optional[torch.Tensor] = None   # (P,2,1,Hp,Wp) refined ROI depth (dense mode)
    e_obj_roi: Optional[torch.Tensor] = None   # (P,2,1,Hp,Wp) or (P,2,1,1,1) relative-error field
    d_obj_roi: Optional[torch.Tensor] = None   # (P,2,1,Hp,Wp) ROIAligned base depth (Eq.9's D_obj)

    contract_version: str = CONTRACT_VERSION

    def num_pairs(self) -> int:
        return int(self.pair_batch_index.shape[0])
