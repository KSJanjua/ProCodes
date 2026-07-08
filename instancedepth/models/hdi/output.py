"""HolisticDepthOutput -- the single contract Phase 2/3 consume (plan SS17).

``contract_version`` is bumped whenever a field is added/removed/reshaped,
so a later phase reading a stale artifact (or calling a stale in-process
API) fails loudly instead of silently misinterpreting a tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

CONTRACT_VERSION = "1.0"


@dataclass
class HolisticDepthOutput:
    depth_final: torch.Tensor          # (B,1,H,W) metric meters, full input resolution
    seg_final: torch.Tensor            # (B,rd,H,W) ordinal per-bin activations, full input resolution
    feat_final: torch.Tensor           # (B,C,h2,w2) finest decoder feature map F_2, native (unupsampled) resolution
    depth_levels: List[torch.Tensor]   # [D_0, D_1, D_2] at the decoder's internal 1/8,1/4,1/2 resolutions
    seg_levels: List[torch.Tensor]     # [S_0, S_1, S_2], same resolutions as depth_levels[0:3]... (S_i at level i)
    image_hw: Tuple[int, int]          # (H, W) of depth_final/seg_final
    feat_hw: Tuple[int, int]           # (h2, w2) of feat_final
    contract_version: str = CONTRACT_VERSION

    def feat_stride(self) -> Tuple[float, float]:
        """(stride_h, stride_w) mapping feat_final's coordinate space to
        image_hw's -- for Phase 2/3 ROI-align (`spatial_scale = 1/stride`)."""
        H, W = self.image_hw
        h2, w2 = self.feat_hw
        return (H / h2, W / w2)
