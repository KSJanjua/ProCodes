"""Eq. 1-4 iterative bin refinement, across the 3 Depth Range Feature
Decoder levels.

Indexing follows the paper's own convention exactly: D_0 is the seed (from
`InitialDepthHead` on F_0); level i (i=0,1,2) consumes D_i and produces
D_{i+1}. D_3 is the final holistic depth (before the last upsample to full
input resolution, which lives in `model.py` -- this module operates at the
decoder's internal 1/8,1/4,1/2 resolutions only).

See the plan (SS1, SS5) for exactly which parts of this are paper-equation
(Eq. 1-4 themselves) vs. this project's resolved reading of the 'i' index
collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.configs.config import BinRefinementConfig
from instancedepth.models.hdi.bin_heads import ConfidenceHead, InitialDepthHead, OrdinalBinHead
from instancedepth.models.hdi.depth_range_decoder import DepthRangeFeatures


@dataclass
class RefinementTrace:
    """Every D_i and S_i produced during the recurrence -- deep supervision
    (SS6) needs D_1, D_2 (not just the seed D_0 or the final D_3), and all
    of S_0, S_1, S_2.

    ``bins`` stores raw logits (pre-sigmoid) -- see bin_heads.py's
    ``OrdinalBinHead`` docstring for why: the loss needs logits for
    ``binary_cross_entropy_with_logits`` (autocast-safe, numerically
    stable), so logits are the canonical stored form and probability is
    derived on demand (here, and in ``HolisticDepthOutput.seg_confidence()``)."""

    depths: List[torch.Tensor] = field(default_factory=list)   # [D_0, D_1, D_2, D_3]
    bins: List[torch.Tensor] = field(default_factory=list)     # [S_0, S_1, S_2] logits

    @property
    def final_depth(self) -> torch.Tensor:
        return self.depths[-1]

    @property
    def final_bins(self) -> torch.Tensor:
        return self.bins[-1]


class IterativeBinRefinement(nn.Module):
    def __init__(self, feat_channels: int, cfg: BinRefinementConfig) -> None:
        super().__init__()
        self.rd = cfg.rd
        self.max_depth = cfg.max_depth

        self.initial_depth_head = InitialDepthHead(feat_channels)
        # Independent heads per level (their inputs -- F_i's semantic
        # content -- genuinely differ level to level; sharing weights
        # across levels is not suggested by the paper either way, and
        # independent heads are the more expressive, standard default).
        self.confidence_heads = nn.ModuleList([ConfidenceHead(feat_channels, cfg.rd) for _ in range(3)])
        self.bin_heads = nn.ModuleList([OrdinalBinHead(feat_channels, cfg.rd) for _ in range(3)])

    def forward(self, features: DepthRangeFeatures) -> RefinementTrace:
        trace = RefinementTrace()

        d_cur = self.initial_depth_head(features.levels[0])   # D_0, at target_res(0)
        trace.depths.append(d_cur)

        for i in range(3):
            f_i = features.levels[i]
            target_hw = f_i.shape[-2:]
            d_upsampled = F.interpolate(d_cur, size=target_hw, mode="bilinear", align_corners=False)

            c_i = self.confidence_heads[i](f_i, d_upsampled)       # Eq. 1: (B, rd, H_i, W_i), already sigmoid-ed
            s_i_logits = self.bin_heads[i](f_i)                     # ordinal bin logits, same shape
            trace.bins.append(s_i_logits)
            s_i = s_i_logits.sigmoid()                              # Eq. 2 needs S_i as a probability

            r_i = (c_i * s_i).sum(dim=1, keepdim=True)              # Eq. 2, summed over the bin dim
            e_i = 2.0 * (r_i - 1.0) * (self.max_depth / self.rd)     # Eq. 3
            d_cur = d_upsampled + e_i                                # Eq. 4 -> D_{i+1}
            trace.depths.append(d_cur)

        return trace
