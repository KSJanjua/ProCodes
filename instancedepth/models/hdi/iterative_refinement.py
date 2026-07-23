"""Eq. 1-4 iterative bin refinement, across the 3 Depth Range Feature
Decoder levels.

This is the mathematical heart of Phase 1 — the paper's equations 1–4. Instead of predicting depth in one shot, it predicts a rough seed and then corrects it 3 times, each time at higher resolution. This "iterative refinement" is what makes the depth both globally correct and locally sharp. It's the single most important file in Phase 1.

Receives: DepthRangeFeatures([F_0,F_1,F_2]). Produces: a RefinementTrace holding every depth D_0..D_3 and every bin output S_0,S_1,S_2.

In: the 3 decoder feature maps.
Out: RefinementTrace(depths=[D_0,D_1,D_2,D_3], bins=[S_0,S_1,S_2]). D_3 is the final depth (before the last upsample). The intermediate D_1,D_2 and all S_i are kept for deep supervision during training.

This implements the paper's Eq. 1–4. We don't regress depth in one shot; we start from a seed depth and apply three coarse-to-fine correction rounds. In each round, a confidence head and an ordinal bin head produce C_i and S_i; we take their weighted sum over bins to get R_i, convert that into a metric correction E_i scaled by the bin width, and add it to the upsampled current depth to get the next depth. We keep every intermediate D_i and S_i because training uses deep supervision on them. 

Indexing follows the paper's own convention exactly: D_0 is the seed (from
`InitialDepthHead` on F_0); level i (i=0,1,2) consumes D_i and produces
D_{i+1}. D_3 is the final holistic depth (before the last upsample to full
input resolution, which lives in `model.py` -- this module operates at the
decoder's internal 1/8,1/4,1/2 resolutions only).

Eq. 1-4 are paper-specified; the resolution of the paper's 'i' index
collision (level index vs. bin-summation index) is this project's reading.
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


#  record of everything produced during the 3 rounds. Needed because training grades not just the final depth but the intermediate ones too.
@dataclass
class RefinementTrace:
    """Every D_i and S_i produced during the recurrence -- deep supervision
 needs D_1, D_2 (not just the seed D_0 or the final D_3), and all
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

# builds one InitialDepthHead and three confidence + three bin heads (one per level; not shared, because each level's features mean different things)
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
            d_upsampled = F.interpolate(d_cur, size=target_hw, mode="bilinear", align_corners=False) # take the current depth and enlarge it to this round's resolution.

            c_i = self.confidence_heads[i](f_i, d_upsampled)       # Eq. 1: (B, rd, H_i, W_i), already sigmoid-ed
            s_i_logits = self.bin_heads[i](f_i)                     # ordinal bin logits, same shape
            trace.bins.append(s_i_logits)
            s_i = s_i_logits.sigmoid()                              # Eq. 2 needs S_i as a probability

            r_i = (c_i * s_i).sum(dim=1, keepdim=True)              # Eq. 2, summed over the bin dim
            e_i = 2.0 * (r_i - 1.0) * (self.max_depth / self.rd)     # Eq. 3
            d_cur = d_upsampled + e_i                                # Eq. 4 -> D_{i+1}
            trace.depths.append(d_cur)

        return trace
