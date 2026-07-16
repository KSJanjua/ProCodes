"""Improved occlusion-pair relation head — drop-in for InstanceDepth's Φo.

Interface-compatible with ``instancedepth.models.phase3.relation_head
.OcclusionRelationHead`` (forward(f_obj, g_obj, d_obj) -> (e_obj, d_hat), same
shapes, e_obj in (0,1), d_hat = 2·e_obj·d_obj), so the existing Phase-3
compositor, losses, and training loop work unchanged. Three targeted fixes
over the paper's MLP head (Eq. 8):

1. **Spatial context.** The original trunk is 1×1 convs only — every ROI cell
   is judged in isolation, so the head cannot even see *where* the overlap
   is. Here: 3×3 conv encoder per member first.

2. **Explicit pair cross-attention.** The original couples the pair by
   channel concat only. Here each member's ROI tokens cross-attend to the
   other member's tokens (a single-head attention over the Hp×Wp grid), so
   "how deep am I?" is answered by *looking at the instance occluding me* —
   genuine relation reasoning, at negligible cost (ROI grids are 28×28).

3. **Bounded correction.** The original E ∈ (0,1) means the ratio 2E spans
   (0,2): the head can halve or double an instance's depth, and any confident
   mistake paints a hard ratio step at the mask boundary (the ring artifact,
   docs/AUDIT_2026.md §3.2). Here the correction is
       ratio = 1 + max_corr·tanh(z)   (⇒ e_obj = ratio/2 ∈ (0.5±max_corr/2))
   with zero-init z ⇒ exact identity at init, and a hard cap on how far the
   refinement can ever push depth (default ±15 %) — occlusion corrections in
   this data are small layer shifts, never depth doublings. Rings are bounded
   by construction; a mis-trained head degrades gracefully instead of
   catastrophically.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class PairCrossAttention(nn.Module):
    """Single-head cross-attention between the two members' ROI token grids.
    x: (P,2,C,Hp,Wp) -> same shape, each member attending to the other."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # zero-init projection: attention starts as a no-op residual
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P, two, C, Hp, Wp = x.shape
        flat = x.reshape(P * 2, C, Hp, Wp)
        q = self.q(flat).reshape(P, 2, C, Hp * Wp)
        k = self.k(flat).reshape(P, 2, C, Hp * Wp)
        v = self.v(flat).reshape(P, 2, C, Hp * Wp)
        # swap members: member 0 queries member 1's keys/values and vice versa
        k, v = k.flip(1), v.flip(1)
        attn = torch.einsum("pmci,pmcj->pmij", q, k) / math.sqrt(C)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("pmij,pmcj->pmci", attn, v)
        out = self.proj(out.reshape(P * 2, C, Hp, Wp)).reshape(P, 2, C, Hp, Wp)
        return x + out


class BoundedPairAttentionHead(nn.Module):
    """Drop-in Φo replacement: conv encoder + pair cross-attention + bounded
    multiplicative correction."""

    def __init__(self, per_member_channels: int, hidden_dim: int = 256,
                 num_conv: int = 3, granularity: str = "dense",
                 max_corr: float = 0.15) -> None:
        super().__init__()
        assert granularity in ("dense", "scalar")
        assert 0.0 < max_corr < 1.0
        self.granularity = granularity
        self.max_corr = max_corr

        enc = [nn.Conv2d(per_member_channels, hidden_dim, 3, padding=1), nn.GELU()]
        for _ in range(max(num_conv - 2, 0)):
            enc += [nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU()]
        self.encoder = nn.Sequential(*enc)          # applied per member
        self.cross = PairCrossAttention(hidden_dim)
        self.out = nn.Conv2d(hidden_dim, 1, 1)      # correction logit per member
        nn.init.zeros_(self.out.weight)             # identity at init
        nn.init.zeros_(self.out.bias)

    def forward(self, f_obj: torch.Tensor, g_obj: torch.Tensor,
                d_obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """f_obj (P,2,C,Hp,Wp), g_obj (P,2,Gc,Hp,Wp), d_obj (P,2,1,Hp,Wp)
        -> (e_obj, d_hat), both (P,2,1,Hp,Wp), Eq. 8/9-compatible."""
        P = f_obj.shape[0]
        Hp, Wp = f_obj.shape[-2:]
        if P == 0:
            z = f_obj.new_zeros((0, 2, 1, Hp, Wp))
            return z, z

        x = torch.cat([f_obj, g_obj], dim=2)                        # (P,2,C+Gc,Hp,Wp)
        M = x.shape[2]
        feat = self.encoder(x.reshape(P * 2, M, Hp, Wp))
        feat = feat.reshape(P, 2, -1, Hp, Wp)
        feat = self.cross(feat)                                     # look at the other member
        z = self.out(feat.reshape(P * 2, -1, Hp, Wp)).reshape(P, 2, 1, Hp, Wp)

        if self.granularity == "scalar":
            z = z.mean(dim=(-2, -1), keepdim=True).expand(P, 2, 1, Hp, Wp)

        ratio = 1.0 + self.max_corr * torch.tanh(z)                 # (1±max_corr)
        e_obj = 0.5 * ratio                                         # ratio = 2E (Eq. 9 algebra)
        d_hat = ratio * d_obj
        return e_obj, d_hat
