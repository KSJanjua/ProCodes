"""Depth Range Feature Decoder (paper Fig. 5).

Architecture, and exactly which parts are paper-derived vs. engineering
judgment, is documented in the implementation plan (SS1 and SS4). Summary:

- 3 coarse-to-fine levels, targeting 1/8, 1/4, 1/2 of the input resolution
  (Fig. 5's own labels) -- figure-inferred.
- Each level: project backbone features -> resize to the level's target
  resolution -> patchify (Conv2d, kernel/stride = level's patch size) ->
  linear projection -> patch attention (pre-LN multi-head self-attention +
  FFN) -> unpatchify -> fuse with the coarser level's (upsampled) output.
- Patch kernels (16, 8, 4 for levels 0, 1, 2) are given by Fig. 5; the
  resize-to-target-resolution and patchify steps are separated (rather than
  done in one strided conv) because the target resolutions are not integer
  multiples of the backbone's native 52x92 grid -- bilinear resize handles
  that exactly, and the patchify kernel then only needs to handle its own,
  much smaller, non-exact-division remainder (handled via padding, see
  `_pad_to_multiple`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.configs.config import DecoderConfig


def _pad_to_multiple(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Zero-pad the last two dims of ``x`` up to the next multiple of ``k``.
    Returns the padded tensor and (pad_h, pad_w) so the caller can crop back."""
    h, w = x.shape[-2:]
    pad_h = (k - h % k) % k
    pad_w = (k - w % k) % k
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)


class PreLNTransformerBlock(nn.Module):
    """Standard pre-LN multi-head self-attention + FFN block ("Patch
    Attention" in Fig. 5 -- the paper gives no internal formula for this
    block; this is the standard ViT-block convention, see plan SS4)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderLevel(nn.Module):
    """One coarse-to-fine level of the Depth Range Feature Decoder."""

    def __init__(
        self,
        in_channels: int,
        channels_dec: int,
        channels_attn: int,
        patch_kernel: int,
        attn_heads: int,
        attn_blocks: int,
        target_hw: Tuple[int, int],
        max_padded_tokens: int,
    ) -> None:
        super().__init__()
        self.patch_kernel = patch_kernel
        self.target_hw = target_hw

        self.project_in = nn.Conv2d(in_channels, channels_dec, kernel_size=1)
        self.patchify = nn.Conv2d(channels_dec, channels_attn, kernel_size=patch_kernel, stride=patch_kernel)
        # Fixed-shape learned positional embedding for this level's (padded)
        # token grid -- the paper specifies nothing about positional
        # encoding for "Patch Attention"; a learned embedding is the
        # simplest standard choice (category 6, plan SS4).
        self.pos_embed = nn.Parameter(torch.zeros(1, max_padded_tokens, channels_attn))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [PreLNTransformerBlock(channels_attn, attn_heads) for _ in range(attn_blocks)]
        )
        self.unpatchify = nn.ConvTranspose2d(
            channels_attn, channels_attn, kernel_size=patch_kernel, stride=patch_kernel
        )

    def forward(self, backbone_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        backbone_feat : (B, in_channels, h_bb, w_bb) -- native backbone
            resolution (same for every level, differs only by which
            transformer block it came from).

        Returns
        -------
        (B, channels_attn, *target_hw) per-pixel feature map for this level.
        """
        x = self.project_in(backbone_feat)
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        x, _ = _pad_to_multiple(x, self.patch_kernel)

        tokens = self.patchify(x)                                   # (B, C_attn, h_p, w_p)
        gh, gw = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)                  # (B, gh*gw, C_attn)
        n_tok = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :n_tok, :]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, gh, gw)
        out = self.unpatchify(tokens)                               # back to (padded_h, padded_w)
        # crop back to the exact target resolution (undo the patchify padding
        # from _pad_to_multiple; padded_h/padded_w == target_hw + pad_h/pad_w)
        out = out[:, :, : self.target_hw[0], : self.target_hw[1]]
        return out


@dataclass
class DepthRangeFeatures:
    """F_0, F_1, F_2 from the Depth Range Feature Decoder -- F_2 is the
    finest level, exposed to Phase 2/3 as ``feat_final`` (plan SS17)."""

    levels: List[torch.Tensor]

    @property
    def finest(self) -> torch.Tensor:
        return self.levels[-1]


class DepthRangeFeatureDecoder(nn.Module):
    """Fig. 5: 3 coarse-to-fine levels, top-down additive fusion."""

    def __init__(self, in_channels: int, image_hw: Tuple[int, int], cfg: DecoderConfig) -> None:
        super().__init__()
        assert len(cfg.target_fractions) == len(cfg.patch_kernels) == 3, (
            "Depth Range Feature Decoder is fixed at 3 levels (Fig. 5); "
            "target_fractions and patch_kernels must each have length 3."
        )
        H, W = image_hw
        self.target_hws: List[Tuple[int, int]] = []
        self.levels = nn.ModuleList()
        for frac, kernel in zip(cfg.target_fractions, cfg.patch_kernels):
            target_hw = (round(H * frac), round(W * frac))
            self.target_hws.append(target_hw)
            padded_h = math.ceil(target_hw[0] / kernel) * kernel
            padded_w = math.ceil(target_hw[1] / kernel) * kernel
            max_tokens = (padded_h // kernel) * (padded_w // kernel)
            self.levels.append(
                DecoderLevel(
                    in_channels=in_channels,
                    channels_dec=cfg.channels_dec,
                    channels_attn=cfg.channels_attn,
                    patch_kernel=kernel,
                    attn_heads=cfg.attn_heads,
                    attn_blocks=cfg.attn_blocks,
                    target_hw=target_hw,
                    max_padded_tokens=max_tokens,
                )
            )

    def forward(self, backbone_features: List[torch.Tensor]) -> DepthRangeFeatures:
        """
        Parameters
        ----------
        backbone_features : list of 3 tensors from ``DINOv2Backbone``,
            coarse-to-fine order (index 0 = shallowest selected block,
            index 2 = deepest/final selected block), each
            (B, embed_dim, h_bb, w_bb).

        Returns
        -------
        DepthRangeFeatures with F_0, F_1, F_2 (per-pixel maps at 1/8, 1/4,
        1/2 of input resolution respectively).
        """
        assert len(backbone_features) == 3
        outputs: List[torch.Tensor] = []
        prev: torch.Tensor | None = None
        for level_module, feat, target_hw in zip(self.levels, backbone_features, self.target_hws):
            level_out = level_module(feat)
            if prev is not None:
                prev_up = F.interpolate(prev, size=target_hw, mode="bilinear", align_corners=False)
                level_out = level_out + prev_up
            outputs.append(level_out)
            prev = level_out
        return DepthRangeFeatures(levels=outputs)
