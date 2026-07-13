"""Prediction heads for the iterative bin refinement (Eq. 1-4).

Head architectures ("a small network" / "a lightweight convolutional
network" in the paper, with no further detail) are a own-engineering-
decision, DPT-output-head-scale conv stacks. Their *existence*, *inputs*, and (for the confidence head) their
*sigmoid activation* are paper-explicit (Eq. 1).

Activation for the bin head (``OrdinalBinHead``) is **sigmoid, not
softmax**: a softmax reading forces the Eq. 2-4 correction to be
one-directional, which the equations don't support. The
independent-per-bin-sigmoid reading matches Fu et al.'s ordinal regression
(DORN, CVPR 2018), directly cited by InstanceDepth (ref [18]).

``OrdinalBinHead`` itself returns **raw logits**, not the sigmoid-activated
probability. The sigmoid is applied explicitly wherever a probability is
actually needed (``IterativeBinRefinement``'s R_i computation, Eq. 2) and
on demand by output consumers (``HolisticDepthOutput.seg_confidence()``).
Keeping the loss (``losses/hdi_losses.py``'s ``OrdinalBinBCE``) on logits
via ``binary_cross_entropy_with_logits`` is not just an autocast
requirement (plain ``binary_cross_entropy`` on an already-sigmoided tensor
is explicitly unsafe under mixed precision) -- it's also more numerically
stable in general, since it avoids computing ``log`` of an already-saturated
sigmoid output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _small_conv_head(in_channels: int, out_channels: int, hidden: int | None = None) -> nn.Sequential:
    hidden = hidden or max(in_channels // 2, out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(32, hidden), num_channels=hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, kernel_size=1),
    )


class InitialDepthHead(nn.Module):
    """Produces the seed depth D_0 from the coarsest level's features F_0.
    Positivity via softplus (smoother than DAv2's plain ReLU near 0; same
    role -- ensure a physically valid, non-negative depth seed)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = _small_conv_head(channels, 1)
        self.softplus = nn.Softplus()

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.head(f0))


class ConfidenceHead(nn.Module):
    """Phi_d in Eq. 1: C_i = Sigmoid(Phi_d(F_i, D_i))."""

    def __init__(self, feat_channels: int, rd: int) -> None:
        super().__init__()
        self.head = _small_conv_head(feat_channels + 1, rd)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_i: torch.Tensor, d_i: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_i, d_i], dim=1)
        return self.sigmoid(self.head(x))


class OrdinalBinHead(nn.Module):
    """S_i in Eq. 2-4: independent per-bin sigmoid (ordinal encoding) --
    NOT a softmax over mutually-exclusive classes.

    Returns raw logits (see module docstring); callers apply ``.sigmoid()``
    where a probability is actually needed."""

    def __init__(self, feat_channels: int, rd: int) -> None:
        super().__init__()
        self.head = _small_conv_head(feat_channels, rd)

    def forward(self, f_i: torch.Tensor) -> torch.Tensor:
        return self.head(f_i)
