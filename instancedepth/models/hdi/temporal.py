"""Temporal alignment module for Phase 1 (FlashDepth Eq. 1-2 adaptation).

A lightweight recurrent module applied
to a decoder feature map, carrying hidden state across frames of a sequence:

    f, H_t   = Recur(Down(F), H_{t-1})       # recurrence over downsampled grid
    F_align  = F + Up(f)                     # residual add   [Paper Specified]

Key properties (all FlashDepth [Paper Specified] unless noted):

  * **Zero-initialized output projection** -> the module contributes exactly 0
    at initialization, so the model *is* the pretrained per-frame model at
    training step 0 and an empty-memory first frame can never underperform it.
  * **Zero hidden state at every sequence start** (no learned initial state)
    -- reset via ``reset_state()``, the ``start_new_sequence()`` analogue.
  * ~10 % per-side downsampling before the recurrence ("dense pixel-level
    features are unnecessary for aligning depths").
  * Recurrent core: **ConvGRU** [Reasonable Assumption] -- FlashDepth defaults
    to Mamba but its own code treats the core as swappable; ConvGRU needs no
    compiled CUDA kernels (mamba-ssm is install-fragile on the training
    server) and gives per-*location* temporal memory with identical streaming
    / O(1)-state properties.

State-safety details: the held state is part of the autograd graph within a
training clip (full BPTT over the clip); callers reset per clip / per
sequence. The state auto-resets if the incoming batch size or grid shape
changes (e.g. switching between training and evaluation resolutions).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """Convolutional GRU (Ballas et al. 2016): per-pixel gated temporal memory
    with 3x3 spatial context in the gates -- the standard recurrent update
    operator in vision models (e.g. RAFT)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.cand = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        if h is None:
            h = torch.zeros_like(x)
        zr = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * c


class TemporalAligner(nn.Module):
    """Residual temporal aligner for one decoder feature level.

    forward(F) -> F_aligned, carrying hidden state across successive calls
    (one call per video frame). Call ``reset_state()`` at sequence starts.
    """

    def __init__(self, feat_channels: int, d_model: int = 128,
                 num_blocks: int = 2, downsample: float = 0.1) -> None:
        super().__init__()
        assert 0.0 < downsample <= 1.0
        self.downsample = downsample
        self.proj_in = nn.Conv2d(feat_channels, d_model, 1)
        self.cells = nn.ModuleList([ConvGRUCell(d_model) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(d_model, feat_channels, 1)
        # Zero-init output projection: the aligner starts as an exact no-op
        # (FlashDepth's stabilizer -- see module docstring).
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self._state: List[Optional[torch.Tensor]] = [None] * num_blocks

    def reset_state(self) -> None:
        """Zero the temporal memory (the start_new_sequence() analogue)."""
        self._state = [None] * len(self.cells)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h, w = feat.shape[-2:]
        dh = max(int(round(h * self.downsample)), 1)
        dw = max(int(round(w * self.downsample)), 1)
        x = F.interpolate(feat, size=(dh, dw), mode="bilinear", align_corners=False)
        x = self.proj_in(x)

        for i, cell in enumerate(self.cells):
            state = self._state[i]
            if state is not None and state.shape != x.shape:
                state = None   # batch/resolution changed -> fresh memory
            x = cell(x, state)
            self._state[i] = x

        out = self.proj_out(x)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return feat + out
