"""Streaming temporal stabilizer for video depth.

Architecture: the same residual ConvGRU recurrence the audit cleared of
blame (``instancedepth/models/hdi/temporal.py`` — zero-init output, per-clip
state reset, O(1) streaming state), with the two knobs the audit flagged
actually turned:

  * ``downsample`` 0.10 -> 0.25: the recurrence now sees a quarter-side grid,
    so it can correct mid-frequency flicker instead of only global drift.
  * trained with an explicit temporal loss (``losses/temporal_losses.py``) —
    the missing gradient that made the first attempt a measured no-op
    (TAE 0.05868 -> 0.05843).

Deliberately NOT Video-Depth-Anything's clip-attention head: that processes
32-frame windows jointly through the DPT head (heavy in memory and latency,
and unusable for streaming). This module is a few hundred K params, adds one
low-res GRU step per frame, and streams forever.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.models.hdi.temporal import ConvGRUCell


class TemporalStabilizer(nn.Module):
    """Residual temporal stabilizer for one decoder feature level.

    forward(F_t) -> F_t + Up(GRU(Down(F_t), H)); zero-init out-projection
    makes it an exact no-op at init, so the wrapped model can never start
    worse than its per-frame baseline. Call ``reset_state()`` at sequence
    boundaries. State auto-resets if batch/grid shape changes.
    """

    def __init__(self, feat_channels: int, d_model: int = 128,
                 num_blocks: int = 2, downsample: float = 0.25) -> None:
        super().__init__()
        assert 0.0 < downsample <= 1.0
        self.downsample = downsample
        self.proj_in = nn.Conv2d(feat_channels, d_model, 1)
        self.cells = nn.ModuleList([ConvGRUCell(d_model) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(d_model, feat_channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self._state: List[Optional[torch.Tensor]] = [None] * num_blocks

    def reset_state(self) -> None:
        self._state = [None] * len(self.cells)

    def detach_state(self) -> None:
        """Cut the autograd graph at the current state (truncated BPTT across
        clip boundaries while keeping the memory itself)."""
        self._state = [None if s is None else s.detach() for s in self._state]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h, w = feat.shape[-2:]
        dh = max(int(round(h * self.downsample)), 1)
        dw = max(int(round(w * self.downsample)), 1)
        x = F.interpolate(feat, size=(dh, dw), mode="bilinear", align_corners=False)
        x = self.proj_in(x)
        for i, cell in enumerate(self.cells):
            state = self._state[i]
            if state is not None and state.shape != x.shape:
                state = None
            x = cell(x, state)
            self._state[i] = x
        out = self.proj_out(x)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return feat + out


class TemporalStabilizerBank(nn.ModuleDict):
    """One stabilizer per configured decoder level, keyed by level index."""

    def __init__(self, levels: Sequence[int], feat_channels: int,
                 d_model: int, num_blocks: int, downsample: float) -> None:
        super().__init__({
            str(lvl): TemporalStabilizer(feat_channels, d_model, num_blocks, downsample)
            for lvl in levels
        })

    def reset_state(self) -> None:
        for m in self.values():
            m.reset_state()

    def detach_state(self) -> None:
        for m in self.values():
            m.detach_state()

    def apply_to(self, levels: List[torch.Tensor]) -> List[torch.Tensor]:
        for lvl_str, stab in self.items():
            lvl = int(lvl_str)
            levels[lvl] = stab(levels[lvl])
        return levels
