"""Depth-layer head (Eq. 5-7's `Dep_i`), Reading A from the Phase 2 plan
(SS1.1): a simple MLP on each query embedding, analogous to Mask2Former's
own existing classification head. Reading B (mask-pooled Phase-1 depth) is
logged as a diagnostic only -- see `diagnostics.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthLayerHead(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.ReLU(inplace=True),
            nn.Linear(inner, 1),
        )
        # Positivity, matching this project's Phase 1 convention for depth
        # outputs (instancedepth/models/hdi/bin_heads.py's InitialDepthHead).
        self.softplus = nn.Softplus()

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """query_embeddings: (B, N, hidden_dim) -> (B, N) depth per query."""
        return self.softplus(self.mlp(query_embeddings)).squeeze(-1)
