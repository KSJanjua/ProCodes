"""Phase2Output -- the contract Phase 3 consumes.
Mirrors instancedepth/models/hdi/output.py's HolisticDepthOutput pattern:
dataclass, contract_version, deliberately over-exposing (query_embeddings
isn't paper-required but is cheap to keep for future use) rather than
under-exposing, since Phase 3 doesn't exist yet to push back on the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

CONTRACT_VERSION = "1.0"


@dataclass
class Phase2Output:
    mask_logits: torch.Tensor       # (B, N, H, W) pre-sigmoid, full input resolution
    class_logits: torch.Tensor      # (B, N, num_classes+1)
    depth_layers: torch.Tensor      # (B, N) predicted Dep_i (Eq. 5-7 head)
    query_embeddings: torch.Tensor  # (B, N, hidden_dim) -- not paper-required, kept for future use
    image_hw: Tuple[int, int]
    contract_version: str = CONTRACT_VERSION

    def scores(self) -> torch.Tensor:
        """(B, N) foreground class confidence -- softmax over classes,
        excluding the no-object slot, max over remaining classes. Used for
        the >0.9 category-confidence filter."""
        probs = self.class_logits.softmax(-1)[..., :-1]
        return probs.max(-1).values

    def mask_confidence(self) -> torch.Tensor:
        """(B, N, H, W) sigmoid mask probability -- for the >0.8 mask-
        confidence filter (same Eq. 8 context)."""
        return self.mask_logits.sigmoid()
