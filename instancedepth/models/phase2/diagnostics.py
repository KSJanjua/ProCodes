"""Reading B diagnostic: mask-pooled Phase-1 depth,
logged during training as a comparison signal for the learned depth head
(Reading A) -- not trained on, purely a sanity check that the learned head
converges to something physically sensible.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mask_pooled_depth(mask_logits: torch.Tensor, holistic_depth: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    mask_logits : (B, N, H, W) pre-sigmoid Phase 2 mask predictions.
    holistic_depth : (B, 1, H, W) Phase 1's D_final (HolisticDepthOutput.depth_final),
        resized to match mask_logits' resolution by the caller if needed.

    Returns (B, N) -- mean Phase-1 depth within each predicted mask
    (Reading B's "average depth of the instance", computed by pooling
    rather than learned).
    """
    masks = (mask_logits.sigmoid() > threshold).float()          # (B, N, H, W)
    depth = holistic_depth.expand(-1, masks.shape[1], -1, -1)     # (B, N, H, W)
    pixel_count = masks.sum(dim=(-2, -1)).clamp_min(1.0)
    return (masks * depth).sum(dim=(-2, -1)) / pixel_count
