"""Bipartite matching cost extended with the depth-layer term (Eq. 5-7 of
the InstanceDepth paper): `min_sigma sum_i lambda1*Lm + lambda2*Lc + lambda3*Ld`.

`Lm`/`Lc` (mask + class cost) reproduce Mask2Former's own matcher exactly
(point-sampled BCE + Dice, CE-as-negative-probability) -- adapted from the
original `mask2former/modeling/matcher.py` read directly earlier in this
project, not re-derived from scratch. `Ld` (smooth-L1 depth-layer cost) is
this paper's own addition (weight not specified anywhere -- `cost_depth` is
a new hyperparameter, category 6, needs its own sweep).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from instancedepth.models.phase2.point_sample import point_sample


def batch_dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_ce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class Phase2HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        cost_depth: float = 1.0,
        num_points: int = 12544,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_depth = cost_depth
        self.num_points = num_points
        assert cost_class or cost_mask or cost_dice or cost_depth

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,   # (B, N, num_classes+1)
        mask_logits: torch.Tensor,    # (B, N, H, W)
        depth_preds: torch.Tensor,    # (B, N)
        targets: List[Dict[str, torch.Tensor]],  # per-image: labels (G,), masks (G,H,W), depths (G,)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = class_logits.shape[:2]
        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            if tgt_ids.numel() == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            out_prob = class_logits[b].softmax(-1)          # (N, num_classes+1)
            cost_class = -out_prob[:, tgt_ids]                # (N, G)

            out_mask = mask_logits[b][:, None]                # (N, 1, H, W)
            tgt_mask = targets[b]["masks"][:, None].to(out_mask)  # (G, 1, H, W)

            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            tgt_mask_pts = point_sample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_mask_pts = point_sample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).squeeze(1)

            with torch.autocast(device_type=out_mask.device.type, enabled=False):
                cost_mask = batch_sigmoid_ce_cost(out_mask_pts.float(), tgt_mask_pts.float())
                cost_dice = batch_dice_cost(out_mask_pts.float(), tgt_mask_pts.float())

            depth_pred = depth_preds[b][:, None]              # (N, 1)
            depth_gt = targets[b]["depths"][None, :]          # (1, G)
            cost_depth = F.smooth_l1_loss(
                depth_pred.expand(-1, depth_gt.shape[1]),
                depth_gt.expand(depth_pred.shape[0], -1),
                reduction="none",
            )

            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
                + self.cost_depth * cost_depth
            )
            C = C.reshape(num_queries, -1).cpu()
            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))

        return indices
