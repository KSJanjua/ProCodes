"""Final training loss for Phase 2, computed on the pairs the matcher
(matcher.py) assigns: mask (point-sampled BCE + Dice, uncertainty-based
importance sampling) and class (CE with a down-weighted no-object class)
reproduce Mask2Former's own `SetCriterion` (adapted from
`mask2former/modeling/criterion.py`, read directly earlier in this
project); depth (smooth-L1 on `Dep_i`, Eq. 5-7) is this paper's own
addition, applied only to matched (non-"no-object") pairs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.models.phase2.point_sample import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


class Phase2Criterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        eos_coef: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        weight_class: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        weight_depth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.weight_class = weight_class
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.weight_depth = weight_depth

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, class_logits, targets, indices) -> torch.Tensor:
        idx = self._src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][col] for t, (_, col) in zip(targets, indices)])
        target_classes = torch.full(
            class_logits.shape[:2], self.num_classes, dtype=torch.int64, device=class_logits.device
        )
        target_classes[idx] = target_classes_o
        return F.cross_entropy(class_logits.transpose(1, 2), target_classes, self.empty_weight)

    def loss_masks_and_dice(self, mask_logits, targets, indices, num_masks: float) -> Dict[str, torch.Tensor]:
        src_idx = self._src_permutation_idx(indices)
        src_masks = mask_logits[src_idx][:, None]   # (M, 1, H, W)
        tgt_masks = torch.cat([t["masks"][col] for t, (_, col) in zip(targets, indices)], dim=0)[:, None].to(src_masks)

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, self.num_points, self.oversample_ratio, self.importance_sample_ratio,
            )
            point_labels = point_sample(tgt_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)

        return {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

    def loss_depth(self, depth_preds, targets, indices, num_masks: float) -> torch.Tensor:
        idx = self._src_permutation_idx(indices)
        src_depths = depth_preds[idx]
        tgt_depths = torch.cat([t["depths"][col] for t, (_, col) in zip(targets, indices)])
        if src_depths.numel() == 0:
            return depth_preds.sum() * 0.0
        return F.smooth_l1_loss(src_depths, tgt_depths, reduction="sum") / num_masks

    def forward(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        depth_preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        num_masks = max(sum(len(t["labels"]) for t in targets), 1)

        losses = {
            "loss_class": self.weight_class * self.loss_labels(class_logits, targets, indices),
            "loss_depth": self.weight_depth * self.loss_depth(depth_preds, targets, indices, num_masks),
        }
        mask_losses = self.loss_masks_and_dice(mask_logits, targets, indices, num_masks)
        losses["loss_mask"] = self.weight_mask * mask_losses["loss_mask"]
        losses["loss_dice"] = self.weight_dice * mask_losses["loss_dice"]
        losses["total"] = sum(losses.values())
        return losses
