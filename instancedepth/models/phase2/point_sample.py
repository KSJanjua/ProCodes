"""Point-sampling utilities (PointRend, Kirillov et al. 2020), reimplemented
standalone here rather than depending on Detectron2 (which this project
deliberately avoids vendoring -- see the Phase 2 plan, SS3). This is the
same, well-known public algorithm Mask2Former's official matcher/criterion
use for memory-efficient mask supervision (point-sampled rather than dense
BCE+Dice); reimplemented directly against `torch.nn.functional.grid_sample`
so no extra heavy dependency is needed for it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    input : (N, C, H, W)
    point_coords : (N, P, 2), normalized to [0, 1]
    returns : (N, C, P)
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Uncertainty = -|logit| (least confident where the sigmoid is closest
    to 0.5). logits: (R, 1, ...) -> same shape."""
    assert logits.shape[1] == 1
    return -logits.abs()


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """coarse_logits: (N, 1, H, W) -> point_coords: (N, num_points, 2)."""
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = calculate_uncertainty(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx = idx + shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)],
            dim=1,
        )
    return point_coords
