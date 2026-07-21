"""Ground-truth targets for the Phase 3 refinement loss.

Two supervision signals (paper Eq. 10-11):

  * L_dist (Eq. 11) needs the SCALAR GT depth layer of each matched pair
    member -> reuses the already-implemented ``build_refine_targets`` in
    ``data/gid_dataset.py`` verbatim (not duplicated here).

  * L_obj (Eq. 10, dense reading) needs the DENSE GT depth ROI of each
    matched member -> ``build_dense_gt_rois`` below ROIAligns the GT depth
    (masked to the member's matched GT instance) into the same Hp x Wp
    normalized ROI frame as D_hat.

Physics: a single RGB-D sensor gives GT only on *visible*
surfaces, so ``dt_valid`` is (GT depth > 0) AND (inside the matched GT
instance mask). There is no GT for truly-hidden pixels; L_obj is masked
to visible GT accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from instancedepth.data.gid_dataset import build_refine_targets
from instancedepth.models.phase3.candidates import PairSet
from instancedepth.models.phase3.roi_extract import roi_align_per_instance


@dataclass
class RefineTargets:
    dt_scalar: torch.Tensor    # (P,2) GT depth layers [main,guest]
    pair_valid: torch.Tensor   # (P,) bool -- both members matched to GT
    dt_dense: torch.Tensor     # (P,2,1,Hp,Wp) GT depth ROI (0 where invalid/unmatched)
    dt_valid: torch.Tensor     # (P,2,1,Hp,Wp) bool -- visible GT inside matched instance


@torch.no_grad()
def build_dense_gt_rois(
    pairs: PairSet,
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    gt_depth: torch.Tensor,          # (B,1,H2,W2) dense GT depth, 0 = invalid
    out_hw: Tuple[int, int],
    sampling_ratio: int,
) -> RefineTargets:
    device = gt_depth.device
    P = len(pairs)

    dt_scalar, pair_valid = build_refine_targets(
        pairs.query_idx, pairs.batch_index, indices, targets
    )
    dt_scalar, pair_valid = dt_scalar.to(device), pair_valid.to(device)

    Hp, Wp = out_hw
    if P == 0:
        z = gt_depth.new_zeros((0, 2, 1, Hp, Wp))
        return RefineTargets(dt_scalar, pair_valid, z, z.bool())

    # per-image pred-query -> gt-index lookup (from the matcher)
    lut: List[Dict[int, int]] = []
    for (pi, gi) in indices:
        lut.append({int(p): int(g) for p, g in zip(pi.tolist(), gi.tolist())})

    H2, W2 = gt_depth.shape[-2:]
    boxes_flat = pairs.boxes_norm.reshape(2 * P, 4)
    dense_maps = gt_depth.new_zeros((2 * P, 1, H2, W2))
    valid_maps = gt_depth.new_zeros((2 * P, 1, H2, W2))
    for m in range(2 * P):
        p, k = m // 2, m % 2
        b = int(pairs.batch_index[p])
        q = int(pairs.query_idx[p, k])
        g = lut[b].get(q, None)
        if g is None:
            continue
        inst_mask = targets[b]["masks"][g].to(device) > 0.5          # (H2,W2)
        depth_b = gt_depth[b, 0]                                      # (H2,W2)
        valid = inst_mask & (depth_b > 0)
        dense_maps[m, 0] = depth_b * valid
        valid_maps[m, 0] = valid.float()

    dt_num = roi_align_per_instance(dense_maps, boxes_flat, out_hw, sampling_ratio)
    dt_valid_soft = roi_align_per_instance(valid_maps, boxes_flat, out_hw, sampling_ratio)
    # Normalized convolution: dense_maps
    # is depth*valid (zeros outside the instance), so a plain ROIAlign dilutes
    # boundary cells toward zero by their invalid fraction -- a 3 m person's
    # edge cells got 1.6-2.9 m targets, training Phi_o to push depth down at
    # instance boundaries. Dividing by the aligned valid-fraction recovers the
    # true local mean depth for every cell that passes the 0.5 validity gate.
    dt_dense = dt_num / dt_valid_soft.clamp_min(1e-6)
    dt_dense = dt_dense.reshape(P, 2, 1, Hp, Wp)
    dt_valid = (dt_valid_soft.reshape(P, 2, 1, Hp, Wp) >= 0.5) & (dt_dense > 0)
    dt_dense = dt_dense * dt_valid   # zero out invalid cells (matches docstring contract)
    return RefineTargets(dt_scalar, pair_valid, dt_dense, dt_valid)
