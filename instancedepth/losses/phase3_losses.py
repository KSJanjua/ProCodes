"""Phase 3 refinement losses (paper Eq. 10-12; plan SS7).

    L_obj  (Eq. 10) = SigLog(D_hat, DT)          # dense, over VISIBLE GT ROI pixels
    L_dist (Eq. 11) = mean_p |(D_hat_i - D_hat_j)^2 - (DT_i - DT_j)^2|   # scalar, per valid pair
    L_ref  (Eq. 12) = lambda_obj * L_obj + lambda_dist * L_dist
    (+ optional holistic anti-forgetting term, plan SS7.4 -- OFF in faithful)

``L_obj`` reuses Phase 1's ``SigLogLoss`` unchanged (the exact Eigen scale-
invariant-log the paper cites for Eq. 10). Masking to visible GT only,
because a single RGB-D sensor has no ground truth for hidden pixels
(plan SS0.3).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from instancedepth.configs.phase3_config import Phase3LossConfig
from instancedepth.losses.hdi_losses import SigLogLoss
from instancedepth.models.phase3.output import RefinedDepthOutput
from instancedepth.models.phase3.targets import RefineTargets


class Phase3Criterion(nn.Module):
    def __init__(self, cfg: Phase3LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.silog = SigLogLoss(cfg.silog_lambda)

    def forward(
        self,
        output: RefinedDepthOutput,
        tgt: RefineTargets,
        gt_depth: torch.Tensor,          # (B,1,H2,W2) dense GT, 0 = invalid
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        # Always-available zero with a grad path into the trainable Phase-1
        # branch, so pair-less batches still produce a valid (zero-grad)
        # backward instead of a graph-less scalar.
        zero = output.base_depth.sum() * 0.0

        losses: Dict[str, torch.Tensor] = {}

        # --- L_obj (dense, Eq. 10) -----------------------------------------
        d_hat = output.d_hat_roi
        dt_valid = tgt.dt_valid
        if cfg.min_valid_roi_px > 0 and dt_valid.numel() > 0:
            # Skip members whose ROI has too few valid GT pixels (defect D5,
            # docs/PHASE3_DIAGNOSIS.md): 1-2 stray sensor returns give
            # maximally noisy per-instance supervision.
            counts = dt_valid.flatten(2).sum(-1)                     # (P,2)
            keep = counts >= cfg.min_valid_roi_px                     # (P,2)
            dt_valid = dt_valid & keep[:, :, None, None, None]
        if d_hat is not None and d_hat.shape[0] > 0 and dt_valid.any():
            l_obj = self.silog(d_hat, tgt.dt_dense, dt_valid)
        else:
            l_obj = zero
        losses["l_obj"] = cfg.lambda_obj * l_obj

        # --- L_dist (scalar, Eq. 11) ---------------------------------------
        rl, dt, pv = output.refined_layers, tgt.dt_scalar, tgt.pair_valid
        if rl.shape[0] > 0 and pv.any():
            pred_gap2 = (rl[:, 0] - rl[:, 1]) ** 2
            gt_gap2 = (dt[:, 0] - dt[:, 1]) ** 2
            l_dist = (pred_gap2 - gt_gap2).abs()[pv].mean()
        else:
            l_dist = zero
        losses["l_dist"] = cfg.lambda_dist * l_dist

        # --- optional anti-forgetting holistic regularizer (plan SS7.4) ----
        if cfg.holistic_weight > 0:
            mask = gt_depth > 0
            losses["l_holistic"] = cfg.holistic_weight * self.silog(output.base_depth, gt_depth, mask)

        losses["total"] = sum(losses.values()) if losses else zero
        return losses
