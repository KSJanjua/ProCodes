"""Phase 1 (Holistic Depth Initialization) losses.

The paper specifies NO training loss for this stage anywhere (Sec. 4.1 /
Eq. 1-4 are forward-computation only) -- every loss term here is this
project's own engineering decision, justified in the plan (SS2/SS6) by
analogy to Eq. 10 (the *same* paper's instance-rectification stage, one
step later) and to Fu et al.'s DORN (CVPR 2018, cited by InstanceDepth as
ref [18]). See the plan for the full comparison against Depth Anything V2 /
Video Depth Anything / AdaBins / BerHu / plain L1-L2 and why each was or
wasn't adopted.

    L_hdi = SigLog(D_final, GT)
          + sum_{i=1,2} w_i * SigLog(D_i, GT)              # deep supervision, D_1/D_2 only
          + lambda_ce * sum_i BCE_ordinal(S_i, GT)          # i = 0,1,2
          + lambda_disp * L1(disp(D_final), disp(GT))       # optional, hdi_enhanced.yaml only

Regression term is pluggable (``REGRESSION_LOSSES`` registry: silog / l1 /
l2 / berhu) precisely because the SigLog choice is an inference, not a
paper fact -- swapping it for an ablation should not require touching the
trainer.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.configs.config import CameraIntrinsics, LossConfig
from instancedepth.utils.camera import depth_to_canonical_disparity

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Regression loss registry
# --------------------------------------------------------------------------- #
class SigLogLoss(nn.Module):
    """Eigen, Puhrsch & Fergus (NeurIPS 2014) scale-invariant log loss --
    the exact form InstanceDepth itself cites for Eq. 10, and the one
    verified directly in Depth Anything V2's ``util/loss.py``
    (``SiLogLoss``): sqrt(mean(d^2) - lambda*mean(d)^2), d = log(pred)-log(gt).
    No scale/shift alignment step (unlike MiDaS/DAv2's ``L_ssi``) -- see
    plan SS6 for why that distinction matters here.
    """

    def __init__(self, lam: float = 0.5) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp_min(_EPS)
        target = target.clamp_min(_EPS)
        diff_log = torch.log(target[mask]) - torch.log(pred[mask])
        if diff_log.numel() == 0:
            return pred.sum() * 0.0
        return torch.sqrt(torch.clamp(
            diff_log.pow(2).mean() - self.lam * diff_log.mean().pow(2), min=0.0
        ) + _EPS)


class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.l1_loss(pred[mask], target[mask])


class L2Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], target[mask])


class BerHuLoss(nn.Module):
    """Laina et al. 2016 (reverse Huber): L1 below a per-batch adaptive
    threshold, L2 above it."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return pred.sum() * 0.0
        diff = (pred[mask] - target[mask]).abs()
        c = 0.2 * diff.max().detach()
        l2_branch = (diff.pow(2) + c.pow(2)) / (2 * c + _EPS)
        return torch.where(diff <= c, diff, l2_branch).mean()


REGRESSION_LOSSES: Dict[str, Callable[[], nn.Module]] = {
    "silog": SigLogLoss,
    "l1": L1Loss,
    "l2": L2Loss,
    "berhu": BerHuLoss,
}


# --------------------------------------------------------------------------- #
# Ordinal bin supervision (plan SS5/SS6 -- own construction, DORN-analogous)
# --------------------------------------------------------------------------- #
def ordinal_bin_targets(gt_depth: torch.Tensor, rd: int, max_depth: float) -> torch.Tensor:
    """
    Parameters
    ----------
    gt_depth : (B,1,H,W) metric depth.

    Returns
    -------
    (B,rd,H,W) target_b = 1{gt_depth > b * (max_depth/rd)} for b=0..rd-1 --
    an ordinal/cumulative encoding (NOT one-hot/mutually-exclusive), matching
    the independent-sigmoid reading of S_i (plan SS5). ``S_i``'s own
    ground-truth semantics are not stated anywhere in the paper (Eq. 1-4 has
    no loss at all); this construction is this project's own, chosen for
    consistency with Eq. 2-4's arithmetic and DORN's ordinal-regression
    precedent.
    """
    bin_width = max_depth / rd
    thresholds = torch.arange(rd, device=gt_depth.device, dtype=gt_depth.dtype) * bin_width
    thresholds = thresholds.view(1, rd, 1, 1)
    return (gt_depth > thresholds).float()


class OrdinalBinBCE(nn.Module):
    def __init__(self, rd: int, max_depth: float) -> None:
        super().__init__()
        self.rd = rd
        self.max_depth = max_depth

    def forward(self, pred_bins: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = ordinal_bin_targets(gt_depth, self.rd, self.max_depth)
        mask_b = mask.expand_as(pred_bins)
        if mask_b.sum() == 0:
            return pred_bins.sum() * 0.0
        return F.binary_cross_entropy(pred_bins[mask_b], target[mask_b])


# --------------------------------------------------------------------------- #
# Combined Phase 1 loss
# --------------------------------------------------------------------------- #
class HDILoss(nn.Module):
    def __init__(self, cfg: LossConfig, rd: int, max_depth: float,
                 camera: Optional[CameraIntrinsics] = None) -> None:
        super().__init__()
        if cfg.regression not in REGRESSION_LOSSES:
            raise ValueError(f"Unknown loss.regression '{cfg.regression}', expected one of {list(REGRESSION_LOSSES)}")
        self.regression = REGRESSION_LOSSES[cfg.regression]()
        self.deep_supervision_weights = cfg.deep_supervision_weights
        self.bin_bce_weight = cfg.bin_bce_weight
        self.disparity_aux_weight = cfg.disparity_aux_weight
        self.camera = camera
        self.bin_bce = OrdinalBinBCE(rd, max_depth)
        if self.disparity_aux_weight > 0:
            assert camera is not None and camera.focal_px and camera.width_px, (
                "HDILoss constructed with disparity_aux_weight > 0 but no "
                "usable camera intrinsics were provided (see plan SS9)."
            )

    def forward(
        self,
        depth_final: torch.Tensor,
        seg_final: torch.Tensor,
        depth_levels: List[torch.Tensor],   # [D_1, D_2] at their own (smaller) resolutions
        seg_levels: List[torch.Tensor],     # [S_0, S_1, S_2] at their own resolutions
        gt_depth: torch.Tensor,             # (B,1,H,W) full resolution, 0 = invalid
    ) -> Dict[str, torch.Tensor]:
        mask = gt_depth > 0
        losses: Dict[str, torch.Tensor] = {}

        losses["regression_final"] = self.regression(depth_final, gt_depth, mask)

        for i, (d_i, w_i) in enumerate(zip(depth_levels, self.deep_supervision_weights)):
            gt_i = F.interpolate(gt_depth, size=d_i.shape[-2:], mode="nearest")
            mask_i = gt_i > 0
            losses[f"regression_deep_{i}"] = w_i * self.regression(d_i, gt_i, mask_i)

        bin_bce_total = depth_final.sum() * 0.0
        for s_i in seg_levels:
            gt_i = F.interpolate(gt_depth, size=s_i.shape[-2:], mode="nearest")
            mask_i = gt_i > 0
            bin_bce_total = bin_bce_total + self.bin_bce(s_i, gt_i, mask_i)
        losses["bin_bce"] = self.bin_bce_weight * bin_bce_total

        if self.disparity_aux_weight > 0:
            disp_pred = depth_to_canonical_disparity(depth_final, self.camera)
            disp_gt = depth_to_canonical_disparity(gt_depth.clamp_min(_EPS), self.camera)
            disp_diff = (disp_pred[mask] - disp_gt[mask]).abs().mean() if mask.any() else depth_final.sum() * 0.0
            losses["disparity_aux"] = self.disparity_aux_weight * disp_diff

        losses["total"] = sum(losses.values())
        return losses
