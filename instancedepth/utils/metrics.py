"""Depth evaluation metrics matching the paper's Table 2/3 definitions
(standard KITTI/NYU-style protocol: RMS, REL, RMSlog, Log10, sigma1-3),
plus optional canonical-disparity diagnostics."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from instancedepth.configs.config import CameraIntrinsics
from instancedepth.utils.camera import depth_to_canonical_disparity

_EPS = 1e-6


@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    pred = pred[mask].clamp_min(_EPS)
    gt = gt[mask].clamp_min(_EPS)
    if pred.numel() == 0:
        return {k: float("nan") for k in ("abs_rel", "rms", "rms_log", "log10", "sigma1", "sigma2", "sigma3")}

    abs_rel = ((pred - gt).abs() / gt).mean().item()
    rms = torch.sqrt(((pred - gt) ** 2).mean()).item()
    rms_log = torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean()).item()
    log10 = (torch.log10(pred) - torch.log10(gt)).abs().mean().item()

    ratio = torch.maximum(pred / gt, gt / pred)
    sigma1 = (ratio < 1.25).float().mean().item()
    sigma2 = (ratio < 1.25 ** 2).float().mean().item()
    sigma3 = (ratio < 1.25 ** 3).float().mean().item()

    return dict(abs_rel=abs_rel, rms=rms, rms_log=rms_log, log10=log10,
                sigma1=sigma1, sigma2=sigma2, sigma3=sigma3)


@torch.no_grad()
def temporal_alignment_error(pred: torch.Tensor, pred_prev: torch.Tensor,
                             gt: torch.Tensor, gt_prev: torch.Tensor,
                             mask: torch.Tensor) -> float:
    """First-order temporal consistency: mean |dpred - dgt| over pixels valid
    in BOTH frames, where dx = x_t - x_{t-1}. A prediction that tracks GT
    perfectly scores 0 regardless of camera/object motion (the motion appears
    in both deltas), so no optical flow is needed -- valid here because the
    dataset has per-frame metric GT. Returns NaN when no pixel is valid."""
    diff = ((pred - pred_prev) - (gt - gt_prev)).abs()[mask]
    return float(diff.mean()) if diff.numel() else float("nan")


@torch.no_grad()
def compute_disparity_diagnostics(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, intrinsics: CameraIntrinsics
) -> Optional[Dict[str, float]]:
    """Diagnostic-only near-field metrics; returns None if
    intrinsics aren't resolvable rather than guessing a camera constant."""
    if intrinsics.focal_px is None or intrinsics.width_px is None:
        return None
    disp_pred = depth_to_canonical_disparity(pred, intrinsics)[mask]
    disp_gt = depth_to_canonical_disparity(gt, intrinsics)[mask]
    if disp_pred.numel() == 0:
        return dict(disparity_abs_rel=float("nan"), disparity_rmse=float("nan"))
    abs_rel = ((disp_pred - disp_gt).abs() / disp_gt.clamp_min(_EPS)).mean().item()
    rmse = torch.sqrt(((disp_pred - disp_gt) ** 2).mean()).item()
    return dict(disparity_abs_rel=abs_rel, disparity_rmse=rmse)
