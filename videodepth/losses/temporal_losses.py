"""Temporal-consistency losses for video depth — the piece InstanceDepth
never had.

The paper (Liang et al., ICCV 2025) is *entirely per-frame*: it enforces
geometric consistency **across instances within a frame**, never **across
frames**. A diagnosis of this repo showed the bolted-on
ConvGRU temporal module was inert precisely because **no loss ever rewarded
temporal smoothness** — the recurrence had state but no gradient telling it to
reduce flicker.

This module supplies that missing signal, following the two lineages the paper
only cites:

  * **Temporal Gradient Matching (TGM)** — Video Depth Anything (Chen et al.,
    2025). Instead of an optical-flow warping loss (which wrongly punishes the
    *legitimate* depth change of a moving object), match the prediction's
    frame-to-frame *temporal gradient* to the ground truth's:

        L_tgm = mean_valid | (d_t − d_{t-1}) − (dt_t − dt_{t-1}) |

    with d = log-depth (scale-consistent, same space as SigLog). A prediction
    that tracks GT's motion scores 0 regardless of how much the scene moves;
    only *flicker* (change GT does not have) is penalised. This is exactly the
    TAE diagnostic (``utils/metrics.py``) turned into a differentiable
    objective — we optimise the very quantity we report.

  * **Flow-warped consistency (optional, NVDS-style)** — used here only for the
    OPW *metric* and an optional inference-time stabiliser, NOT as a default
    training loss, for the moving-object reason above. Lives in
    ``videodepth/utils/temporal_metrics.py`` / ``flow.py``.

Everything is masked to pixels valid in *both* frames, so sensor holes never
manufacture a fake temporal gradient.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


def _log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp_min(_EPS))


class TemporalGradientMatchingLoss(nn.Module):
    """Video-Depth-Anything temporal gradient matching, in log-depth space.

    Parameters
    ----------
    order : 1 or 2
        1 = match first differences (d_t − d_{t-1}); 2 additionally matches
        second differences (acceleration), which suppresses higher-frequency
        jitter. VDA uses first order; second is an opt-in sharpener.
    log_space : bool
        Operate on log-depth (scale-consistent, matches SigLog) vs raw metres.
    """

    def __init__(self, order: int = 1, log_space: bool = True) -> None:
        super().__init__()
        assert order in (1, 2)
        self.order = order
        self.log_space = log_space

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,1,H,W) -> first temporal difference (B,T-1,1,H,W)
        return x[:, 1:] - x[:, :-1]

    def forward(self, pred: torch.Tensor, gt: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """pred / gt : (B,T,1,H,W) metric depth. valid : (B,T,1,H,W) bool
        (defaults to gt>0). Returns a scalar; 0 when the clip has <2 frames or
        no co-valid pixels (with a live grad path so backward never breaks)."""
        if pred.shape[1] < 2:
            return pred.sum() * 0.0
        if valid is None:
            valid = gt > 0

        p = _log(pred) if self.log_space else pred
        g = _log(gt) if self.log_space else gt

        dp, dg = self._delta(p), self._delta(g)
        # a temporal-gradient pixel is usable only where BOTH frames are valid
        vv = valid[:, 1:] & valid[:, :-1]
        total = pred.sum() * 0.0
        count = 0
        diff = (dp - dg).abs()[vv]
        if diff.numel():
            total = total + diff.sum()
            count += diff.numel()

        if self.order == 2 and pred.shape[1] >= 3:
            ddp, ddg = self._delta(dp), self._delta(dg)
            vvv = vv[:, 1:] & vv[:, :-1]
            d2 = (ddp - ddg).abs()[vvv]
            if d2.numel():
                total = total + d2.sum()
                count += d2.numel()

        return total / max(count, 1)


class FlowWarpConsistencyLoss(nn.Module):
    """Optional NVDS-style flow-warped consistency, gated to (near-)static
    pixels. OFF by default: warping D_{t-1} into frame t and demanding equality
    is *wrong* for a moving object whose metric depth genuinely changed, so we
    only apply it where the backward flow magnitude is below ``static_thresh``
    px (background / still regions). Provided for ablations and for datasets
    where TGM's GT-gradient supervision is unavailable.

    warp01 : (B,1,H,W) depth of frame t-1 already warped into frame t's grid
             (caller does the sampling with ``utils.flow.warp``).
    """

    def __init__(self, log_space: bool = True) -> None:
        super().__init__()
        self.log_space = log_space

    def forward(self, depth_t: torch.Tensor, warp01: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        if not valid.any():
            return depth_t.sum() * 0.0
        a = _log(depth_t) if self.log_space else depth_t
        b = _log(warp01) if self.log_space else warp01
        return (a - b).abs()[valid].mean()


class VideoDepthLoss(nn.Module):
    """Per-frame spatial loss (delegated) + temporal gradient matching.

    The spatial term is whatever single-frame criterion the base model already
    uses (Phase-1 ``HDILoss``); this wrapper adds ``temporal_weight * L_tgm``
    over the clip. Keeping the spatial loss pluggable means the temporal head
    is trained *jointly* with — never at the expense of — single-frame
    accuracy (the failure mode where a model goes smooth-but-wrong).
    """

    def __init__(self, spatial_loss: nn.Module, temporal_weight: float = 1.0,
                 tgm_order: int = 1, tgm_log_space: bool = True) -> None:
        super().__init__()
        self.spatial = spatial_loss
        self.temporal_weight = temporal_weight
        self.tgm = TemporalGradientMatchingLoss(order=tgm_order, log_space=tgm_log_space)

    def forward(self, pred_clip: torch.Tensor, gt_clip: torch.Tensor,
                spatial_terms: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """pred_clip / gt_clip : (B,T,1,H,W). ``spatial_terms`` = the already-
        computed per-frame losses (summed/averaged over the clip) to combine
        with the temporal term. Returns the full dict incl. 'total'."""
        out = dict(spatial_terms)
        l_tgm = self.tgm(pred_clip, gt_clip)
        out["temporal_gradient_matching"] = self.temporal_weight * l_tgm
        out["total"] = out.get("total", pred_clip.sum() * 0.0) + out["temporal_gradient_matching"]
        return out
