"""Loss correctness tests, including a regression test of the
SigLog/canonical-disparity equivalence proof -- if this test ever fails,
either the proof was wrong or someone changed the disparity auxiliary loss
to use a log-space formulation (which is redundant and should not be
reintroduced).

Run:  pytest tests/test_hdi_losses.py -v
"""

from __future__ import annotations

import torch

from instancedepth.configs.config import CameraIntrinsics, LossConfig
from instancedepth.losses.hdi_losses import HDILoss, SigLogLoss, ordinal_bin_targets
from instancedepth.utils.camera import depth_to_canonical_disparity


def test_silog_zero_for_identical_input():
    loss_fn = SigLogLoss()
    depth = torch.rand(2, 1, 8, 8) * 5 + 0.1
    mask = torch.ones_like(depth, dtype=torch.bool)
    loss = loss_fn(depth, depth, mask)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)


def test_silog_on_disparity_equals_silog_on_depth():
    """The equivalence proof, as an executable check: log(C/pred)-log(C/gt) ==
    -(log(pred)-log(gt)); since SiLog only uses squared terms of that
    difference, the two losses must be numerically identical."""
    torch.manual_seed(0)
    pred = torch.rand(4, 1, 16, 16) * 8 + 0.1
    gt = torch.rand(4, 1, 16, 16) * 8 + 0.1
    mask = torch.ones_like(gt, dtype=torch.bool)

    intrinsics = CameraIntrinsics(focal_px=860.0, width_px=1288, source="config")
    disp_pred = depth_to_canonical_disparity(pred, intrinsics)
    disp_gt = depth_to_canonical_disparity(gt, intrinsics)

    loss_fn = SigLogLoss()
    loss_depth = loss_fn(pred, gt, mask)
    loss_disp = loss_fn(disp_pred, disp_gt, mask)
    assert torch.allclose(loss_depth, loss_disp, atol=1e-5), (
        f"SigLog(depth)={loss_depth.item()} vs SigLog(disparity)={loss_disp.item()} "
        "-- these must be identical; if you changed the disparity "
        "auxiliary loss to a log-space form, it is redundant, not a new signal."
    )


def test_ordinal_bin_targets_edge_cases():
    rd, max_depth = 5, 10.0
    zero_depth = torch.zeros(1, 1, 2, 2)
    max_depth_t = torch.full((1, 1, 2, 2), max_depth)

    t0 = ordinal_bin_targets(zero_depth, rd, max_depth)
    assert (t0 == 0).all(), "depth=0 must exceed no thresholds"

    t1 = ordinal_bin_targets(max_depth_t, rd, max_depth)
    assert (t1 == 1).all(), "depth==max_depth must exceed every bin threshold (all < max_depth)"


def test_hdi_loss_forward_smoke():
    cfg = LossConfig(regression="silog", disparity_aux_weight=0.0)
    loss_module = HDILoss(cfg, rd=5, max_depth=10.0)

    depth_final = torch.rand(2, 1, 32, 32) * 8 + 0.1
    seg_final = torch.rand(2, 5, 32, 32)
    depth_levels = [torch.rand(2, 1, 8, 8) * 8 + 0.1, torch.rand(2, 1, 16, 16) * 8 + 0.1]
    seg_levels = [torch.rand(2, 5, 8, 8), torch.rand(2, 5, 16, 16), torch.rand(2, 5, 32, 32)]
    gt_depth = torch.rand(2, 1, 32, 32) * 8 + 0.1

    losses = loss_module(depth_final, seg_final, depth_levels, seg_levels, gt_depth)
    assert torch.isfinite(losses["total"])
    for key in ("regression_final", "regression_deep_0", "regression_deep_1", "bin_bce", "total"):
        assert key in losses, f"missing loss component: {key}"


def test_hdi_loss_requires_intrinsics_when_disparity_enabled():
    cfg = LossConfig(regression="silog", disparity_aux_weight=0.2)
    try:
        HDILoss(cfg, rd=5, max_depth=10.0, camera=None)
        assert False, "expected an assertion error when disparity_aux_weight>0 with no camera intrinsics"
    except AssertionError:
        pass


def test_siglog_penalizes_absolute_scale():
    """The user-facing question: is SigLog scale-INVARIANT (a mixed-dataset
    loss that would discard this single-sensor dataset's metric scale)?

    No -- with lambda<1 it is scale-VARIANT. Scaling the prediction by a
    constant (a pure metric-scale error) MUST change the loss at the
    configured lambda=0.5, and provably cannot change it only at lambda=1.0
    (where the loss degenerates to the variance of the log residual). This
    pins the distinction against SSI (MiDaS/DPT's scale-and-shift-invariant
    loss), which is the actual multi-source-dataset loss.
    """
    import torch
    from instancedepth.losses.hdi_losses import SigLogLoss

    torch.manual_seed(0)
    gt = torch.rand(1, 1, 16, 16) * 4 + 1
    pred = gt * 1.25                      # pure 25% scale error, structure identical
    mask = torch.ones_like(gt, dtype=torch.bool)

    default = SigLogLoss(lam=0.5)(pred, gt, mask)
    assert float(default) > 0.05, "lambda=0.5 must penalize a pure scale error"

    fully_invariant = SigLogLoss(lam=1.0)(pred, gt, mask)
    assert float(fully_invariant) < 1e-5, "lambda=1.0 is the scale-invariant degenerate case"

    # and it is not merely insensitive: a bigger scale error costs more
    assert float(SigLogLoss(lam=0.5)(gt * 1.5, gt, mask)) > float(default)


def test_gradient_matching_prefers_sharp_edges():
    """SigLog is pointwise, so a blurred edge can score like a sharp one; the
    gradient-matching term must prefer the sharp reconstruction."""
    import torch
    import torch.nn.functional as F
    from instancedepth.losses.hdi_losses import GradientMatchingLoss

    gt = torch.ones(1, 1, 32, 32) * 2.0
    gt[..., 16:] = 5.0                                  # a hard depth step
    sharp = gt.clone()
    blurred = F.avg_pool2d(gt, 5, stride=1, padding=2)  # same values, smeared edge
    mask = torch.ones_like(gt, dtype=torch.bool)

    loss = GradientMatchingLoss(scales=3)
    assert float(loss(sharp, gt, mask)) < 1e-6          # perfect -> zero
    assert float(loss(blurred, gt, mask)) > 1e-3        # blur is penalized


def test_gradient_matching_ignores_invalid_gt():
    """Sensor holes must never manufacture a fake edge: gradients are counted
    only where both neighbouring pixels have valid GT."""
    import torch
    from instancedepth.losses.hdi_losses import GradientMatchingLoss

    gt = torch.ones(1, 1, 16, 16) * 3.0
    gt[..., 8:] = 0.0                     # right half invalid (sensor hole)
    pred = torch.ones_like(gt) * 3.0
    mask = gt > 0
    assert float(GradientMatchingLoss(scales=2)(pred, gt, mask)) < 1e-6
