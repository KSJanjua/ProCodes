"""Unit tests for the FlashDepth-style temporal integration (synthetic
tensors -- no weights, no dataset files needed)."""

from __future__ import annotations

import torch

from instancedepth.models.hdi.temporal import ConvGRUCell, TemporalAligner
from instancedepth.utils.metrics import temporal_alignment_error


def test_aligner_identity_at_init():
    """Zero-init output projection: the aligner must be an EXACT no-op at
    initialization (FlashDepth's stabilizer), for the first frame and for
    every later frame of a sequence."""
    torch.manual_seed(0)
    aligner = TemporalAligner(feat_channels=32, d_model=16, num_blocks=2, downsample=0.25)
    f1, f2 = torch.randn(2, 32, 20, 36), torch.randn(2, 32, 20, 36)
    assert torch.allclose(aligner(f1), f1, atol=1e-6)
    assert torch.allclose(aligner(f2), f2, atol=1e-6)   # state carried, output still identity


def test_aligner_state_carry_and_reset():
    """After training-like perturbation of the output projection, the second
    frame's output must depend on the first (state carried), and reset_state()
    must restore the length-1-sequence behavior exactly."""
    torch.manual_seed(0)
    aligner = TemporalAligner(feat_channels=8, d_model=8, num_blocks=1, downsample=0.5)
    torch.nn.init.normal_(aligner.proj_out.weight, std=0.1)   # simulate a trained module
    fA, fB = torch.randn(1, 8, 12, 12), torch.randn(1, 8, 12, 12)

    aligner.reset_state()
    out_b_after_a = aligner(fA), aligner(fB)
    aligner.reset_state()
    out_b_fresh = aligner(fB)
    assert not torch.allclose(out_b_after_a[1], out_b_fresh, atol=1e-6), \
        "frame B's output must differ depending on whether A preceded it"

    aligner.reset_state()
    assert torch.allclose(aligner(fB), out_b_fresh, atol=1e-6), \
        "reset must exactly restore the fresh-memory behavior"


def test_aligner_auto_reset_on_shape_change():
    """Batch/resolution changes must not crash: state silently refreshes."""
    aligner = TemporalAligner(feat_channels=4, d_model=4, num_blocks=1, downsample=0.5)
    aligner(torch.randn(2, 4, 16, 16))
    out = aligner(torch.randn(1, 4, 8, 8))   # different batch AND resolution
    assert out.shape == (1, 4, 8, 8)


def test_convgru_gate_bounds():
    """Hidden state stays bounded (tanh candidate, convex gate mix)."""
    cell = ConvGRUCell(8)
    h = None
    x = torch.randn(1, 8, 6, 6) * 10
    for _ in range(20):
        h = cell(x, h)
    assert h.abs().max() <= 1.0 + 1e-5


def test_temporal_alignment_error_metric():
    gt1 = torch.full((1, 1, 4, 4), 3.0)
    gt2 = torch.full((1, 1, 4, 4), 3.5)          # scene moved 0.5m
    pred1 = gt1.clone()
    pred2_track = gt2.clone()                    # perfectly tracks GT
    pred2_flicker = gt2 + 0.2                    # extra frame-to-frame jump
    mask = torch.ones_like(gt1, dtype=torch.bool)
    assert temporal_alignment_error(pred2_track, pred1, gt2, gt1, mask) == 0.0
    assert abs(temporal_alignment_error(pred2_flicker, pred1, gt2, gt1, mask) - 0.2) < 1e-6
    empty = torch.zeros_like(mask)
    assert temporal_alignment_error(pred2_track, pred1, gt2, gt1, empty) != \
        temporal_alignment_error(pred2_track, pred1, gt2, gt1, empty)   # NaN on no valid pixels


def test_temporal_config_wiring():
    from instancedepth.configs.config import HDIConfig
    cfg = HDIConfig.from_yaml("instancedepth/configs/hdi_temporal.yaml")
    assert cfg.temporal.enabled and cfg.temporal.levels == (2,)
    assert cfg.temporal.clip_len == 5 and cfg.temporal.clip_strides == (1, 2, 4, 8)
    # every baseline profile must stay per-frame
    for f in ("hdi.yaml", "hdi_enhanced.yaml", "hdi_dav2.yaml"):
        assert not HDIConfig.from_yaml(f"instancedepth/configs/{f}").temporal.enabled, f


def test_composite_soft_alpha_blends_at_boundary():
    """The composite blends the correction with the SOFT mask probability, so
    the depth transitions smoothly across the silhouette instead of stepping
    at a hard edge (the boundary-line artifact). A soft mask boundary must
    therefore yield partial correction, while a confident interior is fully
    corrected and pixels the mask never touches are untouched."""
    from instancedepth.models.phase3.candidates import PairSet
    from instancedepth.models.phase3.relation_head import composite_refined_depth

    H = W = 64
    base = torch.full((1, 1, H, W), 4.0)
    pairs = PairSet(
        batch_index=torch.tensor([0]), query_idx=torch.tensor([[0, 1]]),
        boxes_norm=torch.tensor([[[0.0, 0.0, 0.75, 0.75], [0.5, 0.5, 1.0, 1.0]]]),
        iou=torch.tensor([0.3]),
    )
    e = torch.full((1, 2, 1, 14, 14), 0.75)      # ratio 1.5 -> 6.0 where alpha == 1
    layers = torch.tensor([[2.0, 3.0]])
    mask_prob = torch.zeros(1, 2, H, W)
    mask_prob[0, 0, 8:40, 8:40] = 1.0            # confident interior
    mask_prob[0, 0, 8:40, 6:8] = 0.6            # a soft fringe (0<alpha<1) at the edge
    mask_prob[0, 1, 44:60, 44:60] = 1.0

    refined = composite_refined_depth(base, pairs, e, layers, mask_prob, 0.5, "scalar")
    assert torch.allclose(refined[0, 0, 20:28, 20:28], torch.full((8, 8), 6.0), atol=1e-3)  # interior: full
    fringe = float(refined[0, 0, 20, 6])         # soft fringe: base*(1+0.6*(1.5-1)) = 4*1.3 = 5.2
    assert abs(fringe - 5.2) < 1e-3, f"soft fringe should blend, got {fringe}"
    assert torch.allclose(refined[0, 0, 0:5, 0:5], torch.full((5, 5), 4.0), atol=1e-6)      # outside: untouched
