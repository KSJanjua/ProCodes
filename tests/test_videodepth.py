"""Unit tests for the videodepth package (synthetic tensors, no weights).

The invariants that matter:
  * TGM loss is zero for a prediction that tracks GT (even at a wrong scale)
    and positive for flicker — so the gradient points at flicker only.
  * The temporal stabilizer is an EXACT no-op at init (the wrapped model can
    never start worse than its per-frame baseline) and carries state.
  * The bounded relation head is identity at init and its correction is
    hard-capped — rings are bounded by construction.
  * Track memory holds an occluded instance's depth and low-passes flicker.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from videodepth.losses.temporal_losses import TemporalGradientMatchingLoss
from videodepth.models.occlusion import BoundedPairAttentionHead
from videodepth.models.temporal_head import TemporalStabilizer, TemporalStabilizerBank
from videodepth.models.track_memory import TrackDepthMemory
from videodepth.data.motion_clips import clip_weights, frame_motion_scores


# --------------------------------------------------------------------------- #
# Temporal Gradient Matching loss
# --------------------------------------------------------------------------- #
def _clip(T=4, H=8, W=8, base=3.0):
    t = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1)
    return base + 0.1 * t + torch.zeros(1, T, 1, H, W)   # depth receding over time


def test_tgm_zero_when_tracking_gt_even_at_wrong_scale():
    gt = _clip()
    tgm = TemporalGradientMatchingLoss()
    assert tgm(gt.clone(), gt).item() == pytest.approx(0.0, abs=1e-6)
    # log-space: a constant SCALE error has zero temporal gradient error
    assert tgm(2.0 * gt, gt).item() == pytest.approx(0.0, abs=1e-5)


def test_tgm_positive_for_flicker_and_masks_invalid():
    gt = _clip()
    pred = gt.clone()
    pred[:, 1] *= 1.2                                    # one flickering frame
    tgm = TemporalGradientMatchingLoss()
    assert tgm(pred, gt).item() > 0.01
    # invalidate GT wherever the flicker lives -> loss ignores it
    gt_masked = gt.clone()
    gt_masked[:, 1] = 0.0
    assert tgm(pred, gt_masked).item() == pytest.approx(0.0, abs=1e-6)


def test_tgm_single_frame_clip_is_zero_with_grad():
    gt = _clip(T=1)
    pred = gt.clone().requires_grad_(True)
    loss = TemporalGradientMatchingLoss()(pred, gt)
    assert loss.item() == 0.0
    loss.backward()                                      # grad path must exist
    assert pred.grad is not None


def test_tgm_order2_penalizes_acceleration_mismatch():
    gt = _clip(T=5)
    pred = gt.clone()
    pred[:, 2] += 0.05                                   # a velocity kink
    l1 = TemporalGradientMatchingLoss(order=1)(pred, gt).item()
    l2 = TemporalGradientMatchingLoss(order=2)(pred, gt).item()
    assert l2 > l1 > 0                                   # order 2 sees the kink twice


# --------------------------------------------------------------------------- #
# Temporal stabilizer
# --------------------------------------------------------------------------- #
def test_stabilizer_exact_noop_at_init():
    torch.manual_seed(0)
    stab = TemporalStabilizer(feat_channels=16, d_model=8, num_blocks=2, downsample=0.25)
    x = torch.randn(2, 16, 20, 24)
    assert torch.equal(stab(x), x)                       # zero-init out-proj -> identity
    assert torch.equal(stab(x), x)                       # still identity with state held


def test_stabilizer_state_carries_and_resets():
    torch.manual_seed(0)
    stab = TemporalStabilizer(feat_channels=8, d_model=8, num_blocks=1, downsample=0.5)
    with torch.no_grad():                                # make it a non-trivial map
        stab.proj_out.weight.normal_(0, 0.1)
    x = torch.randn(1, 8, 12, 12)
    y1 = stab(x)
    y2 = stab(x)                                         # same input, evolved state
    assert not torch.allclose(y1, y2)
    stab.reset_state()
    assert torch.allclose(stab(x), y1)                   # reset reproduces frame 1


def test_stabilizer_state_autoresets_on_shape_change():
    stab = TemporalStabilizer(feat_channels=8, d_model=8, num_blocks=1, downsample=0.5)
    stab(torch.randn(1, 8, 12, 12))
    out = stab(torch.randn(2, 8, 16, 16))                # batch+res change: no crash
    assert out.shape == (2, 8, 16, 16)


def test_stabilizer_bank_applies_only_configured_levels():
    bank = TemporalStabilizerBank(levels=(1,), feat_channels=8, d_model=8,
                                  num_blocks=1, downsample=0.5)
    with torch.no_grad():
        bank["1"].proj_out.weight.normal_(0, 0.1)
    levels = [torch.randn(1, 8, 10, 10) for _ in range(3)]
    orig = [l.clone() for l in levels]
    bank.apply_to(levels)   # frame 1: GRU state was zero-init, output may equal input
    bank.apply_to(levels)   # frame 2: state definitely non-trivial
    assert torch.equal(levels[0], orig[0]) and torch.equal(levels[2], orig[2])
    assert not torch.allclose(levels[1], orig[1])


# --------------------------------------------------------------------------- #
# VideoDepthModel plumbing (dummy spatial stand-in; no pretrained weights)
# --------------------------------------------------------------------------- #
class _DummySpatial(torch.nn.Module):
    """Mimics HolisticDepthModel's backbone/decoder/refinement contract."""

    def __init__(self, C=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, C, 3, padding=1)
        self.head = torch.nn.Conv2d(C, 1, 1)
        self.backbone = lambda img: img
        self.decoder = lambda img: SimpleNamespace(
            levels=[self.conv(img)], finest=None)
        self.refinement = lambda dec: SimpleNamespace(
            final_depth=torch.nn.functional.softplus(self.head(dec.levels[0])) + 0.5)


def test_video_model_clip_matches_per_frame_at_init():
    from videodepth.models.video_model import VideoDepthModel
    torch.manual_seed(0)
    spatial = _DummySpatial()
    bank = TemporalStabilizerBank(levels=(0,), feat_channels=8, d_model=8,
                                  num_blocks=1, downsample=0.5)
    model = VideoDepthModel(spatial, bank)
    clip = torch.randn(1, 3, 3, 16, 16)
    out = model.forward_clip(clip)
    assert out.shape == (1, 3, 1, 16, 16)
    # zero-init stabilizer -> bit-identical to running each frame alone
    for t in range(3):
        model.reset_temporal_state()
        assert torch.equal(out[:, t], model(clip[:, t]))


def test_video_model_freeze_spatial_leaves_temporal_trainable():
    from videodepth.models.video_model import VideoDepthModel
    model = VideoDepthModel(_DummySpatial(), TemporalStabilizerBank(
        levels=(0,), feat_channels=8, d_model=8, num_blocks=1, downsample=0.5))
    model.freeze_spatial()
    assert not any(p.requires_grad for p in model.spatial.parameters())
    assert all(p.requires_grad for p in model.temporal.parameters())


# --------------------------------------------------------------------------- #
# Bounded pair-attention head
# --------------------------------------------------------------------------- #
def _pair_inputs(P=3, C=6, G=4, Hp=7, Wp=7):
    torch.manual_seed(1)
    return (torch.randn(P, 2, C, Hp, Wp), torch.randn(P, 2, G, Hp, Wp),
            torch.rand(P, 2, 1, Hp, Wp) * 5 + 1)


def test_bounded_head_identity_at_init():
    f, g, d = _pair_inputs()
    head = BoundedPairAttentionHead(per_member_channels=10, hidden_dim=16, max_corr=0.15)
    e, d_hat = head(f, g, d)
    assert torch.allclose(e, torch.full_like(e, 0.5))
    assert torch.allclose(d_hat, d)                       # Eq. 9 identity


def test_bounded_head_correction_is_hard_capped():
    f, g, d = _pair_inputs()
    head = BoundedPairAttentionHead(per_member_channels=10, hidden_dim=16, max_corr=0.15)
    with torch.no_grad():                                 # force it far from identity
        head.out.weight.normal_(0, 10.0)
        head.out.bias.fill_(50.0)
    with torch.no_grad():
        e, d_hat = head(f, g, d)
    ratio = d_hat / d
    assert float((ratio - 1).abs().max()) <= 0.15 + 1e-5  # never beyond ±max_corr
    assert float((2 * e - ratio).abs().max()) < 1e-5      # e stays Eq.9-consistent


def test_bounded_head_scalar_granularity_and_empty():
    f, g, d = _pair_inputs()
    head = BoundedPairAttentionHead(per_member_channels=10, hidden_dim=16,
                                    granularity="scalar")
    with torch.no_grad():
        head.out.weight.normal_(0, 1.0)
    e, _ = head(f, g, d)
    # scalar: one value per (pair, member), constant over the ROI
    assert torch.allclose(e.amax(dim=(-2, -1)), e.amin(dim=(-2, -1)))
    e0, d0 = head(f[:0], g[:0], d[:0])
    assert e0.shape[0] == 0 and d0.shape[0] == 0


# --------------------------------------------------------------------------- #
# Track memory (temporal amodal completion)
# --------------------------------------------------------------------------- #
def test_track_memory_holds_depth_through_occlusion():
    mem = TrackDepthMemory(momentum=0.5)
    for _ in range(5):                                    # person visible at 4 m
        mem.step()
        mem.update(7, 4.0, visibility=1.0, area=1000)
    # now occluded: the visible sliver reads the OCCLUDER's depth (2 m)
    mem.step()
    stab = mem.update(7, 2.0, visibility=0.0, area=100)
    assert stab == pytest.approx(4.0, abs=1e-6)           # memory wins when v=0
    assert mem.get(7) == pytest.approx(4.0, abs=1e-6)     # and isn't contaminated


def test_track_memory_visible_updates_move_the_estimate():
    mem = TrackDepthMemory(momentum=0.5)
    mem.step(); mem.update(1, 4.0, 1.0, area=500)
    mem.step(); out = mem.update(1, 5.0, 1.0, area=500)
    assert 4.0 < mem.get(1) <= 5.0
    assert out == pytest.approx(4.5)                      # visible -> SMOOTHED estimate
    assert out == mem.get(1)                              # output IS the EMA


def test_track_memory_visibility_from_area_and_eviction():
    mem = TrackDepthMemory(momentum=1.0, max_age=2)
    mem.step(); mem.update(3, 4.0, 1.0, area=1000)
    assert mem.visibility_from_area(3, 400.0) == pytest.approx(0.4)
    assert mem.visibility_from_area(99, 50.0) == 1.0      # unknown track: neutral
    for _ in range(3):
        mem.step()                                        # unseen 3 > max_age frames
    assert mem.get(3) is None and len(mem) == 0


def test_track_memory_velocity_coasts_while_hidden():
    mem = TrackDepthMemory(momentum=1.0, velocity_momentum=1.0)
    mem.step(); mem.update(5, 4.0, 1.0)
    mem.step(); mem.update(5, 4.2, 1.0)                   # walking away: +0.2/frame
    mem.step()                                            # hidden this frame
    assert mem.get(5) == pytest.approx(4.4, abs=1e-6)     # coasted along velocity


# --------------------------------------------------------------------------- #
# Motion-aware clip weighting
# --------------------------------------------------------------------------- #
def test_frame_motion_scores_static_vs_moving():
    static = [np.full((32, 32), 4.0, np.float32)] * 3
    assert frame_motion_scores(static) == [0.0, 0.0, 0.0]
    moving = [np.full((32, 32), 4.0 * (1.1 ** t), np.float32) for t in range(3)]
    s = frame_motion_scores(moving)
    assert s[0] == 0.0 and s[1] == pytest.approx(np.log(1.1), abs=1e-4) == s[2]
    # holes (0) in either frame contribute nothing
    holey = [m.copy() for m in moving]
    holey[1][:, :] = 0.0
    assert frame_motion_scores(holey)[1] == 0.0


def test_clip_weights_prefer_motion_with_floor():
    scores = {"seqA": [0.0, 0.0, 0.0, 0.0], "seqB": [0.0, 0.5, 0.5, 0.5]}
    index = [("seqA", 0, 1), ("seqB", 0, 1)]
    w = clip_weights(index, scores, clip_len=3, floor=0.1)
    assert w[1] > w[0]                                    # moving clip outweighs static
    assert w[0] == pytest.approx(0.1)                     # static hits the floor
    # unknown sequence -> neutral weight 1, never dropped
    w2 = clip_weights([("mystery", 0, 1)], scores, clip_len=3, floor=0.1)
    assert w2[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Streaming instance stabilizer (tracker + memory end to end)
# --------------------------------------------------------------------------- #
def test_streaming_stabilizer_smooths_layers_and_keeps_ids():
    from videodepth.models.phase3_video import StreamingInstanceStabilizer
    H = W = 40
    m = np.zeros((H, W), bool)
    m[5:35, 5:25] = True
    stab = StreamingInstanceStabilizer(min_hits=1, memory_momentum=0.3)
    layers = [4.0, 4.5, 3.6, 4.1, 4.4]                    # flickering estimate
    out_ids, out_layers = [], []
    for layer in layers:
        _, ls, ids = stab.update([m], [layer])
        out_ids.append(ids[0]); out_layers.append(ls[0])
    assert len(set(out_ids)) == 1                         # one persistent identity
    # stabilized trace varies less than the raw flicker
    assert np.std(out_layers[1:]) < np.std(layers[1:])


# --------------------------------------------------------------------------- #
# Regression: Trainer.fit builds a RunManifest that reads cfg.data.* — the
# first server run crashed with AttributeError because VideoConfig had no
# .data. It now resolves from hdi_config at load time.
# --------------------------------------------------------------------------- #
def test_video_config_exposes_data_for_run_manifest():
    from pathlib import Path
    from instancedepth.utils.manifest import RunManifest
    from videodepth.configs.config import VideoConfig
    cfg = VideoConfig.from_yaml("videodepth/configs/video_temporal.yaml")
    assert cfg.data is not None and cfg.data.annotations_root
    m = RunManifest.build(cfg, repo_root=Path("."))       # must not raise
    assert m.seed == cfg.seed


def test_frame_motion_scores_ignore_nan_and_inf():
    base = np.full((32, 32), 4.0, np.float32)
    corrupt = base.copy()
    corrupt[0, 0] = np.nan
    corrupt[1, 1] = np.inf
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # any RuntimeWarning fails
        s = frame_motion_scores([base, corrupt])
    assert s[1] == pytest.approx(0.0, abs=1e-6)           # finite pixels are static
