"""Shape (and one correctness) tests for the Phase 1 model, runnable
without any pretrained weights except `test_full_model_real_backbone`,
which is skipped automatically if the configured DINOv2 checkpoint isn't
present on disk.

Run:  pytest tests/test_hdi_shapes.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

from instancedepth.configs.config import BinRefinementConfig, DecoderConfig, HDIConfig
from instancedepth.models.hdi.depth_range_decoder import DepthRangeFeatureDecoder, DepthRangeFeatures
from instancedepth.models.hdi.iterative_refinement import IterativeBinRefinement

BACKBONE_EMBED_DIM = 1024   # ViT-L
BACKBONE_HW = (52, 92)      # 728/14, 1288/14
IMAGE_HW = (728, 1288)


def _dummy_backbone_features(batch: int = 2) -> list[torch.Tensor]:
    return [torch.randn(batch, BACKBONE_EMBED_DIM, *BACKBONE_HW) for _ in range(3)]


def test_decoder_output_shapes():
    cfg = DecoderConfig()
    decoder = DepthRangeFeatureDecoder(in_channels=BACKBONE_EMBED_DIM, image_hw=IMAGE_HW, cfg=cfg)
    feats = decoder(_dummy_backbone_features())

    expected_hw = [(91, 161), (182, 322), (364, 644)]   # 1/8, 1/4, 1/2 of (728,1288) --
    for i, (level_out, hw) in enumerate(zip(feats.levels, expected_hw)):
        assert level_out.shape == (2, cfg.channels_attn, *hw), (
            f"level {i}: expected {(2, cfg.channels_attn, *hw)}, got {tuple(level_out.shape)}"
        )
    assert feats.finest.shape == (2, cfg.channels_attn, 364, 644)


def test_refinement_shapes():
    decoder_cfg = DecoderConfig()
    bins_cfg = BinRefinementConfig(rd=5, max_depth=10.0)
    features = DepthRangeFeatures(levels=[
        torch.randn(2, decoder_cfg.channels_attn, 91, 161),
        torch.randn(2, decoder_cfg.channels_attn, 182, 322),
        torch.randn(2, decoder_cfg.channels_attn, 364, 644),
    ])
    refinement = IterativeBinRefinement(feat_channels=decoder_cfg.channels_attn, cfg=bins_cfg)
    trace = refinement(features)

    assert len(trace.depths) == 4   # D_0, D_1, D_2, D_3
    assert len(trace.bins) == 3     # S_0, S_1, S_2
    expected_hw = [(91, 161), (91, 161), (182, 322), (364, 644)]  # D_0 at level0's res, then D_1,D_2,D_3
    for i, (d, hw) in enumerate(zip(trace.depths, expected_hw)):
        assert d.shape == (2, 1, *hw), f"D_{i}: expected (2,1,{hw}), got {tuple(d.shape)}"
    for i, (s, hw) in enumerate(zip(trace.bins, expected_hw[1:])):
        assert s.shape == (2, bins_cfg.rd, *hw), f"S_{i}: expected (2,{bins_cfg.rd},{hw}), got {tuple(s.shape)}"


def test_depth_seed_is_nonnegative():
    """InitialDepthHead uses softplus -- D_0 must never be negative."""
    decoder_cfg = DecoderConfig()
    bins_cfg = BinRefinementConfig(rd=5, max_depth=10.0)
    features = DepthRangeFeatures(levels=[
        torch.randn(4, decoder_cfg.channels_attn, 91, 161) * 10,  # exaggerate to stress-test
        torch.randn(4, decoder_cfg.channels_attn, 182, 322),
        torch.randn(4, decoder_cfg.channels_attn, 364, 644),
    ])
    refinement = IterativeBinRefinement(feat_channels=decoder_cfg.channels_attn, cfg=bins_cfg)
    trace = refinement(features)
    assert (trace.depths[0] >= 0).all(), "D_0 (seed) must be non-negative (softplus)"


def test_correction_is_bidirectional():
    """Regression test for the ordinal-vs-softmax correction: with the ordinal
    (independent-sigmoid) S_i reading, E_i must be able to be positive OR
    negative across a large random batch -- a softmax reading would force
    E_i <= 0 always. This test would FAIL
    if S_i's activation were changed back to softmax."""
    decoder_cfg = DecoderConfig()
    bins_cfg = BinRefinementConfig(rd=5, max_depth=10.0)
    torch.manual_seed(0)
    # batch=4 is plenty: even one sample has ~14k independent spatial
    # locations, each an independent sample of the correction's sign. (A
    # much larger batch here was OOM-killed on a real machine -- the finest
    # feature map alone is B x 256 x 364 x 644 floats.) no_grad() halves
    # memory since this test only inspects output signs, not gradients.
    features = DepthRangeFeatures(levels=[
        torch.randn(4, decoder_cfg.channels_attn, 91, 161),
        torch.randn(4, decoder_cfg.channels_attn, 182, 322),
        torch.randn(4, decoder_cfg.channels_attn, 364, 644),
    ])
    refinement = IterativeBinRefinement(feat_channels=decoder_cfg.channels_attn, cfg=bins_cfg)
    with torch.no_grad():
        trace = refinement(features)

    d0, d1 = trace.depths[0], trace.depths[1]
    e0 = d1 - torch.nn.functional.interpolate(d0, size=d1.shape[-2:], mode="bilinear", align_corners=False)
    assert (e0 > 0).any(), "expected at least some positive corrections across a large random batch"
    assert (e0 < 0).any(), "expected at least some negative corrections across a large random batch"


@pytest.mark.skipif(
    not os.path.isfile(HDIConfig().backbone.checkpoint_path or ""),
    reason="real DINOv2 checkpoint not found at the configured backbone.checkpoint_path",
)
def test_full_model_real_backbone():
    from instancedepth.models.hdi.model import HolisticDepthModel

    cfg = HDIConfig()
    model = HolisticDepthModel(cfg)
    x = torch.randn(1, 3, *cfg.data.image_size)
    out = model(x)
    H, W = cfg.data.image_size
    assert out.depth_final.shape == (1, 1, H, W)
    assert out.seg_final.shape == (1, cfg.bins.rd, H, W)
    assert out.feat_final.shape[-2:] == (364, 644)
    assert out.image_hw == (H, W)
