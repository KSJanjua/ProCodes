"""HolisticDepthOutput contract sanity checks (plan SS17)."""

from __future__ import annotations

import torch

from instancedepth.models.hdi.output import CONTRACT_VERSION, HolisticDepthOutput


def _dummy_output() -> HolisticDepthOutput:
    return HolisticDepthOutput(
        depth_final=torch.rand(1, 1, 728, 1288),
        seg_final=torch.rand(1, 5, 728, 1288),
        feat_final=torch.rand(1, 256, 364, 644),
        depth_levels=[torch.rand(1, 1, 91, 161), torch.rand(1, 1, 182, 322)],
        seg_levels=[torch.rand(1, 5, 91, 161), torch.rand(1, 5, 182, 322), torch.rand(1, 5, 364, 644)],
        image_hw=(728, 1288),
        feat_hw=(364, 644),
    )


def test_contract_version_present():
    out = _dummy_output()
    assert out.contract_version == CONTRACT_VERSION


def test_feat_stride_matches_shapes():
    out = _dummy_output()
    stride_h, stride_w = out.feat_stride()
    assert stride_h == 728 / 364 == 2.0
    assert stride_w == 1288 / 644 == 2.0

    # round-trip: mapping a feat_final-space coordinate through the stride
    # should land back in image_hw-space consistently.
    feat_y, feat_x = 10, 20
    image_y, image_x = feat_y * stride_h, feat_x * stride_w
    assert 0 <= image_y < out.image_hw[0]
    assert 0 <= image_x < out.image_hw[1]


def test_resized_output_after_inference_still_self_consistent():
    """Mirrors what HDIInferencer.predict() does when the caller's image
    isn't cfg.data.image_size (plan: models/hdi/inference.py)."""
    out = _dummy_output()
    orig_hw = (720, 1280)
    resized = HolisticDepthOutput(
        depth_final=torch.nn.functional.interpolate(out.depth_final, size=orig_hw, mode="bilinear", align_corners=False),
        seg_final=torch.nn.functional.interpolate(out.seg_final, size=orig_hw, mode="bilinear", align_corners=False),
        feat_final=out.feat_final,   # left at native decoder resolution
        depth_levels=out.depth_levels,
        seg_levels=out.seg_levels,
        image_hw=orig_hw,
        feat_hw=out.feat_hw,
    )
    assert resized.depth_final.shape[-2:] == orig_hw
    stride_h, stride_w = resized.feat_stride()
    assert abs(stride_h - orig_hw[0] / 364) < 1e-6
    assert abs(stride_w - orig_hw[1] / 644) < 1e-6
