"""Unit tests for the Phase-2 checkpoint key-remapper
(videodepth/tools/convert_phase2_checkpoint.py): repairs a checkpoint saved
under a transformers version with different Swin-attention key names."""

from __future__ import annotations

import torch

from videodepth.tools.convert_phase2_checkpoint import apply_remap, build_remap


def _old_new():
    model_sd = {                                    # current transformers names
        "enc.b0.attention.q_proj.weight": torch.zeros(4, 4),
        "enc.b0.attention.k_proj.weight": torch.zeros(4, 4),
        "enc.b0.attention.v_proj.weight": torch.zeros(4, 4),
        "enc.b0.norm.weight": torch.zeros(4),       # identical in both
        "head.weight": torch.zeros(2, 4),
    }
    ckpt_sd = {                                     # older-transformers names
        "enc.b0.attention.query.weight": torch.ones(4, 4),
        "enc.b0.attention.key.weight": torch.ones(4, 4),
        "enc.b0.attention.value.weight": torch.ones(4, 4),
        "enc.b0.norm.weight": torch.ones(4),
        "head.weight": torch.ones(2, 4),
        "enc.b0.attention.relative_position_index": torch.zeros(9, 9),  # buffer
    }
    return model_sd, ckpt_sd


def test_remap_renames_qkv_by_shape_and_order():
    model_sd, ckpt_sd = _old_new()
    remap, unresolved_model, unresolved_ckpt = build_remap(ckpt_sd, model_sd)
    assert remap == {
        "enc.b0.attention.query.weight": "enc.b0.attention.q_proj.weight",
        "enc.b0.attention.key.weight": "enc.b0.attention.k_proj.weight",
        "enc.b0.attention.value.weight": "enc.b0.attention.v_proj.weight",
    }
    assert unresolved_model == []                                     # every weight covered
    assert unresolved_ckpt == ["enc.b0.attention.relative_position_index"]  # buffer dropped


def test_apply_remap_fills_every_model_key_with_right_tensor():
    model_sd, ckpt_sd = _old_new()
    remap, _, _ = build_remap(ckpt_sd, model_sd)
    new = apply_remap(ckpt_sd, remap, model_sd)
    assert set(new) == set(model_sd)                                 # loads strict=True
    assert torch.equal(new["enc.b0.attention.q_proj.weight"], torch.ones(4, 4))
    assert torch.equal(new["enc.b0.attention.v_proj.weight"], torch.ones(4, 4))


def test_genuinely_missing_weight_is_reported_not_fabricated():
    model_sd, ckpt_sd = _old_new()
    model_sd["enc.b1.extra.weight"] = torch.zeros(7, 7)              # no source anywhere
    _, unresolved_model, _ = build_remap(ckpt_sd, model_sd)
    assert "enc.b1.extra.weight" in unresolved_model                 # flagged, not silently matched


def test_ambiguation_respects_parent_boundaries():
    # two separate attention blocks -> q/k/v must not cross block boundaries
    model_sd = {f"enc.b{b}.attention.q_proj.weight": torch.zeros(4, 4) for b in (0, 1)}
    ckpt_sd = {f"enc.b{b}.attention.query.weight": torch.full((4, 4), float(b)) for b in (0, 1)}
    remap, um, _ = build_remap(ckpt_sd, model_sd)
    assert um == []
    new = apply_remap(ckpt_sd, remap, model_sd)
    assert torch.equal(new["enc.b0.attention.q_proj.weight"], torch.zeros(4, 4))   # block 0 -> 0
    assert torch.equal(new["enc.b1.attention.q_proj.weight"], torch.ones(4, 4))    # block 1 -> 1
