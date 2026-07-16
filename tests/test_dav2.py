"""Tests for the full-DAv2 Phase-1 variant (videodepth/models/dav2_dpt.py).

No pretrained weights needed. The critical guarantees:
  * the head's state-dict KEYS AND SHAPES match the official DAv2 vitl
    checkpoint naming — a mismatch would make the whole module pointless;
  * the loader is fail-loud (partial/wrong checkpoints raise, never load
    silently);
  * forward shapes are exact (output at input resolution, in (0, max_depth));
  * the model composes with the temporal stage (VideoDepthModel contract).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from videodepth.models.dav2_dpt import (
    DAV2MetricModel, DPTNeck, DPTOutput, load_dav2_head_weights,
    VITL_HOOK_LAYERS, VITL_OUT_CHANNELS,
)


# --------------------------------------------------------------------------- #
# Checkpoint key/shape compatibility (vitl)
# --------------------------------------------------------------------------- #
def test_head_keys_and_shapes_match_official_dav2_vitl():
    neck = DPTNeck(embed_dim=1024, out_channels=VITL_OUT_CHANNELS, features=256)
    out = DPTOutput(features=256, max_depth=10.0)
    keys = {k: tuple(v.shape) for k, v in neck.state_dict().items()}
    keys.update({k: tuple(v.shape) for k, v in out.state_dict().items()})

    # Spot-check the exact key names + shapes of the official checkpoint
    # (depth_anything_v2_vitl.pth, 'depth_head.' prefix stripped).
    expected = {
        "projects.0.weight": (256, 1024, 1, 1),
        "projects.3.bias": (1024,),
        "resize_layers.0.weight": (256, 256, 4, 4),      # ConvTranspose2d 4x
        "resize_layers.1.weight": (512, 512, 2, 2),      # ConvTranspose2d 2x
        "resize_layers.3.weight": (1024, 1024, 3, 3),    # stride-2 Conv2d
        "scratch.layer1_rn.weight": (256, 256, 3, 3),
        "scratch.layer4_rn.weight": (256, 1024, 3, 3),
        "scratch.refinenet4.resConfUnit1.conv1.weight": (256, 256, 3, 3),
        "scratch.refinenet1.resConfUnit2.conv2.bias": (256,),
        "scratch.refinenet1.out_conv.weight": (256, 256, 1, 1),
        "scratch.output_conv1.weight": (128, 256, 3, 3),
        "scratch.output_conv2.0.weight": (32, 128, 3, 3),
        "scratch.output_conv2.2.weight": (1, 32, 1, 1),
    }
    for k, shape in expected.items():
        assert k in keys, f"missing official checkpoint key: {k}"
        assert keys[k] == shape, f"{k}: {keys[k]} != official {shape}"
    # rn convs are bias-free in the official head
    assert "scratch.layer1_rn.bias" not in keys


def test_loader_roundtrip_and_fail_loud(tmp_path):
    torch.manual_seed(0)
    ref_neck = DPTNeck(embed_dim=64, out_channels=(8, 16, 32, 32), features=16)
    ref_out = DPTOutput(features=16)
    sd = {f"depth_head.{k}": v for k, v in ref_neck.state_dict().items()}
    sd.update({f"depth_head.{k}": v for k, v in ref_out.state_dict().items()})
    sd["pretrained.blocks.0.attn.qkv.weight"] = torch.zeros(3)   # encoder keys ignored
    ckpt = tmp_path / "dav2.pth"
    torch.save(sd, ckpt)

    model = DAV2MetricModel(backbone=nn.Identity(), embed_dim=64,
                            out_channels=(8, 16, 32, 32), features=16)
    load_dav2_head_weights(model, ckpt)
    for k, v in ref_neck.state_dict().items():
        assert torch.equal(model.decoder.state_dict()[k], v)
    for k, v in ref_out.state_dict().items():
        assert torch.equal(model.refinement.state_dict()[k], v)

    # no depth_head.* keys at all -> loud failure
    torch.save({"pretrained.cls_token": torch.zeros(1)}, tmp_path / "enc_only.pth")
    with pytest.raises(RuntimeError, match="no 'depth_head"):
        load_dav2_head_weights(model, tmp_path / "enc_only.pth")

    # partial head -> loud failure (silent partial load would discard the prior)
    partial = {k: v for i, (k, v) in enumerate(sd.items()) if i % 2 == 0
               and k.startswith("depth_head.")}
    torch.save(partial, tmp_path / "partial.pth")
    with pytest.raises(RuntimeError, match="mismatch"):
        load_dav2_head_weights(model, tmp_path / "partial.pth")


# --------------------------------------------------------------------------- #
# Forward shapes + metric range
# --------------------------------------------------------------------------- #
class _StubEncoder(nn.Module):
    """Emits 4 taps at the 1/14 patch grid, like DINOv2Backbone."""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, 14, stride=14)

    def forward(self, x):
        t = self.proj(x)
        return [t, t, t, t]


def _tiny_model(max_depth=10.0):
    return DAV2MetricModel(backbone=_StubEncoder(32), embed_dim=32,
                           out_channels=(8, 16, 32, 32), features=16,
                           max_depth=max_depth)


def test_dav2_forward_shape_and_range():
    torch.manual_seed(0)
    model = _tiny_model(max_depth=10.0)
    x = torch.randn(2, 3, 56, 70)                    # /14 -> 4x5 patch grid
    with torch.no_grad():
        d = model(x)
    assert d.shape == (2, 1, 56, 70)                 # exactly input resolution
    assert float(d.min()) > 0.0 and float(d.max()) < 10.0   # sigmoid * max_depth


def test_dav2_neck_levels_are_coarse_to_fine():
    torch.manual_seed(0)
    model = _tiny_model()
    dec = model.decoder(model.backbone(torch.randn(1, 3, 56, 56)))
    hs = [f.shape[-1] for f in dec.levels]
    assert hs == sorted(hs) and len(dec.levels) == 3   # coarse -> fine
    assert dec.finest is dec.levels[-1]
    assert dec.patch_hw == (4, 4)


# --------------------------------------------------------------------------- #
# Composes with the temporal stage (VideoDepthModel contract)
# --------------------------------------------------------------------------- #
def test_dav2_composes_with_temporal_stabilizer():
    from videodepth.models.temporal_head import TemporalStabilizerBank
    from videodepth.models.video_model import VideoDepthModel
    torch.manual_seed(0)
    spatial = _tiny_model()
    bank = TemporalStabilizerBank(levels=(2,), feat_channels=16, d_model=8,
                                  num_blocks=1, downsample=0.5)
    model = VideoDepthModel(spatial, bank)
    clip = torch.randn(1, 2, 3, 56, 56)
    out = model.forward_clip(clip)
    assert out.shape == (1, 2, 1, 56, 56)
    # zero-init stabilizer -> identical to the per-frame DAv2 model
    model.reset_temporal_state()
    assert torch.equal(out[:, 0], model(clip[:, 0]))


def test_dav2_config_yaml_parses_and_hooks_are_dav2():
    from videodepth.configs.config import DAV2Config
    cfg = DAV2Config.from_yaml("videodepth/configs/dav2_full.yaml")
    assert tuple(cfg.backbone.hook_layers) == VITL_HOOK_LAYERS
    assert tuple(cfg.out_channels) == VITL_OUT_CHANNELS
    assert cfg.max_depth == 10.0
