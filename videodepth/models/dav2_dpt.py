"""Full Depth-Anything-V2 initialization for Phase 1 — the AbsRel lever.

Evidence (the paper's own Table 2):

  * this repo's from-scratch Depth-Range decoder:            REL 0.078
  * swapping ONLY the encoder to DAv2's:                     REL 0.0778 (−0.5 %)
  * plain DAv2 (pretrained encoder **+ DPT head**) on GID:   REL 0.053
  * the paper's full method on GID:                          REL 0.045

The pretrained **decoder** — not the encoder — carries the metric-depth
prior; training a bespoke decoder from scratch on a small dataset discards
it. This module reproduces DAv2's DPT head with **state-dict-key-for-key
compatibility** with the official checkpoints (``depth_anything_v2_vitl.pth``
and the metric hypersim/vkitti fine-tunes), so the full pretrained model
loads 1:1 and is then fine-tuned on the target data.

Layout mirrors ``HolisticDepthModel``'s ``backbone / decoder / refinement``
contract, so the temporal stage (``videodepth/models/video_model.py``) wraps
this model unchanged — DAv2 spatial + TGM-trained stabilizer composes for
free.

Key mapping of the official checkpoint:
  pretrained.*   -> the DINOv2 encoder (already handled by
                    ``instancedepth.models.backbone.dinov2_wrapper``, which
                    detects the DAv2 naming; point
                    ``backbone.checkpoint_path`` at the same file)
  depth_head.*   -> this module (``load_dav2_head_weights``): ``projects.*``,
                    ``resize_layers.*``, ``scratch.layer{1-4}_rn``,
                    ``scratch.refinenet{1-4}.*`` -> DPTNeck;
                    ``scratch.output_conv{1,2}.*`` -> DPTOutput.

Head output: ``sigmoid(x) * max_depth`` (DAv2's *metric* variant). The final
activation has no parameters, so the relative checkpoint also loads — but it
was trained to emit disparity, so the metric hypersim checkpoint is the
recommended init for this indoor-range metric data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("videodepth.models.dav2_dpt")

# Per-tap projection widths for ViT-L (DAv2's dpt.py `out_channels`).
VITL_OUT_CHANNELS: Tuple[int, ...] = (256, 512, 1024, 1024)
# DAv2's encoder tap layers for vitl ('intermediate_layer_idx' in its dpt.py).
VITL_HOOK_LAYERS: Tuple[int, ...] = (4, 11, 17, 23)


class ResidualConvUnit(nn.Module):
    """DAv2 util/blocks.py ResidualConvUnit (use_bn=False variant)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    """DAv2 util/blocks.py FeatureFusionBlock (align_corners=True, as in the
    official code — pretrained weights were trained under this convention)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1)
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None,
                size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        out = x
        if skip is not None:
            out = out + self.resConfUnit1(skip)
        out = self.resConfUnit2(out)
        if size is not None:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(out)


class _Scratch(nn.Module):
    """Container named 'scratch' so checkpoint keys line up exactly."""

    def __init__(self, out_channels: Tuple[int, ...], features: int) -> None:
        super().__init__()
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features)


class DPTNeck(nn.Module):
    """projects + resize_layers + scratch(rn convs, refinenets) — everything
    of DAv2's DPTHead except the two output convs (which live in DPTOutput so
    the temporal stabilizer can slot between fused features and the output).

    forward: 4 encoder taps (B,C,h,w) at the 1/14 patch grid ->
    SimpleNamespace(levels=[path_3, path_2, path_1], finest=path_1)
    (coarse-to-fine, matching the F_0/F_1/F_2 convention of the existing
    decoder so ``temporal.levels: [2]`` means "finest" for both models).
    """

    def __init__(self, embed_dim: int = 1024,
                 out_channels: Tuple[int, ...] = VITL_OUT_CHANNELS,
                 features: int = 256) -> None:
        super().__init__()
        self.projects = nn.ModuleList(
            [nn.Conv2d(embed_dim, oc, 1) for oc in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ])
        self.scratch = _Scratch(out_channels, features)

    def forward(self, taps: List[torch.Tensor]) -> SimpleNamespace:
        assert len(taps) == 4, f"DPTNeck needs 4 encoder taps, got {len(taps)}"
        l1, l2, l3, l4 = (self.resize_layers[i](self.projects[i](t))
                          for i, t in enumerate(taps))
        r1 = self.scratch.layer1_rn(l1)
        r2 = self.scratch.layer2_rn(l2)
        r3 = self.scratch.layer3_rn(l3)
        r4 = self.scratch.layer4_rn(l4)

        path_4 = self.scratch.refinenet4(r4, size=r3.shape[-2:])
        path_3 = self.scratch.refinenet3(path_4, r3, size=r2.shape[-2:])
        path_2 = self.scratch.refinenet2(path_3, r2, size=r1.shape[-2:])
        path_1 = self.scratch.refinenet1(path_2, r1)
        # patch grid (H/14, W/14) — DPTOutput upsamples to exactly 14x this,
        # matching DAv2's own `interpolate(..., (patch_h*14, patch_w*14))`.
        return SimpleNamespace(levels=[path_3, path_2, path_1], finest=path_1,
                               patch_hw=tuple(taps[0].shape[-2:]))


class DPTOutput(nn.Module):
    """DAv2's scratch.output_conv1/output_conv2 + metric activation."""

    def __init__(self, features: int = 256, max_depth: float = 10.0) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.scratch = nn.Module()   # keeps 'scratch.output_conv*' key naming
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, dec: SimpleNamespace) -> SimpleNamespace:
        ph, pw = dec.patch_hw
        x = self.scratch.output_conv1(dec.levels[-1])
        # DAv2 dpt.py: upsample the fused features to the full input
        # resolution (patch grid x 14) BEFORE the final convs.
        x = F.interpolate(x, size=(ph * 14, pw * 14), mode="bilinear", align_corners=True)
        x = self.scratch.output_conv2(x)
        depth = torch.sigmoid(x) * self.max_depth          # metric-variant head
        return SimpleNamespace(final_depth=depth)


class DAV2MetricModel(nn.Module):
    """Full-DAv2 Phase-1: pretrained encoder + pretrained DPT head + metric
    output, exposed through the backbone/decoder/refinement contract so
    evaluate_video and VideoDepthModel (temporal stage) work unchanged."""

    def __init__(self, backbone: nn.Module, embed_dim: int = 1024,
                 out_channels: Tuple[int, ...] = VITL_OUT_CHANNELS,
                 features: int = 256, max_depth: float = 10.0) -> None:
        super().__init__()
        self.backbone = backbone                 # must emit 4 taps (B,C,h,w)
        self.decoder = DPTNeck(embed_dim, out_channels, features)
        self.refinement = DPTOutput(features, max_depth)

    @classmethod
    def from_config(cls, cfg) -> "DAV2MetricModel":
        from instancedepth.models.backbone.dinov2_wrapper import DINOv2Backbone
        backbone = DINOv2Backbone(cfg.backbone)
        assert len(cfg.backbone.hook_layers) == 4, (
            f"DAv2's DPT head needs 4 encoder taps (vitl: {VITL_HOOK_LAYERS}); "
            f"got hook_layers={cfg.backbone.hook_layers}")
        model = cls(backbone, embed_dim=backbone.embed_dim,
                    out_channels=tuple(cfg.out_channels),
                    features=cfg.features, max_depth=cfg.max_depth)
        if cfg.dav2_checkpoint:
            load_dav2_head_weights(model, cfg.dav2_checkpoint)
        else:
            log.warning("dav2_checkpoint unset — DPT head starts RANDOM "
                        "(smoke test only; the pretrained head IS the point)")
        return model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        H, W = image.shape[-2:]
        dec = self.decoder(self.backbone(image))
        depth = self.refinement(dec).final_depth
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        return depth


# --------------------------------------------------------------------------- #
def load_dav2_head_weights(model: DAV2MetricModel, path: str | Path) -> None:
    """Load ``depth_head.*`` from an official DAv2 checkpoint into
    DPTNeck + DPTOutput. Fail-loud: every checkpoint head key must be
    consumed and every model head key must be filled — a silent partial load
    would quietly discard the prior this model exists to import.

    (Encoder keys ``pretrained.*`` are NOT handled here — the DINOv2 wrapper
    loads those itself from ``backbone.checkpoint_path``.)
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    head = {k[len("depth_head."):]: v for k, v in sd.items()
            if k.startswith("depth_head.")}
    if not head:
        raise RuntimeError(
            f"{path} contains no 'depth_head.*' keys — not a full DAv2 "
            f"checkpoint? (found key prefixes: "
            f"{sorted({k.split('.')[0] for k in sd})[:6]})")

    neck_sd = {k: v for k, v in head.items()
               if not k.startswith("scratch.output_conv")}
    out_sd = {k: v for k, v in head.items()
              if k.startswith("scratch.output_conv")}

    m1, u1 = model.decoder.load_state_dict(neck_sd, strict=False)
    m2, u2 = model.refinement.load_state_dict(out_sd, strict=False)
    problems = [f"neck missing {m1[:4]}" if m1 else "",
                f"neck unexpected {u1[:4]}" if u1 else "",
                f"output missing {m2[:4]}" if m2 else "",
                f"output unexpected {u2[:4]}" if u2 else ""]
    problems = [p for p in problems if p]
    if problems:
        raise RuntimeError(f"DAv2 head load mismatch ({path}): {'; '.join(problems)}")
    log.info("loaded pretrained DAv2 DPT head from %s (%d neck + %d output tensors)",
             path, len(neck_sd), len(out_sd))
