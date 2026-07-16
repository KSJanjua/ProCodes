"""VideoDepthModel — a trained Phase-1 spatial model + streaming temporal
stabilizer, with clip and streaming forward paths.

Composition, not modification: the spatial ``HolisticDepthModel`` is built
with its own temporal module DISABLED and loaded from its trained checkpoint;
this wrapper re-runs the same backbone -> decoder -> refinement pipeline with
the ``TemporalStabilizerBank`` inserted between decoder and refinement. The
per-frame model therefore stays bit-identical when the stabilizer is a no-op
(zero-init), and the spatial weights can be frozen (stage-2a) for clean
attribution of any temporal gain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from videodepth.configs.config import VideoConfig
from videodepth.models.temporal_head import TemporalStabilizerBank

log = logging.getLogger("videodepth.models.video_model")


class VideoDepthModel(nn.Module):
    def __init__(self, spatial: nn.Module, temporal: TemporalStabilizerBank) -> None:
        """``spatial`` must expose ``backbone``, ``decoder``, ``refinement``
        (the HolisticDepthModel contract). Injected rather than constructed so
        tests can pass a lightweight stand-in."""
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal

    # ------------------------------------------------------------- factory
    @classmethod
    def from_config(cls, cfg: VideoConfig) -> "VideoDepthModel":
        from instancedepth.configs.config import HDIConfig
        from instancedepth.models.hdi.model import HolisticDepthModel
        from instancedepth.utils.checkpoint import load_checkpoint

        hdi_cfg = HDIConfig.from_yaml(cfg.hdi_config)
        hdi_cfg.temporal.enabled = False   # the old inline module must stay off
        spatial = HolisticDepthModel(hdi_cfg)
        if cfg.init_checkpoint:
            load_checkpoint(Path(cfg.init_checkpoint), spatial, restore_rng=False)
            log.info("loaded spatial init from %s", cfg.init_checkpoint)
        else:
            log.warning("init_checkpoint unset — spatial model starts UNTRAINED (smoke test only)")

        temporal = TemporalStabilizerBank(
            levels=cfg.temporal.levels,
            feat_channels=hdi_cfg.decoder.channels_attn,
            d_model=cfg.temporal.d_model,
            num_blocks=cfg.temporal.num_blocks,
            downsample=cfg.temporal.downsample,
        )
        model = cls(spatial, temporal)
        if cfg.freeze_spatial:
            model.freeze_spatial()
        return model

    def freeze_spatial(self) -> None:
        n = 0
        for p in self.spatial.parameters():
            p.requires_grad_(False)
            n += p.numel()
        self.spatial.eval()
        t = sum(p.numel() for p in self.temporal.parameters())
        log.info("stage-2a freeze: %.2fM trainable (temporal) / %.1fM frozen (spatial)",
                 t / 1e6, n / 1e6)

    def train(self, mode: bool = True) -> "VideoDepthModel":
        super().train(mode)
        # a frozen spatial branch stays deterministic regardless of train()
        if not any(p.requires_grad for p in self.spatial.parameters()):
            self.spatial.eval()
        return self

    # ------------------------------------------------------------- state
    def reset_temporal_state(self) -> None:
        self.temporal.reset_state()

    # ------------------------------------------------------------- forward
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """One frame, carrying temporal state across successive calls
        (streaming). image (B,3,H,W) -> depth (B,1,H,W)."""
        H, W = image.shape[-2:]
        feats = self.spatial.backbone(image)
        dec = self.spatial.decoder(feats)
        dec.levels = self.temporal.apply_to(dec.levels)
        trace = self.spatial.refinement(dec)
        return F.interpolate(trace.final_depth, size=(H, W),
                             mode="bilinear", align_corners=False)

    def forward_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Ordered clip (B,T,3,H,W) -> depths (B,T,1,H,W); state reset at the
        clip start, carried (with autograd) across its frames — full BPTT with
        truncation = clip length."""
        self.reset_temporal_state()
        return torch.stack([self(images[:, t]) for t in range(images.shape[1])], dim=1)
