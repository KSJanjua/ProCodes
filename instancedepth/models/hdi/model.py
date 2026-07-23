"""HolisticDepthModel -- 
full Phase 1 module. It chains backbone → decoder → iterative refinement, upsamples the final depth to input resolution, and returns a structured output. It exports F_2 as feat_final and all three decoder levels for Phase 3, and keeps D_1/D_2 and the bin outputs for deep supervision.

composes the backbone, Depth Range Feature Decoder,
and Eq. 1-4 iterative bin refinement into the full Phase 1 (Holistic Depth
Initialization) model.

Forward pass, end to end:

    RGB (B,3,H,W)
      -> DINOv2Backbone            -> 3 feature maps, native (H/14,W/14) each
      -> DepthRangeFeatureDecoder  -> F_0 (1/8), F_1 (1/4), F_2 (1/2)
      -> IterativeBinRefinement    -> D_0..D_3, S_0..S_2, at decoder resolutions
      -> final upsample (bilinear, DPT-style last step)
      -> HolisticDepthOutput(depth_final, seg_final @ full res; feat_final = F_2 @ native res)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.configs.config import HDIConfig
from instancedepth.models.backbone.dinov2_wrapper import DINOv2Backbone
from instancedepth.models.hdi.depth_range_decoder import DepthRangeFeatureDecoder
from instancedepth.models.hdi.iterative_refinement import IterativeBinRefinement
from instancedepth.models.hdi.output import HolisticDepthOutput
from instancedepth.models.hdi.temporal import TemporalAligner


class HolisticDepthModel(nn.Module):
    def __init__(self, cfg: HDIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv2Backbone(cfg.backbone)
        self.decoder = DepthRangeFeatureDecoder(
            in_channels=self.backbone.embed_dim,
            image_hw=cfg.data.image_size,
            cfg=cfg.decoder,
        )
        self.refinement = IterativeBinRefinement(
            feat_channels=cfg.decoder.channels_attn,
            cfg=cfg.bins,
        )
        # FlashDepth-style temporal alignment:
        # inserted between decoder and refinement heads, one aligner per
        # configured level. Absent (None) when disabled -> per-frame baseline
        # is bit-identical.
        if cfg.temporal.enabled:
            self.temporal_aligners = nn.ModuleDict({
                str(lvl): TemporalAligner(
                    feat_channels=cfg.decoder.channels_attn,
                    d_model=cfg.temporal.d_model,
                    num_blocks=cfg.temporal.num_blocks,
                    downsample=cfg.temporal.downsample,
                ) for lvl in cfg.temporal.levels
            })
        else:
            self.temporal_aligners = None

    def reset_temporal_state(self) -> None:
        """Zero the temporal memory at sequence boundaries (no-op when the
        temporal module is disabled)."""
        if self.temporal_aligners is not None:
            for aligner in self.temporal_aligners.values():
                aligner.reset_state()

    def forward(self, image: torch.Tensor) -> HolisticDepthOutput:
        H, W = image.shape[-2:]
        backbone_feats = self.backbone(image)
        decoder_feats = self.decoder(backbone_feats)
        if self.temporal_aligners is not None:
            for lvl_str, aligner in self.temporal_aligners.items():
                lvl = int(lvl_str)
                decoder_feats.levels[lvl] = aligner(decoder_feats.levels[lvl])
        trace = self.refinement(decoder_feats)

        depth_final = F.interpolate(trace.final_depth, size=(H, W), mode="bilinear", align_corners=False)
        seg_final = F.interpolate(trace.final_bins, size=(H, W), mode="bilinear", align_corners=False)

        return HolisticDepthOutput(
            depth_final=depth_final,
            seg_final=seg_final,
            feat_final=decoder_feats.finest,
            depth_levels=trace.depths[1:3],   # D_1, D_2 (deep supervision targets; D_0 is the seed, D_3 == depth_final pre-upsample)
            seg_levels=trace.bins,            # S_0, S_1, S_2
            image_hw=(H, W),
            feat_hw=tuple(decoder_feats.finest.shape[-2:]),
            feat_levels=decoder_feats.levels,  # [F_0, F_1, F_2] for Phase 3's optional multi-scale F_obj (contract v1.2)
        )
