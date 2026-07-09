"""Phase2Model: Mask2FormerWrapper (COCO-pretrained Swin-L, Option B) +
DepthLayerHead (Eq. 5-7), producing Phase2Output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.models.phase2.depth_head import DepthLayerHead
from instancedepth.models.phase2.mask2former_wrapper import Mask2FormerWrapper
from instancedepth.models.phase2.output import Phase2Output


class Phase2Model(nn.Module):
    def __init__(self, checkpoint: str = "facebook/mask2former-swin-large-coco-instance",
                 num_classes: int = 1) -> None:
        super().__init__()
        self.backbone_decoder = Mask2FormerWrapper(checkpoint, num_classes=num_classes)
        self.depth_head = DepthLayerHead(self.backbone_decoder.hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> Phase2Output:
        H, W = pixel_values.shape[-2:]
        raw = self.backbone_decoder(pixel_values)

        depth_layers = self.depth_head(raw.query_embeddings)   # (B, N)

        mask_logits = raw.mask_logits
        if mask_logits.shape[-2:] != (H, W):
            mask_logits = F.interpolate(mask_logits, size=(H, W), mode="bilinear", align_corners=False)

        return Phase2Output(
            mask_logits=mask_logits,
            class_logits=raw.class_logits,
            depth_layers=depth_layers,
            query_embeddings=raw.query_embeddings,
            image_hw=(H, W),
        )
