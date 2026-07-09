"""Phase 2 backbone: official, COCO-pretrained Mask2Former (Option B in the
Phase 2 plan -- decoupled from Phase 1's DINOv2, its own Swin-L backbone).

Reuses `transformers.Mask2FormerForUniversalSegmentation` as a maintained
dependency (no Detectron2 vendoring, same principle as Phase 1's DINOv2
wrapper) purely as a **feature/prediction extractor**: backbone + pixel
decoder + transformer decoder + the existing class/mask heads are used
as-is from the COCO checkpoint. Matching and loss (Eq. 5-7, including the
new depth term) are implemented separately in ``matcher.py``/``criterion.py``
rather than relying on the library's internal loss computation, precisely
so this code depends only on the *public, stable* output fields
(``class_queries_logits``, ``masks_queries_logits``, per-query hidden
states) instead of HF's internal matcher/criterion class API, which is not
part of the library's public surface and is more likely to shift between
versions.

**Field names below are this project's well-informed expectation, not yet
confirmed by execution** -- run ``scripts/verify_mask2former_api.py`` on the
Backend.AI server first (see the Phase 2 plan, implementation-order step 1)
and update ``_QUERY_EMBED_FIELDS`` / the asserts below if the real output
uses different names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger("instancedepth.models.phase2.mask2former_wrapper")

# Candidate field names for the per-query embedding *before* the class/mask
# heads are applied (needed as the depth head's input). Tried in order;
# first match wins. Update this list once scripts/verify_mask2former_api.py
# confirms the real name.
_QUERY_EMBED_FIELDS = (
    "transformer_decoder_last_hidden_state",
    "transformer_decoder_hidden_states",
    "last_hidden_state",
)


@dataclass
class RawMask2FormerPrediction:
    class_logits: torch.Tensor       # (B, N, num_classes+1)
    mask_logits: torch.Tensor        # (B, N, H, W) pre-sigmoid
    query_embeddings: torch.Tensor   # (B, N, hidden_dim) -- pre class/mask heads


class Mask2FormerWrapper(nn.Module):
    def __init__(self, checkpoint: str = "facebook/mask2former-swin-large-coco-instance",
                 num_classes: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        checkpoint : HF Hub id of the official COCO instance-segmentation
            checkpoint.
        num_classes : if set, reinitializes the classification head for a
            new class count (this dataset: 1 category, ``person`` --
            standard fine-tuning practice, per the Phase 2 plan SS6 step 5).
            ``None`` keeps COCO's original 80-class head (useful only for
            the verification step / sanity-checking the loaded weights).
        """
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        config: Mask2FormerConfig = Mask2FormerConfig.from_pretrained(checkpoint)
        if num_classes is not None:
            config.num_labels = num_classes

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=(num_classes is not None),
        )
        self.hidden_dim = config.hidden_dim if hasattr(config, "hidden_dim") else config.decoder_config.hidden_dim
        self.num_queries = config.num_queries

    def forward(self, pixel_values: torch.Tensor) -> RawMask2FormerPrediction:
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)

        query_embeddings = None
        for field in _QUERY_EMBED_FIELDS:
            candidate = getattr(out, field, None)
            if candidate is None:
                continue
            if isinstance(candidate, (tuple, list)):
                candidate = candidate[-1]
            if candidate.dim() == 3 and candidate.shape[1] == self.num_queries:
                query_embeddings = candidate
                break
        if query_embeddings is None:
            raise RuntimeError(
                "Could not find a (B, num_queries, hidden_dim) query-embedding "
                f"field on Mask2FormerForUniversalSegmentation's output among "
                f"{_QUERY_EMBED_FIELDS}. Run scripts/verify_mask2former_api.py "
                "and update instancedepth/models/phase2/mask2former_wrapper.py's "
                "_QUERY_EMBED_FIELDS with the confirmed field name -- refusing "
                "to silently guess."
            )

        return RawMask2FormerPrediction(
            class_logits=out.class_queries_logits,
            mask_logits=out.masks_queries_logits,
            query_embeddings=query_embeddings,
        )
