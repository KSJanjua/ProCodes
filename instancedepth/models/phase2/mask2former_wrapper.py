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

**Confirmed by execution** (``scripts/verify_mask2former_api.py``, run
against the real ``facebook/mask2former-swin-large-coco-instance``
checkpoint -- see the Phase 2 plan, implementation-order step 1): the full
``Mask2FormerForUniversalSegmentation`` wrapper's own output (not just the
underlying bare ``Mask2FormerModel`` trunk) already exposes everything this
module needs --

- ``transformer_decoder_last_hidden_state``: ``(B, num_queries, hidden_dim)``
  = ``(1, 200, 256)`` in the verification run -- batch-first, matches
  ``_QUERY_EMBED_FIELDS``'s first (and, in practice, only-needed) entry.
  Note ``transformer_decoder_hidden_states`` (the per-layer tuple, second
  candidate below) is *query-first* -- ``(num_queries, B, hidden_dim)`` --
  a different axis order from ``..._last_hidden_state``; harmless here only
  because the shape check below (``shape[1] == num_queries``) correctly
  rejects it and falls through, but don't assume every field on this output
  shares one layout.
- ``class_queries_logits``: ``(B, num_queries, num_classes+1)``,
  ``masks_queries_logits``: ``(B, num_queries, H/4, W/4)`` pre-sigmoid --
  exactly as assumed.
- ``config.hidden_dim`` (256, confirmed present at the top level of
  ``Mask2FormerConfig`` -- there is no ``decoder_config`` sub-object on this
  config class at all, unlike some other DETR-family configs).

The load report also shows ``MISSING`` for
``pixel_level_module.encoder.swin.layernorm.{weight,bias}`` -- confirmed
harmless (checked directly against ``Mask2FormerPixelLevelModule.forward``
in the installed ``transformers`` version): the pixel decoder only consumes
the backbone's ``.feature_maps``, never this layernorm's output, so it has
zero effect on ``masks_queries_logits``, ``class_queries_logits``, or
``transformer_decoder_last_hidden_state``. Same category of finding as
``dinov2_wrapper.py``'s ``embeddings.mask_token`` (present in the report,
irrelevant to this forward path) -- not suppressed here since HF's own
``from_pretrained`` only logs it (unlike our custom DINOv2 loader, this
never raises), so there's nothing to whitelist.
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


def _resolve_checkpoint_source(
    checkpoint: str,
    checkpoint_dir: Optional[str],
    allow_hub_download: bool,
) -> tuple[str, bool]:
    """Mirrors instancedepth/models/backbone/dinov2_wrapper.py's
    load_dinov2_weights: prefer a local, manually-downloaded snapshot,
    fail loudly if it's missing rather than silently falling back to a
    (possibly proxy-blocked, hanging) network call, and only ever touch
    the network if explicitly opted in.

    Returns (source, local_files_only) where ``source`` is either
    ``checkpoint_dir`` or the Hub id ``checkpoint``.
    """
    import os

    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"model.checkpoint_dir='{checkpoint_dir}' does not exist. "
                "Download the checkpoint on a machine with working Hub access "
                "and copy the folder here -- see scripts/verify_mask2former_api.py's "
                "module docstring for the exact commands, and "
                "instancedepth/configs/phase2_mask2former.yaml for where to "
                "point checkpoint_dir once it's transferred."
            )
        # local_files_only=True guarantees zero network calls even for
        # metadata/etag checks -- this is what actually prevents the hang
        # you saw (Mask2FormerConfig.from_pretrained on a bare Hub id will
        # still try to *validate* against the Hub unless told not to).
        return checkpoint_dir, True

    if not allow_hub_download:
        raise ValueError(
            "model.checkpoint_dir is unset and model.allow_hub_download is "
            "False -- refusing to silently attempt a Hub download (which "
            "hangs indefinitely behind a blocking proxy, as you saw). Set "
            "checkpoint_dir to a local snapshot, or set "
            "allow_hub_download: true if this machine truly has working "
            "Hub access."
        )
    return checkpoint, False


class Mask2FormerWrapper(nn.Module):
    def __init__(
        self,
        checkpoint: str = "facebook/mask2former-swin-large-coco-instance",
        checkpoint_dir: Optional[str] = None,
        allow_hub_download: bool = False,
        num_classes: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        checkpoint : HF Hub id of the official COCO instance-segmentation
            checkpoint (used as the architecture identity always, and as
            the weights source only if ``checkpoint_dir`` is unset).
        checkpoint_dir : local directory containing a full snapshot of
            ``checkpoint`` (config.json, preprocessor_config.json, and the
            model weights file) -- e.g. downloaded via ``huggingface-cli
            download`` or ``snapshot_download`` on a machine with working
            Hub access, then copied over. Preferred over any network call.
        allow_hub_download : explicit opt-in to download directly from the
            Hub. Off by default -- see ``_resolve_checkpoint_source``.
        num_classes : if set, reinitializes the classification head for a
            new class count (this dataset: 1 category, ``person`` --
            standard fine-tuning practice, per the Phase 2 plan SS6 step 5).
            ``None`` keeps COCO's original 80-class head (useful only for
            the verification step / sanity-checking the loaded weights).
        """
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        source, local_files_only = _resolve_checkpoint_source(checkpoint, checkpoint_dir, allow_hub_download)

        config: Mask2FormerConfig = Mask2FormerConfig.from_pretrained(source, local_files_only=local_files_only)
        if num_classes is not None:
            config.num_labels = num_classes

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            source,
            config=config,
            ignore_mismatched_sizes=(num_classes is not None),
            local_files_only=local_files_only,
        )
        # config.hidden_dim (256) is confirmed present directly on
        # Mask2FormerConfig -- no decoder_config sub-object exists on this
        # config class, so the previous hasattr-guarded fallback to
        # config.decoder_config.hidden_dim was unreachable dead code.
        self.hidden_dim = config.hidden_dim
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
