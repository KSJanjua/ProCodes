"""Phase 2, implementation-order step 1 (see the Phase 2 plan, SS5.5/SS6):
verify exactly what `transformers.Mask2FormerForUniversalSegmentation` (and
its underlying `Mask2FormerModel` trunk) actually expose, before writing
the depth-layer head or the `Phase2Output` contract against assumed field
names.

This project's own knowledge of the exact output field names
(`class_queries_logits`, `masks_queries_logits`, `transformer_decoder_last_hidden_state`,
etc.) is a well-informed expectation, not something verified by running the
code -- run this script once, on the Backend.AI server (this needs network
access to the HF Hub the first time, to download the checkpoint), and paste
the output back so `models/phase2/mask2former_wrapper.py` and
`models/phase2/output.py` (Phase2Output) can be written against the real,
confirmed API surface rather than assumed field names.

Usage:

    python -m scripts.verify_mask2former_api \\
        --checkpoint facebook/mask2former-swin-large-coco-instance
"""

from __future__ import annotations

import argparse
import logging

import torch

log = logging.getLogger("scripts.verify_mask2former_api")


def _describe(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        log.info("  %-45s tensor  shape=%-30s dtype=%s", name, tuple(value.shape), value.dtype)
    elif isinstance(value, (tuple, list)):
        log.info("  %-45s %s of length %d", name, type(value).__name__, len(value))
        if len(value) and isinstance(value[0], torch.Tensor):
            log.info("      element[0] shape=%s", tuple(value[0].shape))
    elif value is None:
        log.info("  %-45s None", name)
    else:
        log.info("  %-45s %s", name, type(value).__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="facebook/mask2former-swin-large-coco-instance")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerModel

    log.info("=== Loading %s ===", args.checkpoint)
    full_model = Mask2FormerForUniversalSegmentation.from_pretrained(args.checkpoint)
    full_model.eval()

    config: Mask2FormerConfig = full_model.config
    log.info("config: num_queries=%s, hidden_dim=%s(?), num_labels=%s, backbone=%s",
             getattr(config, "num_queries", "?"),
             getattr(getattr(config, "backbone_config", None), "embed_dim", "?"),
             getattr(config, "num_labels", "?"),
             type(getattr(config, "backbone_config", None)).__name__)

    dummy = torch.randn(1, 3, 384, 384)  # Swin default-friendly size; adjust if it errors

    log.info("\n=== Mask2FormerForUniversalSegmentation(pixel_values=...) output fields ===")
    with torch.no_grad():
        full_out = full_model(pixel_values=dummy, output_hidden_states=True)
    for field in full_out.keys():
        _describe(field, full_out[field])

    log.info("\n=== Underlying Mask2FormerModel trunk output fields (for query embeddings) ===")
    trunk: Mask2FormerModel = full_model.model
    with torch.no_grad():
        trunk_out = trunk(pixel_values=dummy, output_hidden_states=True)
    for field in trunk_out.keys():
        _describe(field, trunk_out[field])

    log.info("\n=== Class/mask prediction heads on the ForUniversalSegmentation wrapper ===")
    for name, module in full_model.named_children():
        if "class" in name.lower() or "mask" in name.lower() or "embed" in name.lower():
            log.info("  %s: %s", name, type(module).__name__)

    log.info(
        "\nPaste this entire output back. It confirms: (1) which output field "
        "holds per-query embeddings suitable for a new depth head (expected: "
        "something like transformer_decoder_last_hidden_state, shape "
        "(1, num_queries, hidden_dim)), (2) which fields hold raw per-query "
        "class/mask logits (expected: class_queries_logits, masks_queries_logits), "
        "(3) whether attaching a depth head to the trunk output or the full "
        "wrapper's hidden state is more direct."
    )


if __name__ == "__main__":
    main()
