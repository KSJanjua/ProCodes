"""Phase 2, implementation-order step 1:
verify exactly what `transformers.Mask2FormerForUniversalSegmentation` (and
its underlying `Mask2FormerModel` trunk) actually expose, before writing
the depth-layer head or the `Phase2Output` contract against assumed field
names.

This project's own knowledge of the exact output field names
(`class_queries_logits`, `masks_queries_logits`, `transformer_decoder_last_hidden_state`,
etc.) is a well-informed expectation, not something verified by running the
code -- run this script once and paste the output back so
`models/phase2/mask2former_wrapper.py` and `models/phase2/output.py`
(Phase2Output) can be written against the real, confirmed API surface
rather than assumed field names.

If the Backend.AI server can reach huggingface.co directly:

    python -m scripts.verify_mask2former_api \\
        --checkpoint facebook/mask2former-swin-large-coco-instance

If it's behind a blocking proxy (as observed -- a bare Hub id hangs
indefinitely with no timeout), download the checkpoint on a machine that
*does* have Hub access first, exactly as you already did for the DINOv2
`.safetensors` file:

    pip install "huggingface_hub[cli]"
    huggingface-cli download facebook/mask2former-swin-large-coco-instance \\
        --local-dir ./mask2former-swin-large-coco-instance

    # (equivalently, in Python:)
    #   from huggingface_hub import snapshot_download
    #   snapshot_download(repo_id="facebook/mask2former-swin-large-coco-instance",
    #                      local_dir="./mask2former-swin-large-coco-instance")

Then copy the resulting folder (config.json, preprocessor_config.json, and
the model weights file -- typically a few hundred MB to ~1GB) to the
Backend.AI server, e.g. next to the DINOv2 weights:

    /home/work/intern_storage/Ayush/weights/mask2former-swin-large-coco-instance/

and point this script at the local folder instead of the Hub id -- this
uses `local_files_only=True` internally, so it will never attempt a network
call and cannot hang:

    python -m scripts.verify_mask2former_api \\
        --checkpoint-dir /home/work/intern_storage/Ayush/weights/mask2former-swin-large-coco-instance

Once confirmed working, put the same local path in
`instancedepth/configs/phase2_mask2former.yaml`'s `model.checkpoint_dir`.
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
    ap.add_argument("--checkpoint", default="facebook/mask2former-swin-large-coco-instance",
                     help="HF Hub id -- only used if --checkpoint-dir is not given")
    ap.add_argument("--checkpoint-dir", default=None,
                     help="local directory with a manually-downloaded snapshot; if set, no network "
                          "call is ever attempted (local_files_only=True)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerModel

    source = args.checkpoint_dir or args.checkpoint
    local_files_only = args.checkpoint_dir is not None
    log.info("=== Loading %s (local_files_only=%s) ===", source, local_files_only)
    full_model = Mask2FormerForUniversalSegmentation.from_pretrained(source, local_files_only=local_files_only)
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
