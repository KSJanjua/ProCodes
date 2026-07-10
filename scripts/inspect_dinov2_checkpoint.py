"""Identify what a locally-downloaded DINOv2-style checkpoint actually is,
without needing network access or a reference file.

Answers the practical question "which encoder did I actually load?" for a
`.safetensors` (or `.pth`) file whose provenance you no longer remember. It
inspects only key names + tensor shapes (no full load, no GPU), reuses the
*same* original-vs-HF format detector the training loader uses
(``instancedepth/models/backbone/dinov2_wrapper._looks_like_original_format``),
and checks for the tell-tale signatures that separate:

  * vanilla facebookresearch/dinov2 (what the paper's main method uses, and
    what this project loads by default), vs.
  * a Depth-Anything-V2 checkpoint (its DINOv2 encoder is fine-tuned for
    relative depth and ships a DPT ``depth_head`` and/or a ``pretrained.``
    encoder prefix), vs.
  * an already-HF-format ``Dinov2Model`` state dict.

Usage (on the Backend.AI server):

    python -m scripts.inspect_dinov2_checkpoint \
        --checkpoint /home/work/intern_storage/Ayush/weights/dinov2_vitl14.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

from instancedepth.models.backbone.dinov2_wrapper import _looks_like_original_format


def _load_keys_and_shapes(path: str):
    """Return {key: shape} without materializing tensors where possible."""
    p = Path(path)
    if p.suffix in (".safetensors", ".st"):
        from safetensors import safe_open
        out = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                out[k] = tuple(f.get_slice(k).get_shape())
        return out
    # .pth / .pt fallback (loads onto CPU)
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]            # our own training checkpoints wrap weights under "model"
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return {k: tuple(v.shape) for k, v in obj.items() if hasattr(v, "shape")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--show-keys", type=int, default=12, help="how many key names to print (0 = none)")
    args = ap.parse_args()

    shapes = _load_keys_and_shapes(args.checkpoint)
    keys = list(shapes.keys())
    n = len(keys)

    def any_key(pred) -> bool:
        return any(pred(k) for k in keys)

    is_original = _looks_like_original_format(keys)
    is_hf = any_key(lambda k: k.startswith("encoder.layer.")) or any_key(lambda k: k == "embeddings.cls_token")
    has_depth_head = any_key(lambda k: "depth_head" in k or k.startswith("head.") or "dpt" in k.lower())
    has_pretrained_prefix = any_key(lambda k: k.startswith("pretrained."))
    has_mask_token = any_key(lambda k: k.endswith("mask_token"))
    has_registers = any_key(lambda k: "register_tokens" in k)

    # infer ViT size from any qkv / attention weight's hidden dim
    hidden = None
    for k, s in shapes.items():
        if ("attn.qkv.weight" in k or "attention.attention.query.weight" in k) and len(s) == 2:
            hidden = s[0] if "qkv" not in k else s[1]  # qkv weight is (3*hidden, hidden); q weight is (hidden, hidden)
            hidden = s[1]
            break
    size_name = {384: "ViT-S", 768: "ViT-B", 1024: "ViT-L", 1536: "ViT-g"}.get(hidden, f"hidden={hidden}")

    pos_shape = next((s for k, s in shapes.items() if "pos_embed" in k or "position_embeddings" in k), None)

    print(f"\n=== {args.checkpoint} ===")
    print(f"tensors: {n}   inferred arch: {size_name}   pos-embed shape: {pos_shape}")
    if args.show_keys:
        print("sample keys:")
        for k in keys[: args.show_keys]:
            print(f"    {k:55s} {shapes[k]}")

    print("\nsignatures:")
    print(f"    original facebookresearch/dinov2 key format : {is_original}")
    print(f"    HF Dinov2Model key format                   : {is_hf}")
    print(f"    Depth-Anything-V2 DPT head present          : {has_depth_head}")
    print(f"    'pretrained.' encoder prefix (DAv2-style)   : {has_pretrained_prefix}")
    print(f"    mask_token present                          : {has_mask_token}")
    print(f"    register tokens (the *_reg variant)         : {has_registers}")

    print("\nverdict:")
    if has_depth_head or has_pretrained_prefix:
        print("    -> This looks like a Depth-Anything-V2 (or other DPT-wrapped) checkpoint,")
        print("       NOT a bare DINOv2 encoder. Its encoder is depth-fine-tuned.")
    elif is_original:
        print("    -> Original facebookresearch/dinov2 encoder, bare (no depth head).")
        print("       This is the vanilla, self-supervised DINOv2 the paper's main method uses.")
        print("       (This is also exactly what your training log reported: 'looks like an")
        print("        original facebookresearch/dinov2 state dict'.)")
    elif is_hf:
        print("    -> HuggingFace Dinov2Model-format encoder, bare (no depth head).")
    else:
        print("    -> Unrecognized layout -- inspect the sample keys above.")
    print("\nNOTE: vanilla DINOv2 and DAv2's encoder share the *identical* ViT architecture,")
    print("so shapes alone can't prove the weight *values* are vanilla vs. depth-fine-tuned")
    print("if someone re-saved DAv2's encoder under bare DINOv2 keys. The absence of any")
    print("depth_head/'pretrained.' keys plus the 'dinov2_vitl14' filename convention is")
    print("strong evidence it is the genuine vanilla checkpoint; a byte-for-byte guarantee")
    print("would require diffing against Meta's official dinov2_vitl14_pretrain weights.\n")


if __name__ == "__main__":
    main()
