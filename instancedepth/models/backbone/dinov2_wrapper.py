"""DINOv2 backbone wrapper for the Holistic Depth Initialization model.

This wraps DINOv2, our pretrained backbone, and exposes three intermediate feature maps — 
from blocks 5, 14, and 23 — so the decoder gets a multi-scale view: shallow features for detail, 
deep features for semantics. It starts from the vanilla self-supervised DINOv2 weights, which is what the paper specifies, 
and we have a DAv2 variant that loads a depth-fine-tuned encoder for the ablation. 

Input : 
A batch of images, ImageNet-normalized, with H and W divisible by 14. Example: (4, 3, 728, 1288) (4 images).

Output : 
A Python list of 3 tensors, each (4, 1024, 52, 92).  (52 = 728/14, 92 = 1288/14).  
The three tensors are the outputs of blocks 5, 14, and 23 of the DINOv2 encoder.
All three are the same size but come from different depths of the network (shallow = fine detail, deep = high-level meaning). 
1024 = DINOv2-Large's feature width (embed_dim).

Design decisions and their provenance:

- We depend on ``transformers.Dinov2Model`` rather than vendoring the
  original facebookresearch/dinov2 (or Depth-Anything-V2's copy of it) code.
  This is a "prefer a maintained dependency over copying code" choice.
- We initialize from the *original, vanilla* DINOv2 self-supervised
  checkpoint, not Depth Anything V2's depth-adapted backbone -- the paper's
  own Sec. 4.3 says "the pretrained DINOv2 [37]" (Oquab et al.), which is a
  different checkpoint from DAv2's further-fine-tuned encoder (used only in
  the paper's *ablation baseline*, Sec. 5.3, which is a separate model).
- The local ``.safetensors`` checkpoint is preferred over any network
  download; see ``load_dinov2_weights`` below for exactly how that's
  resolved, including the case where the checkpoint is in the *original*
  facebookresearch/dinov2 key-naming convention rather than HF's.

Architecture hyperparameters for each DINOv2 size are hardcoded here (not
fetched from the HF Hub) so that constructing the model architecture never
requires network access -- only *loading weights* has a
network-vs-local-file choice, and local is preferred there too. These
hyperparameters are copied from HuggingFace's own official conversion
script (`transformers/models/dinov2/convert_dinov2_to_hf.py`,
`get_dinov2_config`), which is the authoritative source for how the
original facebookresearch/dinov2 checkpoints map onto ``Dinov2Config``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model

from instancedepth.configs.config import BackboneConfig

log = logging.getLogger("instancedepth.models.backbone")

# Pretraining resolution used by every official DINOv2 checkpoint. This only
# determines the *shape* of the position-embedding table baked into the
# checkpoint (37x37 patches + 1 CLS token for patch_size=14); at runtime we
# feed whatever resolution the decoder actually needs (e.g. 728x1288) and
# rely on Dinov2Model's built-in bicubic position-embedding interpolation
# (verified directly in `transformers.models.dinov2.modeling_dinov2
# .Dinov2Embeddings.interpolate_pos_encoding`, which triggers automatically
# whenever the input isn't the pretraining resolution -- no extra flag
# needed in this version of `transformers`).
_PRETRAIN_IMAGE_SIZE = 518
_PATCH_SIZE = 14

# Parameters a feature-extraction DINOv2 checkpoint may legitimately omit and
# that are never read in our forward pass, so a missing entry is harmless.
# - ``embeddings.mask_token``: only used for masked-image-modeling pretraining
#   (never triggered here).
# - ``layernorm.{weight,bias}``: HF Dinov2Model's *final* layernorm, applied
#   only to produce last_hidden_state/pooler. We extract intermediate
#   hidden_states (the hook layers, pre-final-norm block outputs), so this
#   final norm is never applied to anything we read. Kept tolerant because a
#   Depth-Anything-V2 encoder export may not carry it; on the vanilla DINOv2
#   checkpoint it is present anyway, so this never weakens that path.
# See load_dinov2_weights below.
_ALLOWED_MISSING_KEYS = {"embeddings.mask_token", "layernorm.weight", "layernorm.bias"}

# facebook/dinov2-{small,base,large,giant} architecture hyperparameters,
# matching HF's own convert_dinov2_to_hf.py::get_dinov2_config exactly.
_SIZE_CONFIGS: Dict[str, Dict[str, object]] = {
    "vits": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, use_swiglu_ffn=False),
    "vitb": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, use_swiglu_ffn=False),
    "vitl": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, use_swiglu_ffn=False),
    "vitg": dict(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, use_swiglu_ffn=True),
}


def _infer_size_key(name: str) -> str:
    name = name.lower()
    for key in ("vits", "vitb", "vitl", "vitg"):
        if key in name:
            return key
    # also accept the HF hub naming: facebook/dinov2-{small,base,large,giant}
    mapping = {"small": "vits", "base": "vitb", "large": "vitl", "giant": "vitg"}
    for word, key in mapping.items():
        if word in name:
            return key
    raise ValueError(
        f"Cannot infer DINOv2 size from backbone name '{name}'. Expected one "
        f"of vits/vitb/vitl/vitg, or facebook/dinov2-{{small,base,large,giant}}."
    )


def build_dinov2_config(name: str) -> Dinov2Config:
    """Construct the Dinov2Config purely from local hyperparameters -- no
    network access, regardless of where the weights end up coming from."""
    size = _infer_size_key(name)
    return Dinov2Config(image_size=_PRETRAIN_IMAGE_SIZE, patch_size=_PATCH_SIZE, **_SIZE_CONFIGS[size])


# --------------------------------------------------------------------------- #
# Weight loading: local safetensors (either HF-format or original-format),
# with network download as an explicit, opt-in-only fallback.
# --------------------------------------------------------------------------- #
# DINOv2 weights come in different "key-naming" styles depending on where you downloaded them. This renames the original Meta format into the HuggingFace format so the weights fit our model. 
def _rename_original_dinov2_state_dict(state_dict: Dict[str, torch.Tensor], config: Dinov2Config) -> Dict[str, torch.Tensor]:
    """Convert an *original* facebookresearch/dinov2 (torch.hub-style)
    state dict into HF ``Dinov2Model`` key naming.

    This mirrors HuggingFace's own official conversion logic in
    ``transformers/models/dinov2/convert_dinov2_to_hf.py`` (functions
    ``create_rename_keys`` / ``read_in_q_k_v``), adapted in-place here so we
    don't need that (dev-only, not shipped in the pip package) script as a
    runtime dependency.
    """
    sd = dict(state_dict)
    hidden_size = config.hidden_size

    def rename(old: str, new: str) -> None:
        if old in sd:
            sd[new] = sd.pop(old)

    rename("cls_token", "embeddings.cls_token")
    rename("mask_token", "embeddings.mask_token")
    rename("pos_embed", "embeddings.position_embeddings")
    rename("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight")
    rename("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias")

    for i in range(config.num_hidden_layers):
        rename(f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight")
        rename(f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias")
        rename(f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight")
        rename(f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias")

        if config.use_swiglu_ffn:
            rename(f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.weights_in.weight")
            rename(f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.weights_in.bias")
            rename(f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.weights_out.weight")
            rename(f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.weights_out.bias")
        else:
            rename(f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight")
            rename(f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias")
            rename(f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight")
            rename(f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias")

        rename(f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1")
        rename(f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1")

        rename(f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight")
        rename(f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias")

        qkv_w_key = f"blocks.{i}.attn.qkv.weight"
        qkv_b_key = f"blocks.{i}.attn.qkv.bias"
        if qkv_w_key in sd:
            qkv_w = sd.pop(qkv_w_key)
            sd[f"encoder.layer.{i}.attention.attention.query.weight"] = qkv_w[:hidden_size, :]
            sd[f"encoder.layer.{i}.attention.attention.key.weight"] = qkv_w[hidden_size : 2 * hidden_size, :]
            sd[f"encoder.layer.{i}.attention.attention.value.weight"] = qkv_w[-hidden_size:, :]
        if qkv_b_key in sd:
            qkv_b = sd.pop(qkv_b_key)
            sd[f"encoder.layer.{i}.attention.attention.query.bias"] = qkv_b[:hidden_size]
            sd[f"encoder.layer.{i}.attention.attention.key.bias"] = qkv_b[hidden_size : 2 * hidden_size]
            sd[f"encoder.layer.{i}.attention.attention.value.bias"] = qkv_b[-hidden_size:]

    rename("norm.weight", "layernorm.weight")
    rename("norm.bias", "layernorm.bias")
    return sd


def _looks_like_original_format(keys: List[str]) -> bool:
    """Heuristic: original facebookresearch/dinov2 checkpoints have
    top-level keys like 'cls_token', 'blocks.0.attn.qkv.weight',
    'patch_embed.proj.weight'; HF checkpoints have 'embeddings.cls_token',
    'encoder.layer.0...'."""
    markers = ("cls_token", "patch_embed.proj.weight", "blocks.0.attn.qkv.weight")
    return any(k in keys for k in markers)


def _looks_like_dav2_format(keys: List[str]) -> bool:
    """Depth-Anything-V2 checkpoints bundle a DINOv2 encoder (under a
    ``pretrained.`` prefix, in original facebookresearch/dinov2 key naming)
    together with a DPT depth head (under ``depth_head.``) -- e.g.
    ``pretrained.blocks.0.attn.qkv.weight``, ``depth_head.projects.0.weight``.
    The ``pretrained.`` prefix is the reliable signature."""
    return any(k.startswith("pretrained.") for k in keys)

#  if you hand it a Depth-Anything-V2 checkpoint (which is DINOv2 + a depth head bundled together), it extracts just the DINOv2 encoder and throws away the depth head
def _convert_dav2_encoder_state_dict(
    state_dict: Dict[str, torch.Tensor], config: Dinov2Config
) -> Dict[str, torch.Tensor]:
    """Extract just the DINOv2 encoder from a Depth-Anything-V2 checkpoint
    and map it into HF ``Dinov2Model`` key naming.

    DAv2 = DINOv2 encoder (fine-tuned for depth, stored under
    ``pretrained.`` in original DINOv2 key naming) + a DPT depth head
    (``depth_head.``). We keep only the encoder, strip the prefix, drop the
    head, then reuse ``_rename_original_dinov2_state_dict`` -- the encoder is
    architecturally identical to vanilla DINOv2 ViT-L/14, only its weight
    *values* differ (that's the whole point of using it: depth-fine-tuned
    features, matching the encoder behind the paper's reported holistic-stage
    numbers -- see the InstanceDepth paper's Sec. 5.3 ablations)."""
    encoder: Dict[str, torch.Tensor] = {}
    dropped_head = 0
    prefix = "pretrained."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            encoder[k[len(prefix):]] = v
        elif k.startswith("depth_head."):
            dropped_head += 1
        # anything else (rare aux buffers) is intentionally ignored
    log.info(
        "Depth Anything V2 checkpoint: extracted %d encoder tensors (from "
        "'pretrained.'), dropped %d DPT depth-head tensors.",
        len(encoder), dropped_head,
    )
    return _rename_original_dinov2_state_dict(encoder, config)

# reads either .safetensors or .pth, unwrapping common holder keys.
def _read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a raw state dict from either a ``.safetensors`` or a
    ``.pth``/``.pt`` file. The vanilla DINOv2 checkpoint this project uses is
    ``.safetensors``; DAv2's official checkpoints ship as ``.pth``. Unwraps a
    top-level ``model``/``state_dict`` holder if the file has one."""
    import os

    if os.path.splitext(path)[1].lower() in (".safetensors", ".st"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for wrapper in ("model", "state_dict"):
            if wrapper in obj and isinstance(obj[wrapper], dict):
                return obj[wrapper]
    return obj


def load_dinov2_weights(model: Dinov2Model, cfg: BackboneConfig) -> None:
    """Populate ``model`` in place, preferring ``cfg.checkpoint_path`` and
    only ever touching the network if ``cfg.allow_hub_download`` is True and
    the local checkpoint is unusable.

    Raises
    ------
    FileNotFoundError
        if ``checkpoint_path`` is set but doesn't exist and
        ``allow_hub_download`` is False.
    RuntimeError
        if the checkpoint loads but its keys don't match either the HF or
        the original-DINOv2 naming convention (so silently proceeding with
        a partially-initialized backbone is never allowed).
    """
    if cfg.checkpoint_path:
        import os

        if not os.path.isfile(cfg.checkpoint_path):
            if cfg.allow_hub_download:
                log.warning(
                    "Local checkpoint '%s' not found; allow_hub_download=True, "
                    "falling back to HF Hub weights for '%s'.",
                    cfg.checkpoint_path, cfg.name,
                )
                _load_from_hub(model, cfg.name)
                return
            raise FileNotFoundError(
                f"backbone.checkpoint_path='{cfg.checkpoint_path}' does not exist "
                f"and backbone.allow_hub_download is False. Set "
                f"allow_hub_download: true in the config if you intend to "
                f"download '{cfg.name}' from the Hugging Face Hub instead."
            )

        state_dict = _read_state_dict(cfg.checkpoint_path)
        keys = list(state_dict.keys())

        if _looks_like_dav2_format(keys):
            log.info(
                "Checkpoint '%s' looks like a Depth Anything V2 checkpoint; "
                "extracting its DINOv2 encoder (under 'pretrained.') and "
                "dropping the DPT depth head, then renaming to HF Dinov2Model "
                "format.",
                cfg.checkpoint_path,
            )
            state_dict = _convert_dav2_encoder_state_dict(state_dict, model.config)
        elif _looks_like_original_format(keys):
            log.info(
                "Checkpoint '%s' looks like an original facebookresearch/dinov2 "
                "state dict; applying key-renaming to HF Dinov2Model format.",
                cfg.checkpoint_path,
            )
            state_dict = _rename_original_dinov2_state_dict(state_dict, model.config)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # ``embeddings.mask_token`` is only consumed during masked-image-
        # modeling pretraining (when ``bool_masked_pos`` is passed to the
        # backbone). We only ever extract features, so we never pass it and
        # the token is never read -- a feature-extraction DINOv2 checkpoint
        # legitimately omits it. Treat it (and any other key in
        # _ALLOWED_MISSING_KEYS) as harmless; everything else must match.
        blocking_missing = [k for k in missing if k not in _ALLOWED_MISSING_KEYS]
        if blocking_missing or unexpected:
            raise RuntimeError(
                "DINOv2 checkpoint did not load cleanly.\n"
                f"  checkpoint: {cfg.checkpoint_path}\n"
                f"  missing keys ({len(blocking_missing)}): {blocking_missing[:10]}{' ...' if len(blocking_missing) > 10 else ''}\n"
                f"  unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}\n"
                "This usually means the checkpoint's size (vits/vitb/vitl/vitg) "
                "doesn't match backbone.name, or it's in a third naming "
                "convention this loader doesn't recognize yet. Refusing to "
                "silently proceed with a partially-initialized backbone."
            )
        if missing:
            log.info(
                "Loaded DINOv2 weights from '%s' (ignored harmless missing keys "
                "never used in feature extraction: %s).",
                cfg.checkpoint_path, missing,
            )
        else:
            log.info("Loaded DINOv2 weights from local checkpoint '%s'.", cfg.checkpoint_path)
        return

    if cfg.allow_hub_download:
        _load_from_hub(model, cfg.name)
        return

    raise ValueError(
        "backbone.checkpoint_path is unset and backbone.allow_hub_download "
        "is False -- refusing to initialize the backbone with random "
        "weights. Set checkpoint_path to a local .safetensors file, or set "
        "allow_hub_download: true to fetch '{}' from the Hugging Face Hub.".format(cfg.name)
    )


def _load_from_hub(model: Dinov2Model, name: str) -> None:
    pretrained = Dinov2Model.from_pretrained(name)
    model.load_state_dict(pretrained.state_dict())
    log.info("Loaded DINOv2 weights from the Hugging Face Hub ('%s').", name)


# --------------------------------------------------------------------------- #
class DINOv2Backbone(nn.Module):
    """Wraps ``transformers.Dinov2Model`` and exposes exactly the 3
    intermediate hidden states the Depth Range Feature Decoder needs
    (``cfg.hook_layers``), reshaped to spatial feature maps.
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        config = build_dinov2_config(cfg.name)
        self.model = Dinov2Model(config)
        load_dinov2_weights(self.model, cfg)

        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_register_tokens = getattr(config, "num_register_tokens", 0)

        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "DINOv2Backbone":
        super().train(mode)
        if self.cfg.freeze:
            self.model.eval()  # keep frozen backbone's BatchNorm/dropout (if any) in eval mode
        return self

    # runs DINOv2 asking for all hidden states, then picks out the outputs of blocks 23, 14, 5 (hook_layers), drops the special CLS/register tokens, and reshapes the remaining patch tokens back into spatial (B, 1024, h, w) maps.
    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        pixel_values : (B, 3, H, W), H and W divisible by ``patch_size`` (14).

        Returns
        -------
        List of 3 tensors, one per ``cfg.hook_layers`` entry, each
        (B, embed_dim, H/14, W/14) -- CLS/register tokens dropped, patch
        tokens reshaped to spatial. All 3 share the same spatial resolution
        (a ViT preserves resolution across depth); they differ in semantic
        depth, which is what makes them a genuine "multiscale" feature set
        once the decoder resizes each to its own target resolution.
        """
        B, _, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"input ({H},{W}) must be divisible by patch_size={self.patch_size}"
        )
        h, w = H // self.patch_size, W // self.patch_size

        ctx = torch.no_grad() if self.cfg.freeze else torch.enable_grad()
        with ctx:
            out = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # hidden_states: tuple of length num_hidden_layers + 1 (index 0 = post-embedding,
        # before any transformer block). Block i's *output* is hidden_states[i + 1].
        hidden_states = out.hidden_states
        features: List[torch.Tensor] = []
        n_extra = 1 + self.num_register_tokens  # CLS (+ optional register tokens)
        for layer_idx in self.cfg.hook_layers:
            hs = hidden_states[layer_idx + 1]              # (B, n_extra + h*w, embed_dim)
            patch_tokens = hs[:, n_extra:, :]                # (B, h*w, embed_dim)
            spatial = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
            features.append(spatial)
        return features
        
        # DINOv2 outputs a flat list of tokens per patch; this turns that flat list back into a 2-D grid (52×92) so it looks like an image-shaped feature map again