"""DINOv2 backbone wrapper for the Holistic Depth Initialization model.

Design decisions and their provenance (see the implementation plan, SS3, for
the full argument):

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

        from safetensors.torch import load_file

        state_dict = load_file(cfg.checkpoint_path)
        keys = list(state_dict.keys())

        if _looks_like_original_format(keys):
            log.info(
                "Checkpoint '%s' looks like an original facebookresearch/dinov2 "
                "state dict; applying key-renaming to HF Dinov2Model format.",
                cfg.checkpoint_path,
            )
            state_dict = _rename_original_dinov2_state_dict(state_dict, model.config)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # position_embeddings is expected to be present and shape-matched at
        # _PRETRAIN_IMAGE_SIZE; everything else should match exactly.
        if missing or unexpected:
            raise RuntimeError(
                "DINOv2 checkpoint did not load cleanly.\n"
                f"  checkpoint: {cfg.checkpoint_path}\n"
                f"  missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
                f"  unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}\n"
                "This usually means the checkpoint's size (vits/vitb/vitl/vitg) "
                "doesn't match backbone.name, or it's in a third naming "
                "convention this loader doesn't recognize yet. Refusing to "
                "silently proceed with a partially-initialized backbone."
            )
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
        once the decoder resizes each to its own target resolution (SS1/SS4
        of the plan).
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
