"""Configuration for Phase 1 (Holistic Depth Initialization).

Follows the same pattern already established in
``instancedepth/data_engine/config.py``: plain dataclasses + ``from_yaml``/
``from_dict``, no external config framework.

Every field that is a genuine research decision (not a paper-stated fact)
is documented at its point of use; this module only defines *what* is
configurable, not *why* each default was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class BackboneConfig:
    """DINOv2 backbone (vanilla, self-supervised checkpoint -- NOT Depth
    Anything V2's fine-tuned variant
    matters)."""

    name: str = "facebook/dinov2-large"          # HF hub id; used for the model *config* (arch) always,
                                                  # and as the weights source only if checkpoint_path is unset
    checkpoint_path: Optional[str] = (
        "/home/work/intern_storage/Ayush/weights/dinov2_vitl14.safetensors"
    )                                            # local .safetensors preferred over any download
    hook_layers: Tuple[int, ...] = (5, 14, 23)   # 0-indexed transformer blocks fed to the decoder
    freeze: bool = False                          # paper: encoder is trainable in this stage
    allow_hub_download: bool = False              # explicit opt-in; local checkpoint is preferred (see
                                                  # instancedepth/models/backbone/dinov2_wrapper.py)


@dataclass
class DecoderConfig:
    """Depth Range Feature Decoder (Fig. 5): 3 coarse-to-fine levels."""

    target_fractions: Tuple[float, ...] = (1 / 8, 1 / 4, 1 / 2)   # of input resolution
    patch_kernels: Tuple[int, ...] = (16, 8, 4)                    # per-level patchify kernel/stride
    channels_dec: int = 256          # channel width after backbone->level resize
    channels_attn: int = 256         # patch-attention working dimension
    attn_heads: int = 8
    attn_blocks: int = 1             # transformer blocks per level's patch attention


@dataclass
class BinRefinementConfig:
    """Eq. 1-4 iterative bin refinement."""

    rd: int = 5              # number of depth bins (paper's own best ablation: 2m-wide bins)
    max_depth: float = 10.0  # MAX_d; must match instancedepth/configs/gid_custom.yaml's max_depth_m


@dataclass
class TemporalConfig:
    """FlashDepth-style temporal alignment.
    Disabled by default: every existing profile stays the per-frame,
    paper-faithful baseline bit-identically. Enabled only by the dedicated
    stage-2 fine-tune profile (hdi_temporal.yaml)."""

    enabled: bool = False
    core: str = "convgru"              # "convgru" (dependency-free default); core is swappable by design
    d_model: int = 128                 # recurrent working width (module stays ~1% of model params)
    num_blocks: int = 2                # stacked ConvGRU cells
    downsample: float = 0.1            # per-side shrink before recurrence [FlashDepth value]
    levels: Tuple[int, ...] = (2,)     # decoder level(s) to align; 2 = F_2 (last-before-heads,
                                       # the structural analogue of FlashDepth's placement)
    freeze_spatial: bool = True        # stage 2a: train ONLY the temporal module
    clip_len: int = 5                  # frames per training clip [FlashDepth video_length]
    clip_strides: Tuple[int, ...] = (1, 2, 4, 8)   # stride augmentation; max span 4*8+1=33 frames,
                                                    # safe for the shortest (~50-frame) sequences
    init_checkpoint: Optional[str] = None   # trained per-frame Phase-1 weights (stage 1 product)

    def __post_init__(self) -> None:
        assert self.core in ("convgru",), f"unknown temporal.core {self.core!r}"
        assert all(l in (0, 1, 2) for l in self.levels)
        assert self.clip_len >= 2, "a clip needs at least 2 frames for recurrence to matter"


@dataclass
class CameraIntrinsics:
    """Only needed for the optional disparity auxiliary loss / diagnostics.
    None-able by design -- the faithful baseline never reads this."""

    focal_px: Optional[float] = None
    width_px: Optional[int] = None
    source: str = "unknown"   # "metadata" | "config" | "unknown"


@dataclass
class LossConfig:
    regression: str = "silog"                 # registry key: silog | l1 | l2 | berhu
    silog_lambda: float = 0.5                 # Eigen et al. variance-vs-mean trade-off
    deep_supervision_weights: Tuple[float, float] = (0.5, 0.25)   # weight on D_1, D_2 (not D_final)
    bin_bce_weight: float = 1.0               # ordinal per-bin BCE weight
    disparity_aux_weight: float = 0.0         # 0 = off (faithful); >0 only in hdi_enhanced.yaml
    gradient_matching_weight: float = 0.0     # multi-scale gradient matching (Eigen & Fergus 2015 /
                                              # MiDaS L_reg): sharpens depth edges, which SigLog --
                                              # a pointwise statistic -- cannot penalize. 0 = off
                                              # (faithful); >0 in hdi_enhanced.yaml. Phase 1's loss
                                              # is unspecified by the paper, so this is a free choice.
    gradient_matching_scales: int = 4         # strided subsample levels (MiDaS's scheme)


@dataclass
class DataConfig:
    annotations_root: str = "gid_custom"
    image_size: Tuple[int, int] = (728, 1288)  # (H, W); divisible by 14 (DINOv2/14) and 8 (decoder pyramid)
    min_instance_px: int = 64
    hflip_prob: float = 0.5
    color_jitter: float = 0.0                  # off by default; no paper support either way


@dataclass
class OptimConfig:
    lr: float = 1e-5              # paper Sec. 4.3: "Global Depth Range Pretraining" initial LR
    backbone_lr_mult: float = 1.0
    head_lr_mult: float = 10.0    # DAv2 convention: fresh heads trained faster than the pretrained backbone
    weight_decay: float = 0.01
    total_iters: int = 55000      # paper Sec. 4.3
    warmup_iters: int = 0
    poly_power: float = 0.9       # DAv2's (1 - iter/total)**0.9 schedule
    grad_clip_norm: float = 1.0
    precision: str = "bf16"       # fp32 | fp16 | bf16 -- configurable, not assumed
    batch_size: int = 4
    num_workers: int = 4
    log_every: int = 50
    ckpt_every: int = 1000
    eval_every: int = 2000


@dataclass
class HDIConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    bins: BinRefinementConfig = field(default_factory=BinRefinementConfig)
    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 2026
    run_name: str = "hdi_faithful"
    run_root: str = "runs"

    def __post_init__(self) -> None:
        h, w = self.data.image_size
        assert h % 14 == 0 and w % 14 == 0, "image_size must be divisible by 14 (DINOv2/14)"
        assert h % 8 == 0 and w % 8 == 0, (
            "image_size must be divisible by 8 so the 1/8, 1/4, 1/2 decoder "
            "pyramid levels are exact integers"
        )
        if self.loss.disparity_aux_weight > 0:
            assert self.camera.focal_px and self.camera.width_px, (
                "disparity_aux_weight > 0 requires camera intrinsics "
                "(camera.focal_px / camera.width_px) to be set -- refusing "
                "to silently fall back to a guessed constant"
            )

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HDIConfig":
        def sub(klass, key):
            raw = dict(d.get(key, {}))
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = tuple(v)
            return klass(**raw)

        return cls(
            backbone=sub(BackboneConfig, "backbone"),
            decoder=sub(DecoderConfig, "decoder"),
            bins=sub(BinRefinementConfig, "bins"),
            camera=sub(CameraIntrinsics, "camera"),
            temporal=sub(TemporalConfig, "temporal"),
            loss=sub(LossConfig, "loss"),
            data=sub(DataConfig, "data"),
            optim=sub(OptimConfig, "optim"),
            seed=d.get("seed", 2026),
            run_name=d.get("run_name", "hdi_faithful"),
            run_root=d.get("run_root", "runs"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HDIConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_yaml_with_overrides(cls, path: str | Path, dotlist: Optional[list] = None) -> "HDIConfig":
        """Load a YAML profile, then apply ``key.subkey=value`` CLI overrides."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for item in dotlist or []:
            key, _, value = item.partition("=")
            _set_dotted(raw, key.split("."), _coerce(value))
        return cls.from_dict(raw)


def _coerce(value: str) -> Any:
    """Best-effort str -> bool/int/float/str for dotlist CLI overrides."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _set_dotted(d: Dict[str, Any], keys: list, value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
