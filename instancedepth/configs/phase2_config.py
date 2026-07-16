"""Configuration for Phase 2 (Instance Depth Layer Prediction).

Separate from instancedepth/configs/config.py (Phase 1's HDIConfig) --
Phase 2 has its own decoupled backbone and does not
share Phase 1's config tree. Same dataclass + from_yaml pattern as Phase 1
and instancedepth/data_engine/config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class Phase2ModelConfig:
    checkpoint: str = "facebook/mask2former-swin-large-coco-instance"   # HF Hub id (arch identity; also the
                                                                         # fallback weights source if checkpoint_dir is unset)
    checkpoint_dir: Optional[str] = None    # local directory (e.g. a huggingface-cli/snapshot_download
                                             # snapshot moved here manually) -- preferred over any network call
    allow_hub_download: bool = False        # explicit opt-in; local checkpoint_dir is preferred (see
                                             # instancedepth/models/phase2/mask2former_wrapper.py)
    num_classes: int = 1   # this dataset: "person" only (gid_custom.yaml's category_ids)


@dataclass
class Phase2MatcherConfig:
    """Eq. 5-7's matching-cost weights. cost_class/cost_mask/cost_dice
    reuse Mask2Former's own tuned recipe (category 5); cost_depth is this
    paper's own addition with no specified weight (category 6, needs its
    own sweep)."""

    cost_class: float = 2.0
    cost_mask: float = 5.0
    cost_dice: float = 5.0
    cost_depth: float = 1.0
    num_points: int = 12544   # Mask2Former default: 112x112


@dataclass
class Phase2LossConfig:
    eos_coef: float = 0.1
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    weight_class: float = 2.0
    weight_mask: float = 5.0
    weight_dice: float = 5.0
    weight_depth: float = 1.0   # category 6, needs its own sweep


@dataclass
class Phase2DataConfig:
    annotations_root: str = "gid_custom"
    image_size: Tuple[int, int] = (736, 1280)   # (H, W); divisible by 32 for Swin's patch-merging stages
    max_depth: float = 10.0   # dataset metric range; matches the GIDDatasetConfig default
                              # Phase-2 training already uses implicitly (visualization
                              # reads it to colorize GT depth / Dep_i labels)
    min_instance_px: int = 64
    hflip_prob: float = 0.5
    color_jitter: float = 0.0

    def __post_init__(self) -> None:
        h, w = self.image_size
        assert h % 32 == 0 and w % 32 == 0, "image_size must be divisible by 32 (Swin-L's 4 patch-merging stages)"


@dataclass
class Phase2OptimConfig:
    """Mask2Former's own fine-tuning convention, NOT the
    paper's frozen-encoder/25k-iter/1e-5 schedule (calibrated for a
    fresh-decoder-only regime that doesn't apply once the backbone is a
    full Swin-L Mask2Former). lr/total_iters/grad_clip_norm are starting
    points requiring their own empirical sweep -- flagged, not assumed."""

    lr: float = 1.0e-4
    backbone_lr_mult: float = 0.1   # Mask2Former's own documented convention
    weight_decay: float = 0.05
    total_iters: int = 20000
    warmup_iters: int = 500
    poly_power: float = 0.9
    grad_clip_norm: float = 0.1
    precision: str = "bf16"
    batch_size: int = 2
    num_workers: int = 4
    log_every: int = 50
    ckpt_every: int = 1000
    eval_every: int = 2000


@dataclass
class Phase2Config:
    model: Phase2ModelConfig = field(default_factory=Phase2ModelConfig)
    matcher: Phase2MatcherConfig = field(default_factory=Phase2MatcherConfig)
    loss: Phase2LossConfig = field(default_factory=Phase2LossConfig)
    data: Phase2DataConfig = field(default_factory=Phase2DataConfig)
    optim: Phase2OptimConfig = field(default_factory=Phase2OptimConfig)
    seed: int = 2026
    run_name: str = "phase2_mask2former"
    run_root: str = "runs"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase2Config":
        def sub(klass, key):
            raw = dict(d.get(key, {}))
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = tuple(v)
            return klass(**raw)

        return cls(
            model=sub(Phase2ModelConfig, "model"),
            matcher=sub(Phase2MatcherConfig, "matcher"),
            loss=sub(Phase2LossConfig, "loss"),
            data=sub(Phase2DataConfig, "data"),
            optim=sub(Phase2OptimConfig, "optim"),
            seed=d.get("seed", 2026),
            run_name=d.get("run_name", "phase2_mask2former"),
            run_root=d.get("run_root", "runs"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase2Config":
        import yaml

        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_yaml_with_overrides(cls, path: str | Path, dotlist: list | None = None) -> "Phase2Config":
        import yaml

        from instancedepth.configs.config import _coerce, _set_dotted  # reuse Phase 1's helpers, not duplicated

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        for item in dotlist or []:
            key, _, value = item.partition("=")
            _set_dotted(raw, key.split("."), _coerce(value))
        return cls.from_dict(raw)
