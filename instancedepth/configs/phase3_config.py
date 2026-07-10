"""Configuration for Phase 3 (Occlusion-Aware Depth Refinement).

Design source of truth: ``docs/PHASE3_DESIGN.md``. Same dataclass + ``from_yaml``
/ ``from_dict`` / dotlist-override pattern as Phase 1 (``configs/config.py``)
and Phase 2 (``configs/phase2_config.py``); reuses their ``_coerce`` /
``_set_dotted`` helpers rather than duplicating them.

Phase 3 *composes* the two earlier phases rather than replacing them:

* Phase 1 (``HDIConfig``) -- the depth branch, **fine-tuned** here (paper
  Sec. 4.3: "fine-tuning both the depth encoder and decoder ... 1e-6").
* Phase 2 (``Phase2Config``) -- the instance decoder, **frozen** here
  (paper Sec. 4.3: "fixing the instance decoder").

So ``Phase3Config`` embeds a ``phase1`` and a ``phase2`` sub-config (each
loaded from its own YAML path).

Resolution handling (plan SS6.1): rather than forcing a single input
resolution divisible by both DINOv2/14 and Swin/32 (their only shared
multiple is 224, which is restrictive), Phase 3 keeps **each branch at its
own native resolution** and reconciles them with **normalized [0,1] box
coordinates** in ROIAlign. The dataset serves RGB/depth/masks at ``data
.image_size`` (the Phase-2 frame, where GT masks + frozen Phase-2 masks +
boxes all live and align); the model internally resizes RGB to Phase 1's
own resolution for the depth branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from instancedepth.configs.config import HDIConfig, _coerce, _set_dotted
from instancedepth.configs.phase2_config import Phase2Config


@dataclass
class Phase3CandidateConfig:
    """Eq. 8 candidate filtering + occlusion-pair construction.

    Thresholds are [Paper Specified] (Sec. 4.2.2): category confidence > 0.9,
    mask confidence > 0.8, overlap IoU > 0.1. The *reduction* that turns a
    dense mask-probability map into one scalar "mask confidence" is not
    defined by the paper -- ``mask_conf_thresh`` is applied to the mean
    foreground probability (see candidates.py). [Reasonable Assumption]
    """

    cat_conf_thresh: float = 0.9       # [Paper Specified]
    mask_conf_thresh: float = 0.8      # [Paper Specified]
    overlap_iou_thresh: float = 0.1    # [Paper Specified]
    mask_binarize_thresh: float = 0.5  # sigmoid->binary cut for IoU/box/score [Reasonable Assumption]
    guest_rule: str = "nearest_depth"  # "nearest_depth" (to main's Dep) [Strongly Inferred]
                                       # | "frontmost" (smallest Dep) -- ablation alt
    max_candidates: int = 50           # safety cap per image after filtering [Reasonable Assumption]


@dataclass
class Phase3HeadConfig:
    """The Occlusion Pair Relation Reasoning module (Phi_o, Eq. 8-9)."""

    roi_size: Tuple[int, int] = (28, 28)   # (Hp, Wp) ROIAlign output [Reasonable Assumption]
                                           # (Mask R-CNN mask head is 28x28; occlusion
                                           # boundaries want the finer grid, plan SS6.5)
    roi_sampling_ratio: int = 2            # torchvision roi_align sampling_ratio
    refine_granularity: str = "dense"      # "dense" (Reading D, primary) | "scalar" (Reading H)
                                           # -- plan SS5 decision, user-approved dense-primary
    hidden_dim: int = 256                  # Phi_o working width [Reasonable Assumption]
    num_conv: int = 3                      # 1x1 conv layers in Phi_o [Reasonable Assumption]
    use_multiscale_feat: bool = False      # False: F_obj = F_2 only (faithful default);
                                           # True: concat ROIAligned F_0,F_1,F_2 (plan SS3 Reading B)
    geom_coord: bool = True                # include normalized-coordinate channels in G_obj
    geom_global_depth: bool = True         # include ROIAligned global depth in G_obj
    geom_mask_logit: bool = True           # include ROIAligned mask logits in G_obj


@dataclass
class Phase3LossConfig:
    """Eq. 12: L_ref = lambda1 * L_obj + lambda2 * L_dist.

    lambda1/lambda2 have NO paper value (Table 6 only shows L_obj dominates,
    L_dist is secondary) -> sweep targets, not assumptions. ``holistic_weight``
    is the anti-forgetting regularizer (plan SS7.4): OFF in the faithful
    profile (Eq. 12 verbatim), ON at a small weight in the enhanced profile.
    Deviation, opt-in.
    """

    lambda_obj: float = 1.0            # weight on L_obj (Eq. 10) [Reasonable Assumption]
    lambda_dist: float = 0.5           # weight on L_dist (Eq. 11) [Reasonable Assumption]
    silog_lambda: float = 0.5          # SigLog variance-vs-mean term (reuses Phase 1's)
    holistic_weight: float = 0.0       # 0 = faithful (Eq. 12 only); >0 = anti-forgetting aux
    min_valid_roi_px: int = 16         # skip an ROI's L_obj if it has fewer valid GT px


@dataclass
class Phase3DataConfig:
    """Dataset served in the **Phase-2 frame** (where GT masks + frozen
    Phase-2 predictions + boxes align). size_divisor=32 for Swin-L."""

    annotations_root: str = "gid_custom"
    image_size: Tuple[int, int] = (736, 1280)   # (H, W); Phase-2 frame, /32
    size_divisor: int = 32
    max_depth: float = 10.0
    min_instance_px: int = 64
    hflip_prob: float = 0.5
    color_jitter: float = 0.0

    def __post_init__(self) -> None:
        h, w = self.image_size
        assert h % self.size_divisor == 0 and w % self.size_divisor == 0, (
            f"data.image_size {(h, w)} must be divisible by size_divisor="
            f"{self.size_divisor} (Swin-L stride)"
        )


@dataclass
class Phase3OptimConfig:
    """Paper Sec. 4.3 "Occlusion-Aware Joint Refinement": fine-tune the depth
    encoder+decoder at 1e-6 for 25k iters; the instance decoder is frozen.
    ``head_lr_mult`` gives the fresh Phi_o a higher LR (Phase 1's convention
    for fresh heads); [Reasonable Assumption] on its value.
    """

    lr: float = 1.0e-6                # [Paper Specified] depth-branch fine-tune LR
    head_lr_mult: float = 10.0        # Phi_o LR = lr * head_lr_mult [Reasonable Assumption]
    weight_decay: float = 0.01
    total_iters: int = 25000          # [Paper Specified]
    warmup_iters: int = 0
    poly_power: float = 0.9
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    grad_checkpointing: bool = False
    batch_size: int = 1               # two backbones resident (P1 grad + P2 frozen) -> small
    num_workers: int = 4
    log_every: int = 50
    ckpt_every: int = 1000
    eval_every: int = 2000


@dataclass
class Phase3Config:
    # Composed phase configs, loaded from their own YAML paths so the exact
    # Phase 1 / Phase 2 architectures are reproduced, not re-specified.
    phase1_config: str = "instancedepth/configs/hdi.yaml"
    phase2_config: str = "instancedepth/configs/phase2_mask2former.yaml"
    phase1_checkpoint: Optional[str] = None   # trained Phase 1 weights (required at runtime)
    phase2_checkpoint: Optional[str] = None   # trained Phase 2 weights (required at runtime)

    candidate: Phase3CandidateConfig = field(default_factory=Phase3CandidateConfig)
    head: Phase3HeadConfig = field(default_factory=Phase3HeadConfig)
    loss: Phase3LossConfig = field(default_factory=Phase3LossConfig)
    data: Phase3DataConfig = field(default_factory=Phase3DataConfig)
    optim: Phase3OptimConfig = field(default_factory=Phase3OptimConfig)

    seed: int = 2026
    run_name: str = "phase3_refine"
    run_root: str = "runs"

    # resolved in from_dict
    phase1: Optional[HDIConfig] = None
    phase2: Optional[Phase2Config] = None

    def __post_init__(self) -> None:
        assert self.head.refine_granularity in ("dense", "scalar"), (
            f"head.refine_granularity must be 'dense' or 'scalar', got "
            f"{self.head.refine_granularity!r}"
        )
        assert self.candidate.guest_rule in ("nearest_depth", "frontmost")

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase3Config":
        def sub(klass, key):
            raw = dict(d.get(key, {}))
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = tuple(v)
            return klass(**raw)

        sub_keys = ("candidate", "head", "loss", "data", "optim", "phase1", "phase2")
        top = {k: v for k, v in d.items() if k not in sub_keys}

        cfg = cls(
            candidate=sub(Phase3CandidateConfig, "candidate"),
            head=sub(Phase3HeadConfig, "head"),
            loss=sub(Phase3LossConfig, "loss"),
            data=sub(Phase3DataConfig, "data"),
            optim=sub(Phase3OptimConfig, "optim"),
            **top,
        )
        cfg.phase1 = HDIConfig.from_yaml(cfg.phase1_config)
        cfg.phase2 = Phase2Config.from_yaml(cfg.phase2_config)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase3Config":
        import yaml

        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_yaml_with_overrides(cls, path: str | Path, dotlist: Optional[list] = None) -> "Phase3Config":
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        for item in dotlist or []:
            key, _, value = item.partition("=")
            _set_dotted(raw, key.split("."), _coerce(value))
        return cls.from_dict(raw)
