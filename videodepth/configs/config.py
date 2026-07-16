"""Config for the videodepth package (temporal stage + video-aware Phase 3).

Composes the existing Phase-1 config (``hdi_config`` YAML path) rather than
re-specifying the spatial architecture — the temporal stage is strictly
additive on top of a trained Phase-1 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class TemporalHeadConfig:
    """Lightweight streaming stabilizer (ConvGRU) on decoder features.

    Kept deliberately small (mentor constraint: VDA's temporal attention is
    too heavy). Streaming O(1) state -> arbitrary-length videos at inference.
    """

    levels: Tuple[int, ...] = (2,)   # decoder levels to stabilize (2 = finest F_2)
    d_model: int = 128
    num_blocks: int = 2
    # 0.25 (was 0.1 in the inert first attempt): a 1/4-side grid can correct
    # mid-frequency flicker, not just global drift, while staying cheap.
    downsample: float = 0.25


@dataclass
class ClipConfig:
    """Motion-aware clip sampling (see data/motion_clips.py).

    Strides up to 24 make a clip_len=5 clip span 97 frames — the range over
    which this dataset's motion is actually visible (user-reported). Short
    sequences simply can't host long-span clips (index construction skips
    them); ``min_seq_len`` additionally drops whole sequences (e.g. the
    50-frame ones) from temporal training.
    """

    clip_len: int = 5
    strides: Tuple[int, ...] = (1, 2, 4, 8, 16, 24)
    min_seq_len: int = 60            # skip sequences shorter than this entirely
    motion_weighting: bool = True    # sample clips ∝ measured GT motion
    motion_floor: float = 0.15       # weight floor so static clips still appear
    motion_cache: str = "motion_scores.json"   # cached under run_dir


@dataclass
class VideoLossConfig:
    temporal_weight: float = 1.0     # weight on L_tgm vs the spatial loss
    tgm_order: int = 1               # 1 = VDA first differences; 2 adds accel
    tgm_log_space: bool = True


@dataclass
class VideoOptimConfig:
    lr: float = 5.0e-5               # only the small temporal head trains
    weight_decay: float = 0.01
    total_iters: int = 12000
    warmup_iters: int = 250
    poly_power: float = 0.9
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    batch_size: int = 2
    num_workers: int = 4
    log_every: int = 50
    ckpt_every: int = 1000
    eval_every: int = 1000


@dataclass
class VideoEvalConfig:
    """Model selection = streaming abs_rel + tae_weight * TAE.

    This fixes the selection-blindness defect (docs/AUDIT_2026.md): the old
    temporal run selected best.pth on shuffled per-frame abs_rel, a mode where
    the temporal state is reset every batch and the module is invisible.
    """

    max_frames: int = 800            # periodic-eval subset (full split offline)
    tae_weight: float = 4.0          # TAE ~0.05 vs abs_rel ~0.08 -> comparable scale


@dataclass
class VideoConfig:
    hdi_config: str = "instancedepth/configs/hdi_enhanced.yaml"
    init_checkpoint: Optional[str] = None    # trained Phase-1 weights (required for real runs)
    freeze_spatial: bool = True              # stage-2a: only the temporal head learns

    temporal: TemporalHeadConfig = field(default_factory=TemporalHeadConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    loss: VideoLossConfig = field(default_factory=VideoLossConfig)
    optim: VideoOptimConfig = field(default_factory=VideoOptimConfig)
    eval: VideoEvalConfig = field(default_factory=VideoEvalConfig)

    seed: int = 2026
    run_name: str = "video_temporal"
    run_root: str = "runs"

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoConfig":
        def sub(klass, key):
            raw = dict(d.get(key, {}))
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = tuple(v)
            return klass(**raw)

        sub_keys = ("temporal", "clips", "loss", "optim", "eval")
        top = {k: v for k, v in d.items() if k not in sub_keys}
        return cls(
            temporal=sub(TemporalHeadConfig, "temporal"),
            clips=sub(ClipConfig, "clips"),
            loss=sub(VideoLossConfig, "loss"),
            optim=sub(VideoOptimConfig, "optim"),
            eval=sub(VideoEvalConfig, "eval"),
            **top,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VideoConfig":
        import yaml
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_yaml_with_overrides(cls, path: str | Path, dotlist=None) -> "VideoConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        for item in dotlist or []:
            key, _, value = item.partition("=")
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            try:
                import ast
                node[parts[-1]] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                node[parts[-1]] = value
        return cls.from_dict(raw)
