"""Ordered video-clip dataset for temporal (stage-2) Phase-1 training.

Serves fixed-length, stride-augmented clips of consecutive frames from one
sequence -- the FlashDepth training regime (short clips, full BPTT within the
clip, hidden-state reset per clip; longer strides teach long-video
generalization from short clips). The per-frame dataset stays untouched: this
is an additive path used only by ``train_hdi_temporal``.

Augmentation is decided ONCE per clip and applied identically to every frame
(a per-frame random hflip would present physically impossible motion to the
recurrent module).

Each item:
    images (T,3,H,W) float32, ImageNet-normalized
    depths (T,1,H,W) float32 metric metres, 0 = invalid
    meta   dict(sequence, start, stride)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from instancedepth.data.gid_dataset import IMAGENET_MEAN, IMAGENET_STD, GIDInstanceDepthDataset


@dataclass
class ClipDatasetConfig:
    annotations_root: str
    split: str = "train"
    image_size: Tuple[int, int] = (728, 1288)   # Phase-1 frame (divisible by 14 and 8)
    max_depth: float = 10.0
    clip_len: int = 5
    strides: Tuple[int, ...] = (1, 2, 4, 8)
    hflip_prob: float = 0.5

    def __post_init__(self) -> None:
        h, w = self.image_size
        assert h % 14 == 0 and w % 14 == 0, "image_size must be divisible by 14 (DINOv2/14)"
        assert self.clip_len >= 2 and all(s >= 1 for s in self.strides)


class GIDClipDataset(Dataset):
    """Index = every (sequence, start, stride) whose clip fits the sequence.
    With clip_len=5 and strides up to 8 the max span is 33 frames -- inside
    even the shortest (~50-frame) sequences of this dataset."""

    def __init__(self, cfg: ClipDatasetConfig) -> None:
        self.cfg = cfg
        root = Path(cfg.annotations_root)
        seq_ids = [s for s in (root / f"{cfg.split}.txt").read_text().splitlines() if s.strip()]

        self._manifests: Dict[str, Dict] = {}
        self.index: List[Tuple[str, int, int]] = []   # (sequence id, start frame idx, stride)
        for sid in seq_ids:
            with open(root / sid / "annotations.json") as f:
                man = json.load(f)
            self._manifests[sid] = man
            n = len(man["frames"])
            span = (cfg.clip_len - 1)
            for stride in cfg.strides:
                last_start = n - span * stride - 1
                for start in range(0, max(last_start + 1, 0)):
                    self.index.append((sid, start, stride))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        cfg = self.cfg
        sid, start, stride = self.index[i]
        man = self._manifests[sid]
        frame_keys = sorted(man["frames"].keys())
        H, W = cfg.image_size
        flip = cfg.split == "train" and random.random() < cfg.hflip_prob   # once per CLIP

        images, depths = [], []
        for t in range(cfg.clip_len):
            frame = man["frames"][frame_keys[start + t * stride]]
            rgb = GIDInstanceDepthDataset._load_rgb(frame["rgb"])
            depth = GIDInstanceDepthDataset._load_depth(frame, man["depth_scale_to_m"])
            depth[(depth < 0) | (depth > cfg.max_depth) | ~np.isfinite(depth)] = 0.0

            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            if flip:
                rgb, depth = rgb[:, ::-1], depth[:, ::-1]

            img = rgb.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
            images.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))))
            depths.append(torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0))

        return dict(
            images=torch.stack(images),   # (T,3,H,W)
            depths=torch.stack(depths),   # (T,1,H,W)
            meta=dict(sequence=sid, start=start, stride=stride),
        )


def collate_clips(batch: List[Dict]) -> Dict[str, object]:
    return dict(
        images=torch.stack([b["images"] for b in batch]),   # (B,T,3,H,W)
        depths=torch.stack([b["depths"] for b in batch]),   # (B,T,1,H,W)
        meta=[b["meta"] for b in batch],
    )
