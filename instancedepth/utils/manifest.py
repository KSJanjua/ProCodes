"""Run manifest: everything needed to reproduce a training run months later
. Written once at training start, re-verified (RNG/optimizer/
scheduler state) at every checkpoint.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def git_commit_hash(repo_root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=repo_root, stderr=subprocess.DEVNULL
        ) != 0
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None, "note": "not a git repo, or git unavailable"}


def _file_hash(path: Path) -> Optional[str]:
    import hashlib

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _split_hash(annotations_root: Path, split: str) -> Optional[str]:
    import hashlib

    p = annotations_root / f"{split}.txt"
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


@dataclass
class RunManifest:
    experiment_id: str
    architecture_version: str
    config: Dict[str, Any]
    config_hash: str
    git: Dict[str, Any]
    software_versions: Dict[str, str]
    train_split_hash: Optional[str]
    test_split_hash: Optional[str]
    meta_json_hash: Optional[str]
    seed: int
    created_at: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: Any, repo_root: Path, architecture_version: str = "hdi-1.0") -> "RunManifest":
        """``cfg`` is duck-typed -- any dataclass with ``.run_name``,
        ``.seed``, ``.data.annotations_root`` works (Phase 1's ``HDIConfig``,
        Phase 2's ``Phase2Config``, ...)."""
        import hashlib

        cfg_dict = asdict(cfg)
        cfg_hash = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:16]
        ann_root = Path(cfg.data.annotations_root)

        return cls(
            experiment_id=f"{cfg.run_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            architecture_version=architecture_version,
            config=cfg_dict,
            config_hash=cfg_hash,
            git=git_commit_hash(repo_root),
            software_versions={
                "python": sys.version,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda": torch.version.cuda or "cpu",
            },
            train_split_hash=_split_hash(ann_root, "train"),
            test_split_hash=_split_hash(ann_root, "test"),
            meta_json_hash=_file_hash(ann_root / "meta.json"),
            seed=cfg.seed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
