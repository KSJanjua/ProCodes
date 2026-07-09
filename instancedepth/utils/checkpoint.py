"""Checkpointing: a rolling `latest.pth` (bounded disk usage) plus periodic
and best-metric snapshots, each carrying model + optimizer + scheduler +
iteration + RNG state (plan SS14/SS16) -- not just model weights, so a
resumed run is bit-for-bit continuable, not just "restarted with warm
weights".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from instancedepth.utils.seed import load_rng_state, rng_state


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    iteration: int,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            iteration=iteration,
            best_metric=best_metric,
            rng_state=rng_state(),
            extra=extra or {},
        ),
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    # weights_only=False: these checkpoints are entirely self-authored by
    # save_checkpoint above (never loaded from an untrusted/external
    # source), and rng_state()'s numpy.random.get_state() pickles through
    # numpy._core.multiarray._reconstruct, which PyTorch >=2.6's new
    # weights_only=True default (torch.load's own default since that
    # version) does not allowlist.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and ckpt.get("rng_state") is not None:
        load_rng_state(ckpt["rng_state"])
    return ckpt
