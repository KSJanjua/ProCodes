"""Regression: Trainer.fit must log LRs for ANY number of param groups.

The smoke run of videodepth/run_pipeline.sh crashed here:
    self.optimizer.param_groups[1]["lr"]   -> IndexError
whenever the optimizer has a single group -- which happens for the video
temporal stage (only the stabilizer trains) and for Phase 3 with
freeze_phase1 (the empty depth group is dropped). This drives a real (tiny)
Trainer.fit() loop with a one-group optimizer and asserts it completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch

from instancedepth.engine.trainer import Trainer


@dataclass
class _Optim:
    lr: float = 1e-3
    weight_decay: float = 0.0
    total_iters: int = 2
    warmup_iters: int = 0
    poly_power: float = 0.9
    grad_clip_norm: float = 1.0
    precision: str = "fp32"
    log_every: int = 1          # force the LR-logging path every iter
    ckpt_every: int = 1
    eval_every: int = 100
    num_workers: int = 0
    batch_size: int = 1


@dataclass
class _Data:
    annotations_root: str = "."   # RunManifest.build reads this; a dataclass so asdict/json work


@dataclass
class _Cfg:
    optim: _Optim = field(default_factory=_Optim)
    seed: int = 0
    run_name: str = "t"
    data: _Data = field(default_factory=_Data)


def _loader():
    x = {"x": torch.randn(1, 4)}
    return [x, x, x]   # a trivially iterable "dataloader"


def test_fit_single_group_optimizer_logs_without_indexerror(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    cfg = _Cfg(data=_Data(annotations_root=str(tmp_path)))   # empty dir -> split hashes None, fine

    def compute_loss(m, batch, device):
        out = m(batch["x"].to(device))
        return {"total": out.pow(2).mean()}

    # single param group -- the exact shape that triggered the crash
    def one_group(m, ocfg):
        return torch.optim.SGD(m.parameters(), lr=ocfg.lr)

    trainer = Trainer(
        cfg=cfg, model=model, compute_loss=compute_loss,
        train_loader=_loader(), run_dir=tmp_path / "run",
        eval_fn=None, device=torch.device("cpu"),
        build_optimizer_fn=one_group,
    )
    assert len(trainer.optimizer.param_groups) == 1
    trainer.fit()                       # must not raise IndexError
    assert (tmp_path / "run" / "latest.pth").exists()
