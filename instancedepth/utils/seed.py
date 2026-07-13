"""Deterministic seeding, for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state/set_rng_state_all require a *CPU* ByteTensor --
    # load_checkpoint's torch.load(..., map_location=<training device>)
    # moves every tensor in the checkpoint (including these) onto that
    # device, so a CUDA-trained run's saved RNG tensors come back as
    # torch.cuda.ByteTensor and fail the underlying C++ type check
    # ("RNG state must be a torch.ByteTensor"). Force them back to CPU
    # regardless of where load_checkpoint's map_location put them.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["torch_cuda"]])
