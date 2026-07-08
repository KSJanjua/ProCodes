"""Canonical disparity conversion + camera intrinsics handling (plan SS8/SS9).

The network's only output space is metric depth (meters). Disparity is a
deterministic, invertible, *post-hoc* transform of a metric depth
prediction, used only for (a) an optional auxiliary training loss and (b)
always-on diagnostic evaluation metrics -- never as an internal
representation, never as the bin axis. See ``instancedepth/losses/
hdi_losses.py`` for how (a) is wired in, and ``instancedepth/engine/
evaluate_hdi.py`` for (b).

    disp = focal_px / (width_px * depth)

This is *not* true stereo disparity (there's no baseline term) -- it's a
resolution-normalized inverse depth, sometimes called "canonical disparity"
in monocular contexts. The important property used elsewhere in this
codebase (see ``losses/hdi_losses.py``'s docstring) is that a *log-space*
loss computed on this quantity is mathematically identical to the same loss
computed on depth (the per-image constant `focal_px/width_px` cancels out
of any log-difference), which is why the auxiliary loss here is plain L1,
not a log-space loss.
"""

from __future__ import annotations

import torch

from instancedepth.configs.config import CameraIntrinsics

_MIN_DEPTH_CLAMP = 1e-4
_MAX_DEPTH_CLAMP = 1e4


def depth_to_canonical_disparity(depth: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor:
    """
    Parameters
    ----------
    depth : any-shape tensor of metric depth (meters).
    intrinsics : must have ``focal_px`` and ``width_px`` set (raises
        otherwise -- see plan SS9: this function never silently guesses a
        camera constant).

    Returns
    -------
    Tensor of the same shape, canonical disparity (units: 1/meters).
    """
    if intrinsics.focal_px is None or intrinsics.width_px is None:
        raise ValueError(
            "depth_to_canonical_disparity requires intrinsics.focal_px and "
            "intrinsics.width_px to be set. This function is only reachable "
            "when loss.disparity_aux_weight > 0 or when disparity "
            "diagnostics are requested; HDIConfig.__post_init__ should have "
            "already validated this -- if you're seeing this error, camera "
            "intrinsics were not resolved before calling into this code path."
        )
    depth = torch.clamp(depth, _MIN_DEPTH_CLAMP, _MAX_DEPTH_CLAMP)
    const = intrinsics.focal_px / intrinsics.width_px
    return const / depth
