"""End-to-end Phase2Model test against the real COCO-pretrained checkpoint.

Skipped by default -- resolving the checkpoint used to probe
`Mask2FormerConfig.from_pretrained(hub_id)` directly, which hangs
indefinitely (no timeout) behind a blocking proxy, as observed on the
Backend.AI server. This now **only** runs if you explicitly point it at a
local, manually-downloaded snapshot via the ``PHASE2_CHECKPOINT_DIR`` env
var (see ``scripts/verify_mask2former_api.py``'s module docstring for the
download commands) -- never attempts a network call on its own.

Run (after downloading + transferring the checkpoint, mirroring how the
DINOv2 `.safetensors` file was handled):

    PHASE2_CHECKPOINT_DIR=/home/work/intern_storage/Ayush/weights/mask2former-swin-large-coco-instance \\
        pytest tests/test_phase2_model.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

_CHECKPOINT_DIR = os.environ.get("PHASE2_CHECKPOINT_DIR")


@pytest.mark.skipif(
    not _CHECKPOINT_DIR,
    reason="set PHASE2_CHECKPOINT_DIR to a local Mask2Former snapshot to run this test (never attempts a network call)",
)
def test_phase2_model_forward_shapes():
    from instancedepth.models.phase2.model import Phase2Model

    model = Phase2Model(checkpoint_dir=_CHECKPOINT_DIR, num_classes=1)
    model.eval()
    x = torch.randn(1, 3, 736, 1280)
    with torch.no_grad():
        out = model(x)

    assert out.image_hw == (736, 1280)
    assert out.mask_logits.shape[-2:] == (736, 1280)
    assert out.class_logits.shape[0] == 1
    assert out.class_logits.shape[-1] == 2   # 1 class + no-object
    assert out.depth_layers.shape[0] == 1
    assert out.depth_layers.shape[1] == out.mask_logits.shape[1]   # == num_queries
    assert (out.depth_layers >= 0).all()
