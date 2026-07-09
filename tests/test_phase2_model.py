"""End-to-end Phase2Model test against the real COCO-pretrained checkpoint.
Requires network access (first run) to fetch
`facebook/mask2former-swin-large-coco-instance` from the HF Hub -- skipped
automatically if that's unavailable, mirroring
`tests/test_hdi_shapes.py::test_full_model_real_backbone`'s pattern for
Phase 1.

Run:  pytest tests/test_phase2_model.py -v
"""

from __future__ import annotations

import pytest
import torch

_CHECKPOINT = "facebook/mask2former-swin-large-coco-instance"


def _checkpoint_loadable() -> bool:
    try:
        from transformers import Mask2FormerConfig

        Mask2FormerConfig.from_pretrained(_CHECKPOINT)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _checkpoint_loadable(), reason=f"{_CHECKPOINT} not reachable (no network / not cached)")
def test_phase2_model_forward_shapes():
    from instancedepth.models.phase2.model import Phase2Model

    model = Phase2Model(checkpoint=_CHECKPOINT, num_classes=1)
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
