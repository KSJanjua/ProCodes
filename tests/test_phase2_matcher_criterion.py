"""Shape/correctness tests for Phase 2's matcher and criterion -- the most
novel, hand-written code in the Phase 2 redesign (everything else reuses
official Mask2Former via `transformers`). No network access or pretrained
checkpoint needed; operates on synthetic tensors.

Run:  pytest tests/test_phase2_matcher_criterion.py -v
"""

from __future__ import annotations

import torch

from instancedepth.models.phase2.criterion import Phase2Criterion
from instancedepth.models.phase2.depth_head import DepthLayerHead
from instancedepth.models.phase2.matcher import Phase2HungarianMatcher
from instancedepth.models.phase2.output import Phase2Output


def _dummy_batch(batch=2, num_queries=20, num_classes=1, h=32, w=32, gt_per_image=(2, 3)):
    class_logits = torch.randn(batch, num_queries, num_classes + 1)
    mask_logits = torch.randn(batch, num_queries, h, w)
    depth_preds = torch.rand(batch, num_queries) * 8 + 0.5

    targets = []
    for b in range(batch):
        g = gt_per_image[b % len(gt_per_image)]
        targets.append(dict(
            labels=torch.zeros(g, dtype=torch.long),   # single class: "person" -> id 0
            masks=(torch.rand(g, h, w) > 0.7).float(),
            depths=torch.rand(g) * 8 + 0.5,
        ))
    return class_logits, mask_logits, depth_preds, targets


def test_matcher_returns_one_to_one_indices_within_bounds():
    class_logits, mask_logits, depth_preds, targets = _dummy_batch()
    matcher = Phase2HungarianMatcher(num_points=64)   # small for a fast test
    indices = matcher(class_logits, mask_logits, depth_preds, targets)

    assert len(indices) == len(targets)
    for (row, col), tgt in zip(indices, targets):
        assert row.numel() == col.numel() == tgt["labels"].numel()
        assert row.max() < class_logits.shape[1]
        assert col.max() < tgt["labels"].numel()
        assert len(set(row.tolist())) == row.numel(), "matcher must not assign one query to two GT instances"


def test_matcher_handles_empty_targets():
    class_logits, mask_logits, depth_preds, _ = _dummy_batch(batch=1)
    empty_targets = [dict(labels=torch.zeros(0, dtype=torch.long), masks=torch.zeros(0, 32, 32), depths=torch.zeros(0))]
    matcher = Phase2HungarianMatcher(num_points=64)
    indices = matcher(class_logits, mask_logits, depth_preds, empty_targets)
    assert indices[0][0].numel() == 0 and indices[0][1].numel() == 0


def test_criterion_forward_is_finite_and_has_all_terms():
    class_logits, mask_logits, depth_preds, targets = _dummy_batch()
    matcher = Phase2HungarianMatcher(num_points=64)
    criterion = Phase2Criterion(num_classes=1, num_points=64)

    indices = matcher(class_logits, mask_logits, depth_preds, targets)
    losses = criterion(class_logits, mask_logits, depth_preds, targets, indices)

    for key in ("loss_class", "loss_mask", "loss_dice", "loss_depth", "total"):
        assert key in losses
        assert torch.isfinite(losses[key]), f"{key} is not finite: {losses[key]}"


def test_depth_head_output_shape_and_nonnegative():
    head = DepthLayerHead(hidden_dim=256)
    embeddings = torch.randn(2, 20, 256) * 5
    depths = head(embeddings)
    assert depths.shape == (2, 20)
    assert (depths >= 0).all()


def test_phase2_output_scores_and_mask_confidence():
    out = Phase2Output(
        mask_logits=torch.randn(1, 5, 16, 16),
        class_logits=torch.randn(1, 5, 2),   # 1 class + no-object
        depth_layers=torch.rand(1, 5) * 8,
        query_embeddings=torch.randn(1, 5, 256),
        image_hw=(16, 16),
    )
    scores = out.scores()
    conf = out.mask_confidence()
    assert scores.shape == (1, 5)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert conf.shape == (1, 5, 16, 16)
    assert (conf >= 0).all() and (conf <= 1).all()
