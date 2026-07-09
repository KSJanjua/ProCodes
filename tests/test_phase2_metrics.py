"""Tests for the occlusion-focused evaluation metrics (Phase 2 plan, SS6
step 7) -- synthetic masks only, no model/network needed.

Run:  pytest tests/test_phase2_metrics.py -v
"""

from __future__ import annotations

import torch

from instancedepth.utils.phase2_metrics import aggregate, evaluate_frame, has_overlapping_instances, mask_iou_matrix


def _rect_mask(h, w, y0, y1, x0, x1) -> torch.Tensor:
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_iou_matrix_identical_masks_gives_iou_one():
    m = _rect_mask(20, 20, 2, 10, 2, 10).unsqueeze(0)
    iou = mask_iou_matrix(m, m)
    assert torch.allclose(iou, torch.ones(1, 1), atol=1e-6)


def test_mask_iou_matrix_disjoint_masks_gives_iou_zero():
    a = _rect_mask(20, 20, 0, 5, 0, 5).unsqueeze(0)
    b = _rect_mask(20, 20, 10, 15, 10, 15).unsqueeze(0)
    iou = mask_iou_matrix(a, b)
    assert torch.allclose(iou, torch.zeros(1, 1), atol=1e-6)


def test_has_overlapping_instances():
    non_overlapping = torch.stack([
        _rect_mask(20, 20, 0, 5, 0, 5),
        _rect_mask(20, 20, 10, 15, 10, 15),
    ])
    overlapping = torch.stack([
        _rect_mask(20, 20, 0, 10, 0, 10),
        _rect_mask(20, 20, 5, 15, 5, 15),
    ])
    assert not has_overlapping_instances(non_overlapping)
    assert has_overlapping_instances(overlapping)


def test_evaluate_frame_perfect_prediction():
    gt = torch.stack([_rect_mask(20, 20, 0, 10, 0, 10), _rect_mask(20, 20, 10, 20, 10, 20)])
    gt_depths = torch.tensor([1.0, 2.0])
    pred = gt.clone()
    pred_scores = torch.tensor([0.99, 0.95])
    pred_depths = torch.tensor([1.0, 2.0])

    result = evaluate_frame(pred, pred_scores, pred_depths, gt, gt_depths, iou_threshold=0.5)
    assert result["tp"] == 2 and result["fp"] == 0 and result["fn"] == 0
    assert abs(result["mean_iou"] - 1.0) < 1e-6
    assert abs(result["depth_mae"]) < 1e-6


def test_evaluate_frame_missed_detection():
    gt = torch.stack([_rect_mask(20, 20, 0, 10, 0, 10), _rect_mask(20, 20, 10, 20, 10, 20)])
    gt_depths = torch.tensor([1.0, 2.0])
    pred = gt[:1].clone()   # only detect the first instance
    pred_scores = torch.tensor([0.9])
    pred_depths = torch.tensor([1.0])

    result = evaluate_frame(pred, pred_scores, pred_depths, gt, gt_depths, iou_threshold=0.5)
    assert result["tp"] == 1 and result["fp"] == 0 and result["fn"] == 1


def test_aggregate_precision_recall():
    frames = [
        dict(tp=2, fp=0, fn=0, mean_iou=1.0, depth_mae=0.0),
        dict(tp=1, fp=1, fn=1, mean_iou=0.6, depth_mae=0.2),
    ]
    agg = aggregate(frames)
    assert agg["precision"] == 3 / 4
    assert agg["recall"] == 3 / 4
    assert agg["num_frames"] == 2
