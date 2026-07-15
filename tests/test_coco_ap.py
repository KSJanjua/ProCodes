"""Correctness tests for the standalone COCO-style mask AP (the metric
Mask2Former reports), pinned against cases whose exact AP is known
analytically -- pycocotools isn't available here, so the implementation is
validated by construction rather than by differential testing."""

from __future__ import annotations

import numpy as np
import torch

from instancedepth.utils.phase2_metrics import compute_mask_ap


def _mask(h, w, y0, y1, x0, x1):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_perfect_predictions_give_ap_one():
    """Predictions identical to GT -> IoU 1.0 at every threshold -> AP = 1."""
    gt = torch.stack([_mask(200, 200, 0, 100, 0, 100), _mask(200, 200, 120, 190, 120, 190)])
    ap = compute_mask_ap([dict(pred_masks=gt.clone(), scores=torch.tensor([0.9, 0.8]), gt_masks=gt)])
    assert abs(ap["AP"] - 1.0) < 1e-6
    assert abs(ap["AP50"] - 1.0) < 1e-6
    assert abs(ap["AP75"] - 1.0) < 1e-6


def test_no_predictions_give_ap_zero():
    gt = torch.stack([_mask(200, 200, 0, 100, 0, 100)])
    ap = compute_mask_ap([dict(pred_masks=torch.zeros(0, 200, 200, dtype=torch.bool),
                               scores=torch.zeros(0), gt_masks=gt)])
    assert ap["AP"] == 0.0


def test_ap_penalizes_false_positives():
    """One perfect detection + one spurious high-scoring FP must score below
    a clean single detection."""
    gt = torch.stack([_mask(200, 200, 0, 100, 0, 100)])
    clean = compute_mask_ap([dict(pred_masks=gt.clone(), scores=torch.tensor([0.9]), gt_masks=gt)])
    noisy_masks = torch.stack([gt[0], _mask(200, 200, 150, 190, 150, 190)])
    noisy = compute_mask_ap([dict(pred_masks=noisy_masks,
                                  scores=torch.tensor([0.9, 0.95]),   # FP ranked FIRST
                                  gt_masks=gt)])
    assert clean["AP"] == 1.0
    assert noisy["AP"] < clean["AP"]


def test_iou_threshold_sensitivity():
    """A prediction with IoU ~0.6 counts at AP50 but not at AP75 -- the exact
    behaviour that makes AP a stricter metric than fixed-threshold F1."""
    gt = torch.stack([_mask(100, 100, 0, 60, 0, 100)])        # 6000 px
    pred = torch.stack([_mask(100, 100, 0, 100, 0, 100)])     # 10000 px, intersection 6000
    # IoU = 6000/10000 = 0.6  -> matches at .5/.55/.6, not at .65+
    ap = compute_mask_ap([dict(pred_masks=pred, scores=torch.tensor([0.9]), gt_masks=gt)])
    assert abs(ap["AP50"] - 1.0) < 1e-6
    assert ap["AP75"] == 0.0
    assert 0.0 < ap["AP"] < 1.0


def test_ap_ranking_matters():
    """Same detections, better score ordering -> higher AP (AP is rank-based,
    unlike the project's fixed-threshold precision/recall)."""
    gt = torch.stack([_mask(200, 200, 0, 100, 0, 100), _mask(200, 200, 120, 190, 120, 190)])
    fp = _mask(200, 200, 0, 20, 150, 190)
    masks = torch.stack([gt[0], gt[1], fp])
    good = compute_mask_ap([dict(pred_masks=masks, scores=torch.tensor([0.9, 0.8, 0.1]), gt_masks=gt)])
    bad = compute_mask_ap([dict(pred_masks=masks, scores=torch.tensor([0.5, 0.4, 0.99]), gt_masks=gt)])
    assert good["AP"] > bad["AP"]


def test_area_bands_route_to_correct_metric():
    """A small GT (<32^2=1024 px) must populate APs and leave APl as NaN
    (no large positives exist -> that slice is undefined, not 0)."""
    small_gt = torch.stack([_mask(200, 200, 0, 20, 0, 20)])        # 400 px -> small
    ap = compute_mask_ap([dict(pred_masks=small_gt.clone(), scores=torch.tensor([0.9]),
                               gt_masks=small_gt)])
    assert abs(ap["APs"] - 1.0) < 1e-6
    assert ap["APl"] != ap["APl"]          # NaN: no large GT in this slice

    large_gt = torch.stack([_mask(200, 200, 0, 150, 0, 150)])      # 22500 px -> large
    ap2 = compute_mask_ap([dict(pred_masks=large_gt.clone(), scores=torch.tensor([0.9]),
                                gt_masks=large_gt)])
    assert abs(ap2["APl"] - 1.0) < 1e-6
    assert ap2["APs"] != ap2["APs"]


def test_multi_image_global_ranking():
    """AP accumulates detections across images by global score rank (COCO
    semantics), so a confident FP in image A can outrank a TP in image B."""
    g = _mask(200, 200, 0, 100, 0, 100)
    gt = torch.stack([g])
    fp = torch.stack([_mask(200, 200, 150, 190, 150, 190)])
    images = [
        dict(pred_masks=fp, scores=torch.tensor([0.99]), gt_masks=gt),        # confident FP
        dict(pred_masks=torch.stack([g]), scores=torch.tensor([0.5]), gt_masks=gt),   # quieter TP
    ]
    ap = compute_mask_ap(images)
    assert 0.0 < ap["AP50"] < 1.0     # recall 0.5 achievable, precision hurt by the FP
