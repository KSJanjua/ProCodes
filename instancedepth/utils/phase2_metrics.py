"""Phase 2 evaluation metrics (Phase 2 plan, SS6 step 7).

Deliberately a lighter-weight suite than full COCO-style AP (which would
pull in `pycocotools` and a COCO-format conversion step for comparatively
little benefit at this project's stage): greedy best-IoU matching between
predictions and GT per image, mask IoU / precision / recall at a fixed IoU
threshold, and depth-layer MAE on matched pairs. If COCO AP is wanted later
for external comparability, it can be added without touching this module's
callers (`evaluate_phase2.py`).

The **occlusion-focused slice** (frames with >=2 overlapping GT instances)
is the primary metric this redesign is judged against (plan SS2.3/SS6) --
implemented here as a boolean mask over frames, applied by the caller.
"""

from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def mask_iou_matrix(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """pred_masks: (P,H,W) bool, gt_masks: (G,H,W) bool -> (P,G) IoU."""
    if pred_masks.numel() == 0 or gt_masks.numel() == 0:
        return torch.zeros(pred_masks.shape[0], gt_masks.shape[0])
    pred_flat = pred_masks.flatten(1).float()
    gt_flat = gt_masks.flatten(1).float()
    inter = pred_flat @ gt_flat.T
    union = pred_flat.sum(1)[:, None] + gt_flat.sum(1)[None, :] - inter
    return inter / union.clamp_min(1e-6)


@torch.no_grad()
def _mask_bboxes(masks: torch.Tensor) -> torch.Tensor:
    """(K,H,W) bool -> (K,4) [x1,y1,x2,y2] float; empty masks -> zeros."""
    k = masks.shape[0]
    boxes = torch.zeros(k, 4, dtype=torch.float32)
    for i in range(k):
        ys, xs = torch.where(masks[i])
        if xs.numel() == 0:
            continue
        boxes[i] = torch.tensor(
            [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]
        )
    return boxes


def _box_iomin(a: torch.Tensor, b: torch.Tensor) -> float:
    """Intersection-over-min-area of two [x1,y1,x2,y2] boxes (1.0 when the
    smaller box is fully inside the larger -- the right measure for a small
    instance occluded by a large one, where plain IoU would be tiny)."""
    ix1, iy1 = torch.maximum(a[0], b[0]), torch.maximum(a[1], b[1])
    ix2, iy2 = torch.minimum(a[2], b[2]), torch.minimum(a[3], b[3])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    m = torch.minimum(area_a, area_b)
    return float(inter / m) if float(m) > 0 else 0.0


@torch.no_grad()
def has_overlapping_instances(gt_masks: torch.Tensor, min_box_iomin: float = 0.1) -> bool:
    """True if any two GT instances plausibly form an occlusion/contact pair.

    IMPORTANT: GT instance masks in this dataset are *modal and disjoint* --
    every pixel belongs to exactly one instance (via the flattened id-map),
    so two instances have ~0 mask-IoU even under heavy occlusion. Occlusion
    therefore does NOT show up as overlapping masks; it shows up as spatially
    interleaved image regions, i.e. **overlapping bounding boxes**. We detect
    it via pairwise box intersection-over-min-area (``_box_iomin``), which
    fires when one instance's box substantially overlaps another's -- exactly
    the occluder/occludee geometry the paper's occlusion slice cares about.

    (An earlier version tested mask IoU > 0 here, which was structurally
    guaranteed to return False for disjoint masks -- so the occlusion slice
    was always empty. See git history.)"""
    k = gt_masks.shape[0]
    if k < 2:
        return False
    boxes = _mask_bboxes(gt_masks.bool())
    for i in range(k):
        for j in range(i + 1, k):
            if _box_iomin(boxes[i], boxes[j]) > min_box_iomin:
                return True
    return False


@torch.no_grad()
def evaluate_frame(
    pred_masks: torch.Tensor,      # (P,H,W) bool, already score-thresholded
    pred_scores: torch.Tensor,     # (P,)
    pred_depths: torch.Tensor,     # (P,)
    gt_masks: torch.Tensor,        # (G,H,W) bool
    gt_depths: torch.Tensor,       # (G,)
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Greedy best-IoU matching (highest-scoring prediction first), then
    precision/recall/mean-IoU/depth-MAE on the matched set."""
    P, G = pred_masks.shape[0], gt_masks.shape[0]
    if G == 0:
        return dict(tp=0, fp=P, fn=0, mean_iou=float("nan"), depth_mae=float("nan"))
    if P == 0:
        return dict(tp=0, fp=0, fn=G, mean_iou=float("nan"), depth_mae=float("nan"))

    iou = mask_iou_matrix(pred_masks, gt_masks)   # (P, G)
    order = torch.argsort(pred_scores, descending=True)

    matched_gt = torch.zeros(G, dtype=torch.bool)
    tp, ious, depth_errors = 0, [], []
    for p in order.tolist():
        if matched_gt.all():
            break
        candidate_iou = iou[p].clone()
        candidate_iou[matched_gt] = -1.0
        best_g = int(candidate_iou.argmax())
        if candidate_iou[best_g] >= iou_threshold:
            matched_gt[best_g] = True
            tp += 1
            ious.append(float(candidate_iou[best_g]))
            depth_errors.append(float((pred_depths[p] - gt_depths[best_g]).abs()))

    fp = P - tp
    fn = G - tp
    return dict(
        tp=tp, fp=fp, fn=fn,
        mean_iou=float(sum(ious) / len(ious)) if ious else float("nan"),
        depth_mae=float(sum(depth_errors) / len(depth_errors)) if depth_errors else float("nan"),
    )


def aggregate(frame_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    total_tp = sum(m["tp"] for m in frame_metrics)
    total_fp = sum(m["fp"] for m in frame_metrics)
    total_fn = sum(m["fn"] for m in frame_metrics)
    ious = [m["mean_iou"] for m in frame_metrics if m["mean_iou"] == m["mean_iou"]]        # drop NaN
    depth_maes = [m["depth_mae"] for m in frame_metrics if m["depth_mae"] == m["depth_mae"]]

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    return dict(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-6),
        mean_iou=sum(ious) / len(ious) if ious else float("nan"),
        depth_mae=sum(depth_maes) / len(depth_maes) if depth_maes else float("nan"),
        num_frames=len(frame_metrics),
    )
