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
def has_overlapping_instances(gt_masks: torch.Tensor, iou_threshold: float = 0.0) -> bool:
    """True if any two GT instances in this frame overlap at all (used to
    build the occlusion-focused eval slice)."""
    if gt_masks.shape[0] < 2:
        return False
    iou = mask_iou_matrix(gt_masks.bool(), gt_masks.bool())
    iou.fill_diagonal_(0.0)
    return bool((iou > iou_threshold).any())


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
