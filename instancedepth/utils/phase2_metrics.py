"""Phase 2 evaluation metrics.

Two suites, reported side by side:

1. **Project metrics** -- greedy best-IoU matching at a fixed IoU threshold:
   precision / recall / F1 / mean IoU, plus depth-layer MAE on matched pairs
   (the depth layer is this paper's addition, so no standard metric covers it).
2. **COCO-style mask AP** (``compute_mask_ap``) -- AP, AP50, AP75, APs, APm,
   APl: exactly what Mask2Former reports, giving externally comparable numbers.
   Implemented standalone against the COCO protocol rather than depending on
   ``pycocotools`` (absent here, and it would need a COCO-format conversion
   step) -- the same "reimplement the well-known algorithm rather than vendor
   a heavy dependency" choice already made for ``point_sample.py``.

The **occlusion-focused slice** (frames with >=2 overlapping GT instances)
is the primary metric this redesign is judged against --
implemented here as a boolean mask over frames, applied by the caller.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


@torch.no_grad()
def mask_quality_scores(mask_prob: torch.Tensor, binarize_thresh: float = 0.5) -> torch.Tensor:
    """(B,N,H,W) sigmoid probs -> (B,N) mask confidence = mean foreground
    probability over the binarized mask -- Mask2Former's own mask-quality
    term (its ``instance_inference`` multiplies it with class confidence to
    form the final instance score). Lives here rather than in the Phase-3
    candidate code because it is a Phase-2/Mask2Former concept that both the
    Phase-2 evaluator and Phase 3's candidate filter consume."""
    fg = (mask_prob >= binarize_thresh).float()
    num = (mask_prob * fg).flatten(2).sum(-1)
    den = fg.flatten(2).sum(-1).clamp_min(1.0)
    return num / den


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


# --------------------------------------------------------------------------- #
# COCO-style mask AP (the metric Mask2Former itself reports)
# --------------------------------------------------------------------------- #
# COCO protocol constants (cocoeval.py's Params for iouType='segm').
_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)          # .50:.05:.95
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)           # 101-point interpolation
_AREA_RANGES: Dict[str, Tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}


def _match_image(iou: np.ndarray, gt_ignore: np.ndarray, thr: float) -> np.ndarray:
    """COCO's per-image greedy matcher (cocoeval.evaluateImg) at one IoU
    threshold. ``iou`` is (D,G) with detections already sorted by descending
    score and GT sorted non-ignored-first. Returns (D,) matched GT index, or
    -1 for unmatched -- the caller derives TP/FP/ignore from it."""
    D, G = iou.shape
    gt_matched = np.full(G, False)
    dt_match = np.full(D, -1, dtype=int)
    for d in range(D):
        best_iou, best_g = thr, -1
        for g in range(G):
            if gt_matched[g] and not gt_ignore[g]:
                continue                      # real GT already taken
            if best_g > -1 and not gt_ignore[best_g] and gt_ignore[g]:
                break                         # had a real match; rest are ignored GT
            if iou[d, g] < best_iou:
                continue
            best_iou, best_g = iou[d, g], g
        if best_g == -1:
            continue
        dt_match[d] = best_g
        gt_matched[best_g] = True
    return dt_match


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """COCO's 101-point interpolated AP from score-sorted TP/FP flags."""
    if n_gt == 0:
        return float("nan")                   # no positives -> AP undefined for this slice
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rc = tp_cum / n_gt
    pr = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    # make precision monotonically non-increasing (COCO smooths right-to-left)
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    idx = np.searchsorted(rc, _RECALL_POINTS, side="left")
    q = np.zeros(len(_RECALL_POINTS))
    for ri, pi in enumerate(idx):
        if pi < len(pr):
            q[ri] = pr[pi]
    return float(q.mean())


@torch.no_grad()
def compute_mask_ap(images: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """COCO-style mask AP over a list of per-image predictions.

    Each item: ``dict(pred_masks=(D,H,W) bool, scores=(D,), gt_masks=(G,H,W) bool)``.
    Detections must NOT be score-thresholded -- AP ranks them by score, so a
    cut would silently truncate the precision-recall curve. ``scores`` should
    be Mask2Former's own instance score (class confidence x mask quality).

    Returns AP (mean over IoU .50:.05:.95), AP50, AP75, and APs/APm/APl
    (GT area bands 32^2 / 96^2, COCO's ignore semantics: detections matched to
    an out-of-band GT are ignored, and unmatched detections count as FP only
    if their own area falls in the band).
    """
    per_image = []
    for im in images:
        pm, gm = im["pred_masks"], im["gt_masks"]
        scores = im["scores"].detach().float().cpu().numpy() if len(im["scores"]) else np.zeros(0)
        order = np.argsort(-scores, kind="mergesort")     # stable, descending
        iou = mask_iou_matrix(pm.bool(), gm.bool()).detach().float().cpu().numpy() if len(pm) and len(gm) \
            else np.zeros((len(pm), len(gm)))
        per_image.append(dict(
            scores=scores[order],
            iou=iou[order] if len(pm) else iou,
            dt_area=(pm.flatten(1).sum(1).detach().float().cpu().numpy()[order] if len(pm) else np.zeros(0)),
            gt_area=(gm.flatten(1).sum(1).detach().float().cpu().numpy() if len(gm) else np.zeros(0)),
        ))

    out: Dict[str, float] = {}
    ap_per_range: Dict[str, List[float]] = {k: [] for k in _AREA_RANGES}
    ap50 = ap75 = None

    for rng_name, (lo, hi) in _AREA_RANGES.items():
        for thr in _IOU_THRESHOLDS:
            all_scores, all_tp, all_fp = [], [], []
            n_gt = 0
            for im in per_image:
                gt_ig = (im["gt_area"] < lo) | (im["gt_area"] >= hi)
                n_gt += int((~gt_ig).sum())
                D = len(im["scores"])
                if D == 0:
                    continue
                # COCO sorts GT non-ignored-first so the matcher prefers real GT
                gorder = np.argsort(gt_ig, kind="mergesort")
                iou_s = im["iou"][:, gorder] if im["iou"].size else im["iou"]
                gt_ig_s = gt_ig[gorder]
                m = _match_image(iou_s, gt_ig_s, float(thr)) if iou_s.size else np.full(D, -1, dtype=int)

                matched = m > -1
                dt_ig = np.where(matched, gt_ig_s[np.clip(m, 0, None)], False)
                # unmatched detections outside the area band are ignored, not FP
                dt_ig |= ~matched & ((im["dt_area"] < lo) | (im["dt_area"] >= hi))
                all_scores.append(im["scores"])
                all_tp.append(matched & ~dt_ig)
                all_fp.append(~matched & ~dt_ig)

            if all_scores:
                s = np.concatenate(all_scores)
                order = np.argsort(-s, kind="mergesort")   # global rank ACROSS images (COCO)
                tp = np.concatenate(all_tp)[order].astype(np.float64)
                fp = np.concatenate(all_fp)[order].astype(np.float64)
            else:
                # No detections: recall 0 -> AP 0. (_average_precision still
                # returns NaN if there were no positives to find either.)
                tp = fp = np.zeros(0, dtype=np.float64)
            ap = _average_precision(tp, fp, n_gt)
            ap_per_range[rng_name].append(ap)
            if rng_name == "all" and abs(thr - 0.50) < 1e-9:
                ap50 = ap
            if rng_name == "all" and abs(thr - 0.75) < 1e-9:
                ap75 = ap

    def _mean(vals: List[float]) -> float:
        keep = [v for v in vals if v == v]                # drop NaN slices
        return float(np.mean(keep)) if keep else float("nan")

    out["AP"] = _mean(ap_per_range["all"])
    out["AP50"] = ap50 if ap50 is not None else float("nan")
    out["AP75"] = ap75 if ap75 is not None else float("nan")
    out["APs"] = _mean(ap_per_range["small"])
    out["APm"] = _mean(ap_per_range["medium"])
    out["APl"] = _mean(ap_per_range["large"])
    return out
