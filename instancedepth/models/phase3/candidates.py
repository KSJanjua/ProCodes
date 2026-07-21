"""Candidate filtering + occlusion-pair construction (paper Sec. 4.2.2, the
front-end of "Occlusion Pair Relation Reasoning").

Reads the frozen Phase-2 predictions and produces, per batch, a flat set of
occlusion pairs ready for ROIAlign:

    1. Filter queries: category confidence > 0.9 AND mask confidence > 0.8.   [Paper Specified]
    2. Among survivors, form overlapping pairs (IoU > 0.1) and keep, per
       instance, its depth-nearest partner.                                  [Paper Specified rule]
    3. Deduplicate to unordered pairs (member order nearer-first) and flatten
       across the batch with a batch_index.

Overlap uses bounding-box IoU by default, not mask IoU: this dataset's GT
(and hence Phase-2's predicted) masks are modal and disjoint, so occluding
instances have ~0 mask IoU regardless of confidence -- see
``Phase3CandidateConfig.overlap_metric``.
Deduplication prevents the same instance being refined twice
with disagreeing correction fields.

Everything here is non-differentiable (thresholds / argmin over frozen
Phase-2 predictions), so it runs under no_grad on detached tensors.

The scalar "mask confidence" reduction (paper says only "mask confidence
> 0.8", not how a dense HxW map becomes one scalar) uses the standard
Mask2Former mask-quality score: mean sigmoid over the binarized foreground.
[Reasonable Assumption]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from instancedepth.configs.phase3_config import Phase3CandidateConfig
from instancedepth.models.phase2.output import Phase2Output
from instancedepth.utils.phase2_metrics import mask_quality_scores


@dataclass
class PairSet:
    """Occlusion pairs flattened across the batch (P total).

    Pairs are UNORDERED-DEDUPLICATED (see build_pairs): {A,B} appears once, never as both (A,B) and
    (B,A). Member order is canonical: index 0 = the nearer instance (smaller
    predicted Dep, i.e. the occluder), index 1 = the farther one. This gives
    Phi_o a consistent channel semantic and prevents the same person being
    refined twice with disagreeing fields. [Reasonable Assumption]
    """

    batch_index: torch.Tensor   # (P,) long -- which image
    query_idx: torch.Tensor     # (P,2) long -- [nearer/occluder, farther] query indices
    boxes_norm: torch.Tensor    # (P,2,4) float -- normalized [0,1] xyxy, same member order
    iou: torch.Tensor           # (P,) float -- pair overlap IoU (symmetric)

    def __len__(self) -> int:
        return int(self.batch_index.shape[0])

    def to(self, device) -> "PairSet":
        return PairSet(
            self.batch_index.to(device), self.query_idx.to(device),
            self.boxes_norm.to(device), self.iou.to(device),
        )

    @staticmethod
    def empty(device) -> "PairSet":
        return PairSet(
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, 2, dtype=torch.long, device=device),
            torch.zeros(0, 2, 4, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
        )


def boxes_and_masks_from_probs(mask_prob: torch.Tensor, binarize_thresh: float):
    """(B,N,H,W) -> (binary (B,N,H,W) bool, boxes (B,N,4) normalized xyxy,
    valid (B,N) bool). Boxes are normalized to [0,1] by (W,H) so downstream
    ROIAlign is resolution-independent. Empty masks -> valid=False,
    box=zeros."""
    B, N, H, W = mask_prob.shape
    binary = mask_prob >= binarize_thresh
    boxes = mask_prob.new_zeros((B, N, 4))
    valid = torch.zeros((B, N), dtype=torch.bool, device=mask_prob.device)
    for b in range(B):
        for n in range(N):
            ys, xs = torch.where(binary[b, n])
            if xs.numel() == 0:
                continue
            x1, x2 = xs.min().float(), xs.max().float() + 1.0
            y1, y2 = ys.min().float(), ys.max().float() + 1.0
            boxes[b, n] = torch.stack([x1 / W, y1 / H, x2 / W, y2 / H])
            valid[b, n] = True
    return binary, boxes, valid


def _pairwise_mask_iou(binary_masks: torch.Tensor) -> torch.Tensor:
    """(K,H,W) bool -> (K,K) IoU. Vectorized over the flattened masks.

    Structurally near-zero for genuinely occluded pairs on this dataset (see
    Phase3CandidateConfig.overlap_metric's docstring) -- kept only as the
    configurable "mask_iou" alternative, not the default."""
    m = binary_masks.flatten(1).float()                      # (K, H*W)
    inter = m @ m.t()                                        # (K, K)
    area = m.sum(-1)                                         # (K,)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp_min(1.0)


def _pairwise_box_iou(boxes_norm: torch.Tensor) -> torch.Tensor:
    """(K,4) normalized xyxy -> (K,K) IoU. Default overlap metric (see
    Phase3CandidateConfig.overlap_metric's docstring): unlike predicted mask
    IoU, box IoU correctly detects two occluding people even though their
    (modal, disjoint-by-training) predicted masks don't overlap."""
    x1, y1, x2, y2 = boxes_norm.unbind(-1)                    # each (K,)
    area = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)    # (K,)
    ix1 = torch.max(x1[:, None], x1[None, :])
    iy1 = torch.max(y1[:, None], y1[None, :])
    ix2 = torch.min(x2[:, None], x2[None, :])
    iy2 = torch.min(y2[:, None], y2[None, :])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp_min(1e-8)


@torch.no_grad()
def build_pairs(p2: Phase2Output, cfg: Phase3CandidateConfig) -> PairSet:
    """Filter candidates and build occlusion pairs for a whole batch."""
    device = p2.mask_logits.device
    cat_conf = p2.scores()                                   # (B,N)
    mask_prob = p2.mask_logits.sigmoid()                     # (B,N,H,W)
    mask_conf = mask_quality_scores(mask_prob, cfg.mask_binarize_thresh)  # (B,N)
    binary, boxes, box_valid = boxes_and_masks_from_probs(mask_prob, cfg.mask_binarize_thresh)
    dep = p2.depth_layers                                    # (B,N)

    B, N = cat_conf.shape
    keep = (cat_conf > cfg.cat_conf_thresh) & (mask_conf > cfg.mask_conf_thresh) & box_valid

    all_bidx: List[int] = []
    all_pairs: List[torch.Tensor] = []
    all_boxes: List[torch.Tensor] = []
    all_iou: List[float] = []

    for b in range(B):
        cand = torch.where(keep[b])[0]                       # candidate query indices
        if cand.numel() < 2:
            continue
        if cand.numel() > cfg.max_candidates:
            # keep the highest category-confidence candidates
            top = torch.topk(cat_conf[b, cand], cfg.max_candidates).indices
            cand = cand[top]

        if cfg.overlap_metric == "box_iou":
            iou = _pairwise_box_iou(boxes[b, cand])          # (C, C)
        else:
            iou = _pairwise_mask_iou(binary[b, cand])        # (C, C)
        iou.fill_diagonal_(0.0)
        dep_c = dep[b, cand]                                 # (C,)

        # Deduplicate to UNORDERED pairs:
        # the paper's directional phrasing ("for each main ... retain the
        # nearest guest") yields both (A,B) and (B,A) for a mutually
        # overlapping pair, which wrote the same person twice with two
        # disagreeing correction fields. Keep one entry per unordered pair,
        # member order canonicalized nearer-first (occluder = channel 0).
        chosen: dict = {}   # (min_q, max_q) -> ((q_near, q_far), iou)
        for i in range(cand.numel()):
            overlaps = torch.where(iou[i] > cfg.overlap_iou_thresh)[0]
            if overlaps.numel() == 0:
                continue
            if cfg.guest_rule == "nearest_depth":
                score = (dep_c[overlaps] - dep_c[i]).abs()
            else:  # "frontmost": smallest depth among overlappers
                score = dep_c[overlaps]
            guest_local = overlaps[torch.argmin(score)]

            q_i, q_j = int(cand[i]), int(cand[guest_local])
            key = (min(q_i, q_j), max(q_i, q_j))
            if key in chosen:
                continue
            if float(dep[b, q_i]) <= float(dep[b, q_j]):
                ordered = (q_i, q_j)
            else:
                ordered = (q_j, q_i)
            chosen[key] = (ordered, float(iou[i, guest_local]))

        for (q_near, q_far), pair_iou in chosen.values():
            all_bidx.append(b)
            all_pairs.append(torch.tensor([q_near, q_far], dtype=torch.long))
            all_boxes.append(torch.stack([boxes[b, q_near], boxes[b, q_far]]))   # (2,4)
            all_iou.append(pair_iou)

    if not all_bidx:
        return PairSet.empty(device)

    return PairSet(
        batch_index=torch.tensor(all_bidx, dtype=torch.long, device=device),
        query_idx=torch.stack(all_pairs).to(device),
        boxes_norm=torch.stack(all_boxes).to(device),
        iou=torch.tensor(all_iou, dtype=torch.float32, device=device),
    )
