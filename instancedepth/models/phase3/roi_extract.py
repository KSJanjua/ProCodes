"""ROIAlign feature extraction for occlusion pairs (paper Eq. 8 inputs).
Reconciles the two branches' *different* native
resolutions by working in normalized [0,1] box coordinates:

    F_obj  = ROIAlign(feat_final,  box)   -> (P,2,C,Hp,Wp)   depth-feature evidence   [Paper Specified]
    D_obj  = ROIAlign(depth_final, box)   -> (P,2,1,Hp,Wp)   Eq.9's base depth        [Paper Specified]
    G_obj  = [mask logits, norm coords, global depth]        geometric priors         [Paper Specified contents]

All ROIAligns use ``torchvision.ops.roi_align`` with ``spatial_scale=1.0``
after scaling the normalized box into each map's own pixel grid, so the same
box tensor aligns F_obj (Phase-1 feature res), D_obj (Phase-1 depth res),
mask logits (Phase-2 res) and the GT depth (Phase-2 res) into one shared
Hp x Wp normalized ROI frame.

Pair member ordering is [main, guest]: ``PairSet.boxes_norm`` (P,2,4) is
flattened to 2P rows as [p0_main, p0_guest, p1_main, p1_guest, ...] and
reshaped back to (P,2,...) on return.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.ops import roi_align

from instancedepth.models.phase3.candidates import PairSet


def _boxes_to_pixel_rois(boxes_norm_flat: torch.Tensor, batch_index_flat: torch.Tensor,
                         feat_hw: Tuple[int, int]) -> torch.Tensor:
    """(K,4) normalized xyxy + (K,) batch idx -> (K,5) rois in ``feat``'s
    pixel frame, ready for roi_align(spatial_scale=1.0)."""
    hf, wf = feat_hw
    scale = boxes_norm_flat.new_tensor([wf, hf, wf, hf])
    boxes_px = boxes_norm_flat * scale
    return torch.cat([batch_index_flat.to(boxes_px).unsqueeze(1), boxes_px], dim=1)


def roi_align_shared(feat: torch.Tensor, boxes_norm_flat: torch.Tensor,
                     batch_index_flat: torch.Tensor, out_hw: Tuple[int, int],
                     sampling_ratio: int) -> torch.Tensor:
    """ROIAlign a batch-shared feature map ``feat`` (B,C,H,W) at ``(K,4)``
    normalized boxes indexed by ``batch_index_flat`` -> (K,C,Hp,Wp)."""
    rois = _boxes_to_pixel_rois(boxes_norm_flat, batch_index_flat, feat.shape[-2:])
    return roi_align(feat, rois, output_size=out_hw, spatial_scale=1.0,
                     sampling_ratio=sampling_ratio, aligned=True)


def roi_align_per_instance(per_inst_maps: torch.Tensor, boxes_norm_flat: torch.Tensor,
                           out_hw: Tuple[int, int], sampling_ratio: int) -> torch.Tensor:
    """ROIAlign per-instance maps: ``per_inst_maps`` is (K,C,H,W) where row k
    already corresponds to box k (each 'image' is one instance's map), so the
    roi batch index is arange(K). Used for the per-query mask logits."""
    K = per_inst_maps.shape[0]
    idx = torch.arange(K, device=per_inst_maps.device)
    rois = _boxes_to_pixel_rois(boxes_norm_flat, idx, per_inst_maps.shape[-2:])
    return roi_align(per_inst_maps, rois, output_size=out_hw, spatial_scale=1.0,
                     sampling_ratio=sampling_ratio, aligned=True)


def normalized_coord_grid(boxes_norm_flat: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """(K,4) normalized xyxy -> (K,2,Hp,Wp) per-cell full-frame normalized
    (x,y). This is G_obj's "normalized coordinates" -- the geometric glue
    that lets Phi_o relate two independently-cropped ROIs."""
    K = boxes_norm_flat.shape[0]
    Hp, Wp = out_hw
    device, dtype = boxes_norm_flat.device, boxes_norm_flat.dtype
    x1, y1, x2, y2 = boxes_norm_flat.unbind(-1)                       # each (K,)
    jj = (torch.arange(Wp, device=device, dtype=dtype) + 0.5) / Wp    # (Wp,)
    ii = (torch.arange(Hp, device=device, dtype=dtype) + 0.5) / Hp    # (Hp,)
    xs = x1[:, None] + jj[None, :] * (x2 - x1)[:, None]               # (K,Wp)
    ys = y1[:, None] + ii[None, :] * (y2 - y1)[:, None]               # (K,Hp)
    grid_x = xs[:, None, :].expand(K, Hp, Wp)
    grid_y = ys[:, :, None].expand(K, Hp, Wp)
    return torch.stack([grid_x, grid_y], dim=1)                       # (K,2,Hp,Wp)


def extract_pair_roi_inputs(
    pairs: PairSet,
    feat_final: torch.Tensor,        # (B,C,hf,wf)  Phase-1 finest feature (F_2), or multi-scale concat
    depth_final: torch.Tensor,       # (B,1,H1,W1)  Phase-1 dense depth
    mask_logits: torch.Tensor,       # (B,N,H2,W2)  Phase-2 mask logits
    out_hw: Tuple[int, int],
    sampling_ratio: int,
    geom_coord: bool,
    geom_global_depth: bool,
    geom_mask_logit: bool,
):
    """Assemble F_obj, D_obj, and G_obj for all P pairs.

    Returns
    -------
    f_obj : (P,2,C,Hp,Wp)   ROIAligned depth features
    d_obj : (P,2,1,Hp,Wp)   ROIAligned base depth (Eq.9's D_obj; also the G_obj "global depth")
    g_obj : (P,2,Gc,Hp,Wp)  geometric priors (Gc = mask?1 + coord?2 + gdepth?1)
    roi_mask_logit : (P,2,1,Hp,Wp)  ROIAligned per-member mask logits -- always
        returned (independent of geom_mask_logit) because the scalar depth
        reduction for L_dist / refined layers needs a per-instance ROI weight.
    """
    P = len(pairs)
    Hp, Wp = out_hw
    boxes_flat = pairs.boxes_norm.reshape(2 * P, 4)                   # [main,guest] interleaved
    bidx_flat = pairs.batch_index.repeat_interleave(2)               # (2P,)

    # F_obj -- batch-shared feature map
    f_flat = roi_align_shared(feat_final, boxes_flat, bidx_flat, out_hw, sampling_ratio)
    f_obj = f_flat.reshape(P, 2, *f_flat.shape[1:])                  # (P,2,C,Hp,Wp)

    # D_obj / global depth -- batch-shared depth map
    d_flat = roi_align_shared(depth_final, boxes_flat, bidx_flat, out_hw, sampling_ratio)
    d_obj = d_flat.reshape(P, 2, 1, Hp, Wp)                          # (P,2,1,Hp,Wp)

    # per-member mask logits (always computed; used both for G_obj and the
    # scalar-depth reduction weight downstream)
    q_flat = pairs.query_idx.reshape(2 * P)
    per_inst = mask_logits[bidx_flat, q_flat].unsqueeze(1)          # (2P,1,H2,W2)
    m_flat = roi_align_per_instance(per_inst, boxes_flat, out_hw, sampling_ratio)
    roi_mask_logit = m_flat.reshape(P, 2, 1, Hp, Wp)

    geom_channels = []
    if geom_mask_logit:
        geom_channels.append(roi_mask_logit)
    if geom_coord:
        c_flat = normalized_coord_grid(boxes_flat, out_hw)          # (2P,2,Hp,Wp)
        geom_channels.append(c_flat.reshape(P, 2, 2, Hp, Wp))
    if geom_global_depth:
        geom_channels.append(d_obj)                                 # reuse ROIAligned depth

    g_obj = torch.cat(geom_channels, dim=2) if geom_channels else \
        feat_final.new_zeros((P, 2, 0, Hp, Wp))
    return f_obj, d_obj, g_obj, roi_mask_logit
