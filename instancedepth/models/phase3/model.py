"""Phase3Model -- composes the (fine-tuned) Phase-1 depth branch and the
(frozen) Phase-2 instance decoder, adds the Occlusion Pair Relation Reasoning
head Phi_o, and produces occlusion-corrected depth (paper Sec. 4.2.2 / Eq.
8-12). See ``docs/PHASE3_DESIGN.md``.

Freeze/train split (paper Sec. 4.3, "Occlusion-Aware Joint Refinement"):
  * Phase 1 (DINOv2 + Depth Range Decoder + Eq.1-4)  : TRAINABLE  (LR 1e-6)
  * Phase 2 (Swin-L Mask2Former + DepthLayerHead)    : FROZEN     (no grad)
  * Phi_o                                            : TRAINABLE  (fresh)

Resolution: each branch runs at its own trained resolution;
ROIAlign reconciles them via normalized boxes. ``forward`` receives RGB at
the Phase-2 frame (where GT masks/boxes align) and internally resizes for
the depth branch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.models.phase2.model import Phase2Model
from instancedepth.models.phase3.candidates import build_pairs
from instancedepth.models.phase3.output import RefinedDepthOutput
from instancedepth.models.phase3.relation_head import (
    OcclusionRelationHead, composite_refined_depth, roi_masked_mean,
)
from instancedepth.models.phase3.roi_extract import extract_pair_roi_inputs
from instancedepth.utils.checkpoint import load_checkpoint

log = logging.getLogger("instancedepth.models.phase3.model")


def _geom_channel_count(cfg) -> int:
    return (int(cfg.geom_mask_logit) * 1
            + int(cfg.geom_coord) * 2
            + int(cfg.geom_global_depth) * 1)


class Phase3Model(nn.Module):
    def __init__(self, cfg: Phase3Config) -> None:
        super().__init__()
        self.config = cfg
        self.phase1 = HolisticDepthModel(cfg.phase1)
        self.phase2 = Phase2Model(
            checkpoint=cfg.phase2.model.checkpoint,
            checkpoint_dir=cfg.phase2.model.checkpoint_dir,
            allow_hub_download=cfg.phase2.model.allow_hub_download,
            num_classes=cfg.phase2.model.num_classes,
        )

        feat_c = cfg.phase1.decoder.channels_attn
        if cfg.head.use_multiscale_feat:
            feat_c *= 3   # F_0 + F_1 + F_2 concatenated
        per_member = feat_c + _geom_channel_count(cfg.head)
        self.relation_head = OcclusionRelationHead(
            per_member_channels=per_member,
            hidden_dim=cfg.head.hidden_dim,
            num_conv=cfg.head.num_conv,
            granularity=cfg.head.refine_granularity,
        )
        self._feat_channels = feat_c

        if cfg.phase1_checkpoint:
            load_checkpoint(Path(cfg.phase1_checkpoint), self.phase1, restore_rng=False)
            log.info("loaded Phase 1 weights from %s", cfg.phase1_checkpoint)
        else:
            log.warning("no phase1_checkpoint -- Phase 1 starts from its own init (smoke test only)")
        if cfg.phase2_checkpoint:
            load_checkpoint(Path(cfg.phase2_checkpoint), self.phase2, restore_rng=False)
            log.info("loaded Phase 2 weights from %s", cfg.phase2_checkpoint)
        else:
            log.warning("no phase2_checkpoint -- Phase 2 uses COCO weights only (smoke test only)")

        self._freeze_phase2()
        if getattr(cfg, "freeze_phase1", False):
            self._freeze_phase1()

    # ------------------------------------------------------------------ #
    def _freeze_phase1(self) -> None:
        """Pin the Phase-1 depth branch (paper fine-tunes it at 1e-6; on this
        single-sensor, ROI-only-supervised data that fine-tune drifts the
        dense depth 0.078 -> 0.139 abs_rel -- docs/AUDIT_2026.md). Frozen, the
        dense base stays at Phase-1 quality and refinement is non-degrading by
        construction."""
        for p in self.phase1.parameters():
            p.requires_grad_(False)
        self.phase1.eval()

    def reset_temporal_state(self) -> None:
        """Passthrough to Phase 1's temporal memory (no-op for per-frame
        Phase-1 checkpoints). Call at sequence boundaries when streaming."""
        if hasattr(self.phase1, "reset_temporal_state"):
            self.phase1.reset_temporal_state()

    def _freeze_phase2(self) -> None:
        for p in self.phase2.parameters():
            p.requires_grad_(False)
        self.phase2.eval()

    def train(self, mode: bool = True) -> "Phase3Model":
        """Keep the frozen instance decoder in eval() regardless of the
        module's train/eval state (paper Sec. 4.3: instance decoder fixed)."""
        super().train(mode)
        self.phase2.eval()
        if getattr(self.config, "freeze_phase1", False):
            self.phase1.eval()   # keep the pinned depth branch deterministic
        return self

    # ------------------------------------------------------------------ #
    def _build_feat_map(self, p1out) -> torch.Tensor:
        """F_obj source: F_2 alone (faithful) or F_0/F_1/F_2 concatenated
        (multi-scale). Multi-scale needs contract v1.2 feat_levels."""
        if not self.config.head.use_multiscale_feat:
            return p1out.feat_final
        assert p1out.feat_levels is not None, (
            "head.use_multiscale_feat=True requires HolisticDepthOutput v1.2 "
            "feat_levels; got None"
        )
        target_hw = p1out.feat_final.shape[-2:]
        ups = [F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
               if f.shape[-2:] != target_hw else f for f in p1out.feat_levels]
        return torch.cat(ups, dim=1)

    def forward(self, image: torch.Tensor) -> Tuple[RefinedDepthOutput, Dict]:
        """``image`` : (B,3,H2,W2) at the Phase-2 frame (== cfg.data.image_size).

        Returns (RefinedDepthOutput, aux) where aux carries the frozen
        Phase-2 output + pair set needed by the training criterion (matcher +
        dense-GT extraction). At inference the caller ignores aux.
        """
        H2, W2 = image.shape[-2:]
        cfg = self.config

        # --- Phase 1 (trainable), at its own resolution ---------------------
        p1_hw = tuple(cfg.phase1.data.image_size)
        image_p1 = image if (H2, W2) == p1_hw else \
            F.interpolate(image, size=p1_hw, mode="bilinear", align_corners=False)
        p1out = self.phase1(image_p1)
        feat_map = self._build_feat_map(p1out)          # (B,C,hf,wf)
        depth_p1 = p1out.depth_final                    # (B,1,*p1_hw)

        # --- Phase 2 (frozen) ----------------------------------------------
        with torch.no_grad():
            p2 = self.phase2(image)                     # masks/classes/Dep at (H2,W2)

        # --- candidate + pair construction ---------------------------------
        pairs = build_pairs(p2, cfg.candidate)

        out_hw = tuple(cfg.head.roi_size)
        if len(pairs) == 0:
            # no confident overlapping pairs -> refined == base (Phase-1 depth)
            base_p2 = F.interpolate(depth_p1, size=(H2, W2), mode="bilinear", align_corners=False)
            empty = pairs
            zeros = depth_p1.new_zeros((0, 2, 1, *out_hw))
            output = RefinedDepthOutput(
                refined_depth=base_p2, base_depth=base_p2, image_hw=(H2, W2),
                pair_batch_index=empty.batch_index, pair_query_idx=empty.query_idx,
                pair_iou=empty.iou, refined_layers=depth_p1.new_zeros((0, 2)),
                base_layers=depth_p1.new_zeros((0, 2)),
                d_hat_roi=zeros, e_obj_roi=zeros, d_obj_roi=zeros,
            )
            return output, {"pairs": pairs, "p2": p2, "depth_p1": depth_p1}

        # --- ROIAlign feature extraction (normalized boxes) ----------------
        f_obj, d_obj, g_obj, roi_mask_logit = extract_pair_roi_inputs(
            pairs, feat_map, depth_p1, p2.mask_logits, out_hw,
            cfg.head.roi_sampling_ratio,
            geom_coord=cfg.head.geom_coord,
            geom_global_depth=cfg.head.geom_global_depth,
            geom_mask_logit=cfg.head.geom_mask_logit,
        )

        # --- Phi_o relation reasoning + Eq. 9 ------------------------------
        e_obj, d_hat = self.relation_head(f_obj, g_obj, d_obj)   # (P,2,1,Hp,Wp)

        # --- scalar reductions for L_dist / reported layers ----------------
        roi_w = (roi_mask_logit.sigmoid() >= cfg.candidate.mask_binarize_thresh).float()
        refined_layers = roi_masked_mean(d_hat, roi_w)          # (P,2)
        base_layers = roi_masked_mean(d_obj, roi_w)             # (P,2)

        # --- composite the correction RATIO into the dense map (P-2 frame) ---
        # (eval-only artifact; the loss trains d_hat directly. See
        # relation_head.composite_refined_depth for the ratio semantics.)
        base_p2 = F.interpolate(depth_p1, size=(H2, W2), mode="bilinear", align_corners=False)
        mask_prob = p2.mask_logits.sigmoid()
        refined_depth = composite_refined_depth(
            base_p2, pairs, e_obj, refined_layers, mask_prob,
            cfg.candidate.mask_binarize_thresh, ratio_mode=cfg.head.composite_ratio,
            feather_px=cfg.head.composite_feather_px,
        )

        output = RefinedDepthOutput(
            refined_depth=refined_depth, base_depth=base_p2, image_hw=(H2, W2),
            pair_batch_index=pairs.batch_index, pair_query_idx=pairs.query_idx,
            pair_iou=pairs.iou, refined_layers=refined_layers, base_layers=base_layers,
            d_hat_roi=d_hat, e_obj_roi=e_obj, d_obj_roi=d_obj,
        )
        aux = {"pairs": pairs, "p2": p2, "roi_mask_weight": roi_w, "depth_p1": depth_p1}
        return output, aux
