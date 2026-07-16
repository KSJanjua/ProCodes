"""Unit tests for the Phase-3-specific logic (paper Eq. 8-12), using
synthetic tensors so the DINOv2 / Swin-L backbones (and their server-only
weights) are never needed. Covers the real correctness risk: pair building,
normalized-box ROIAlign shapes, Eq. 9 arithmetic, compositing (nearest-wins),
dense-GT ROI extraction, and the refinement losses.
"""

from __future__ import annotations

import torch

from instancedepth.configs.phase3_config import Phase3CandidateConfig, Phase3HeadConfig, Phase3LossConfig
from instancedepth.models.phase2.output import Phase2Output
from instancedepth.models.phase3.candidates import build_pairs
from instancedepth.models.phase3.output import RefinedDepthOutput
from instancedepth.models.phase3.relation_head import (
    OcclusionRelationHead, composite_refined_depth, roi_masked_mean,
)
from instancedepth.models.phase3.roi_extract import extract_pair_roi_inputs
from instancedepth.models.phase3.targets import build_dense_gt_rois, RefineTargets
from instancedepth.losses.phase3_losses import Phase3Criterion


def _blob(mask_logits, b, n, r0, r1, c0, c1, val=10.0):
    mask_logits[b, n, r0:r1, c0:c1] = val


def _synthetic_p2(H=32, W=32):
    """B=1, N=3: q0 & q1 overlap (IoU>0.1) at near depths; q2 isolated."""
    B, N = 1, 3
    mask_logits = torch.full((B, N, H, W), -10.0)
    _blob(mask_logits, 0, 0, 0, 16, 0, 16)      # q0 top-left
    _blob(mask_logits, 0, 1, 8, 24, 8, 24)      # q1 overlaps q0
    _blob(mask_logits, 0, 2, 0, 16, 24, 32)     # q2 isolated (cols 24:32)
    class_logits = torch.zeros(B, N, 2)
    class_logits[..., 0] = 3.0                  # person logit -> softmax ~0.95
    depth_layers = torch.tensor([[2.0, 2.2, 5.0]])
    query_embeddings = torch.zeros(B, N, 8)
    return Phase2Output(mask_logits, class_logits, depth_layers, query_embeddings, (H, W))


def test_build_pairs_finds_overlap_pair_deduplicated():
    p2 = _synthetic_p2()
    cfg = Phase3CandidateConfig()
    pairs = build_pairs(p2, cfg)
    # q0<->q1 mutually overlap -> ONE unordered pair (defect D2 fix: the old
    # directional build produced both (0,1) and (1,0), writing the same
    # person twice with disagreeing correction fields). q2 stays isolated.
    assert len(pairs) == 1
    # canonical member order: nearer (dep 2.0) first
    assert pairs.query_idx[0].tolist() == [0, 1]
    assert (pairs.iou > cfg.overlap_iou_thresh).all()
    # normalized boxes in [0,1]
    assert (pairs.boxes_norm >= 0).all() and (pairs.boxes_norm <= 1).all()


def test_build_pairs_filters_low_confidence():
    p2 = _synthetic_p2()
    p2.class_logits[..., 0] = -3.0   # person conf now ~0.05 < 0.9 -> everything filtered
    pairs = build_pairs(p2, Phase3CandidateConfig())
    assert len(pairs) == 0


def test_box_iou_catches_modal_disjoint_masks_that_mask_iou_misses():
    """The empirically-motivated fix: this dataset's GT (and, by inheritance,
    Phase 2's predicted) masks are modal/disjoint for occluded pairs -- two
    people who visually overlap can still have ~0 predicted-mask IoU. Box
    IoU must still catch them; mask IoU must NOT (this is the exact failure
    mode observed as num_pairs~=0 on a real training run).

    Geometry: instance 0's visible remainder is split into two blobs (e.g.
    head + legs, as if instance 1 stands in front and covers its torso);
    instance 1 occupies the region in between. The two instances' actual
    mask pixels never touch (mask IoU == 0 exactly), but instance 0's
    bounding box (spanning both its blobs) still overlaps instance 1's box.
    """
    H = W = 32
    ml = torch.full((1, 2, H, W), -10.0)
    ml[0, 0, 0:6, 10:22] = 10.0     # instance 0, blob 1 ("head")
    ml[0, 0, 26:32, 10:22] = 10.0   # instance 0, blob 2 ("legs")
    ml[0, 1, 6:26, 8:24] = 10.0     # instance 1 ("torso", occluder, in between)
    cl = torch.zeros(1, 2, 2); cl[..., 0] = 3.0
    dep = torch.tensor([[3.0, 2.0]])   # instance 1 (occluder) is nearer
    p2 = Phase2Output(ml, cl, dep, torch.zeros(1, 2, 8), (H, W))

    cfg_box = Phase3CandidateConfig(overlap_metric="box_iou")
    pairs_box = build_pairs(p2, cfg_box)
    assert len(pairs_box) > 0, "box_iou must detect the pair despite disjoint masks"
    # canonical order: instance 1 (dep 2.0) is nearer -> member 0
    assert pairs_box.query_idx[0].tolist() == [1, 0]

    cfg_mask = Phase3CandidateConfig(overlap_metric="mask_iou")
    pairs_mask = build_pairs(p2, cfg_mask)
    assert len(pairs_mask) == 0, "mask_iou should fail here -- masks never touch (IoU=0)"


def test_guest_rule_nearest_depth():
    # q0 overlaps q1 (depth 2.2) and q2 (depth 5.0); nearest to q0(2.0) is q1.
    H = W = 32
    ml = torch.full((1, 3, H, W), -10.0)
    _blob(ml, 0, 0, 0, 16, 0, 16)
    _blob(ml, 0, 1, 8, 20, 8, 20)      # overlaps q0
    _blob(ml, 0, 2, 4, 18, 4, 18)      # also overlaps q0
    cl = torch.zeros(1, 3, 2); cl[..., 0] = 3.0
    dep = torch.tensor([[2.0, 2.2, 5.0]])
    p2 = Phase2Output(ml, cl, dep, torch.zeros(1, 3, 8), (H, W))
    pairs = build_pairs(p2, Phase3CandidateConfig())
    main0 = [p for p in range(len(pairs)) if int(pairs.query_idx[p, 0]) == 0]
    assert main0, "query 0 should be a main"
    assert int(pairs.query_idx[main0[0], 1]) == 1   # nearest-depth guest is q1


def test_roi_extract_shapes():
    p2 = _synthetic_p2()
    pairs = build_pairs(p2, Phase3CandidateConfig())
    P = len(pairs)
    C, hf, wf = 16, 16, 16
    feat = torch.randn(1, C, hf, wf)
    depth = torch.rand(1, 1, 28, 28) + 0.5
    out_hw = (14, 14)
    f_obj, d_obj, g_obj, roi_mask = extract_pair_roi_inputs(
        pairs, feat, depth, p2.mask_logits, out_hw, 2,
        geom_coord=True, geom_global_depth=True, geom_mask_logit=True,
    )
    assert f_obj.shape == (P, 2, C, 14, 14)
    assert d_obj.shape == (P, 2, 1, 14, 14)
    assert g_obj.shape == (P, 2, 1 + 2 + 1, 14, 14)   # mask + coord(2) + gdepth
    assert roi_mask.shape == (P, 2, 1, 14, 14)


def test_relation_head_identity_at_init():
    """Zero-init output conv -> E=0.5 -> Eq. 9 is identity (D_hat == D_obj)."""
    P, C, Hp, Wp = 3, 16, 14, 14
    head = OcclusionRelationHead(per_member_channels=C + 4, granularity="dense")
    f = torch.randn(P, 2, C, Hp, Wp)
    g = torch.randn(P, 2, 4, Hp, Wp)
    d = torch.rand(P, 2, 1, Hp, Wp) + 1.0
    e, d_hat = head(f, g, d)
    assert torch.allclose(e, torch.full_like(e, 0.5), atol=1e-5)
    assert torch.allclose(d_hat, d, atol=1e-5)


def test_eq9_bounds_after_bias():
    """E->1 => D_hat->2D ; E->0 => D_hat->0  (Eq. 9)."""
    P, C, Hp, Wp = 2, 8, 10, 10
    d = torch.rand(P, 2, 1, Hp, Wp) + 1.0
    for bias, expected in [(20.0, 2.0), (-20.0, 0.0)]:
        head = OcclusionRelationHead(per_member_channels=C + 0, granularity="dense")
        torch.nn.init.constant_(head.out.bias, bias)
        _, d_hat = head(torch.randn(P, 2, C, Hp, Wp), torch.zeros(P, 2, 0, Hp, Wp), d)
        assert torch.allclose(d_hat, expected * d, atol=1e-3)


def test_relation_head_scalar_granularity():
    P, C, Hp, Wp = 2, 8, 10, 10
    head = OcclusionRelationHead(per_member_channels=C, granularity="scalar")
    e, d_hat = head(torch.randn(P, 2, C, Hp, Wp), torch.zeros(P, 2, 0, Hp, Wp),
                    torch.rand(P, 2, 1, Hp, Wp) + 1.0)
    # scalar mode broadcasts one value per instance across the ROI
    assert torch.allclose(e[:, :, :, 0:1, 0:1].expand_as(e), e, atol=1e-6)


def _mk_pairset(boxes, query_idx=None, b=0):
    from instancedepth.models.phase3.candidates import PairSet
    boxes = torch.tensor(boxes, dtype=torch.float32)[None]         # (1,2,4)
    return PairSet(
        batch_index=torch.tensor([b]),
        query_idx=torch.tensor(query_idx or [[0, 1]]),
        boxes_norm=boxes,
        iou=torch.tensor([0.3]),
    )


def test_composite_identity_at_e_half():
    """Defect D1 regression: E=0.5 must be an EXACT no-op on the full-res base
    (the old paste-low-res-depth composite could never satisfy this)."""
    H = W = 64
    base = torch.rand(1, 1, H, W) * 3 + 1                          # textured base
    pairs = _mk_pairset([[0.1, 0.1, 0.6, 0.6], [0.4, 0.4, 0.9, 0.9]])
    e = torch.full((1, 2, 1, 14, 14), 0.5)
    layers = torch.tensor([[2.0, 3.0]])
    mask_prob = torch.zeros(1, 2, H, W)
    mask_prob[0, 0, 8:36, 8:36] = 1.0
    mask_prob[0, 1, 28:56, 28:56] = 1.0
    refined = composite_refined_depth(base, pairs, e, layers, mask_prob, 0.5, "dense")
    assert torch.allclose(refined, base, atol=1e-5)


def test_composite_ratio_preserves_base_geometry():
    """Constant E != 0.5 -> refined = (2E)*base inside the mask (geometry
    preserved, uniformly rescaled), base untouched outside."""
    H = W = 64
    base = torch.rand(1, 1, H, W) * 3 + 1
    pairs = _mk_pairset([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    e = torch.zeros(1, 2, 1, 14, 14)
    e[0, 0] = 0.7                                                  # ratio 1.4 for member 0
    e[0, 1] = 0.5                                                  # identity for member 1
    layers = torch.tensor([[2.0, 3.0]])
    mask_prob = torch.zeros(1, 2, H, W)
    mask_prob[0, 0, 4:28, 4:28] = 1.0                              # inside member-0 box
    mask_prob[0, 1, 36:60, 36:60] = 1.0
    for mode in ("dense", "scalar"):
        refined = composite_refined_depth(base, pairs, e, layers, mask_prob, 0.5, mode)
        inside = refined[0, 0, 4:28, 4:28] / base[0, 0, 4:28, 4:28]
        assert torch.allclose(inside, torch.full_like(inside, 1.4), atol=1e-3), mode
        assert torch.allclose(refined[0, 0, 36:60, 36:60], base[0, 0, 36:60, 36:60], atol=1e-5)
        outside = refined[0, 0, 30:34, 0:34]                       # between the masks
        assert torch.allclose(outside, base[0, 0, 30:34, 0:34], atol=1e-6)


def test_composite_nearer_layer_wins_contested_pixels():
    """Defect D3 regression: cross-instance contention is arbitrated by
    per-instance LAYER (occluder wins), mirroring _flatten_id_map -- not by
    per-pixel value comparison."""
    H = W = 64
    base = torch.full((1, 1, H, W), 4.0)
    pairs = _mk_pairset([[0.0, 0.0, 0.75, 0.75], [0.25, 0.25, 1.0, 1.0]])
    e = torch.zeros(1, 2, 1, 14, 14)
    e[0, 0] = 0.75    # nearer instance: ratio 1.5 -> 6.0  (HIGHER value, still must win)
    e[0, 1] = 0.25    # farther instance: ratio 0.5 -> 2.0
    layers = torch.tensor([[2.0, 5.0]])                            # member 0 nearer
    mask_prob = torch.zeros(1, 2, H, W)
    mask_prob[0, 0, 0:48, 0:48] = 1.0
    mask_prob[0, 1, 16:64, 16:64] = 1.0                            # overlaps member 0 in 16:48
    refined = composite_refined_depth(base, pairs, e, layers, mask_prob, 0.5, "dense")
    contested = refined[0, 0, 20:44, 20:44]
    # per-pixel min would pick 2.0 (farther member's lower value); layer-based
    # arbitration must pick the NEARER instance's 6.0 despite the higher value
    assert torch.allclose(contested, torch.full_like(contested, 6.0), atol=1e-3)


def test_dense_gt_rois_no_boundary_dilution():
    """Defect D4 regression: a uniform-depth instance must yield targets equal
    to that depth in EVERY valid cell -- the old ROIAlign of depth*valid
    diluted boundary cells toward zero."""
    from instancedepth.models.phase3.candidates import PairSet
    H = W = 32
    pairs = PairSet(
        batch_index=torch.tensor([0]),
        query_idx=torch.tensor([[0, 1]]),
        # member-0 box deliberately extends PAST the instance boundary
        # (instance covers cols 0:16 = x<0.5; box spans x in [0, 0.7])
        boxes_norm=torch.tensor([[[0.0, 0.0, 0.7, 1.0], [0.5, 0.5, 1.0, 1.0]]]),
        iou=torch.tensor([0.3]),
    )
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    masks = torch.zeros(2, H, W)
    masks[0, :, 0:16] = 1.0
    masks[1, 16:32, 16:32] = 1.0
    targets = [{"masks": masks, "depths": torch.tensor([3.0, 4.0]),
                "labels": torch.tensor([0, 0])}]
    gt_depth = torch.zeros(1, 1, H, W)
    gt_depth[0, 0, :, 0:16] = 3.0
    gt_depth[0, 0, 16:32, 16:32] = 4.0
    t = build_dense_gt_rois(pairs, indices, targets, gt_depth, (14, 14), 2)
    vals = t.dt_dense[0, 0][t.dt_valid[0, 0]]
    assert vals.numel() > 0
    assert torch.allclose(vals, torch.full_like(vals, 3.0), atol=1e-3), (
        f"boundary dilution detected: min target {vals.min():.3f} (expected 3.0)")


def test_min_valid_roi_px_filters_sparse_rois():
    """Defect D5 regression: ROIs with fewer valid GT pixels than
    min_valid_roi_px must contribute no L_obj."""
    P, Hp, Wp = 1, 14, 14
    d_hat = (torch.rand(P, 2, 1, Hp, Wp) + 1.0).requires_grad_(True)
    base_depth = torch.rand(1, 1, 32, 32) + 0.5
    dt_valid = torch.zeros(P, 2, 1, Hp, Wp, dtype=torch.bool)
    dt_valid[..., 0, :3] = True                                    # only 3 valid px per member
    output = RefinedDepthOutput(
        refined_depth=base_depth, base_depth=base_depth, image_hw=(32, 32),
        pair_batch_index=torch.zeros(P, dtype=torch.long),
        pair_query_idx=torch.zeros(P, 2, dtype=torch.long),
        pair_iou=torch.zeros(P), refined_layers=torch.rand(P, 2) + 1.0,
        base_layers=torch.rand(P, 2) + 1.0, d_hat_roi=d_hat,
        e_obj_roi=torch.rand(P, 2, 1, Hp, Wp), d_obj_roi=d_hat.detach().clone(),
    )
    tgt = RefineTargets(
        dt_scalar=torch.rand(P, 2) + 1.0, pair_valid=torch.zeros(P, dtype=torch.bool),
        dt_dense=(torch.rand(P, 2, 1, Hp, Wp) + 1.0) * dt_valid, dt_valid=dt_valid,
    )
    crit = Phase3Criterion(Phase3LossConfig(min_valid_roi_px=16))
    losses = crit(output, tgt, torch.rand(1, 1, 32, 32) + 0.5)
    assert float(losses["l_obj"].detach()) == 0.0


def test_build_dense_gt_rois():
    from instancedepth.models.phase3.candidates import PairSet
    H = W = 32
    pairs = PairSet(
        batch_index=torch.tensor([0]),
        query_idx=torch.tensor([[0, 1]]),
        boxes_norm=torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]]),
        iou=torch.tensor([0.3]),
    )
    # boxes strictly INTERIOR to disjoint instance regions, so ROIAlign never
    # samples across an instance boundary (edge bilinear would blend in 0s).
    pairs.boxes_norm = torch.tensor([[[0.05, 0.05, 0.45, 0.45],
                                      [0.55, 0.55, 0.95, 0.95]]])
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]   # q0->gt0, q1->gt1
    masks = torch.zeros(2, H, W)
    masks[0, 0:16, 0:16] = 1.0        # instance 0 (disjoint from instance 1)
    masks[1, 16:32, 16:32] = 1.0
    targets = [{"masks": masks, "depths": torch.tensor([2.0, 4.0]),
                "labels": torch.tensor([0, 0])}]
    gt_depth = torch.zeros(1, 1, H, W)
    gt_depth[0, 0, 0:16, 0:16] = 2.0
    gt_depth[0, 0, 16:32, 16:32] = 4.0
    t = build_dense_gt_rois(pairs, indices, targets, gt_depth, (14, 14), 2)
    assert isinstance(t, RefineTargets)
    assert t.pair_valid.item() is True
    assert torch.allclose(t.dt_scalar, torch.tensor([[2.0, 4.0]]))
    assert t.dt_valid.any()
    # GT depth inside each member's interior ROI == its instance depth
    assert torch.allclose(t.dt_dense[0, 0][t.dt_valid[0, 0]],
                          torch.full_like(t.dt_dense[0, 0][t.dt_valid[0, 0]], 2.0), atol=1e-4)
    assert torch.allclose(t.dt_dense[0, 1][t.dt_valid[0, 1]],
                          torch.full_like(t.dt_dense[0, 1][t.dt_valid[0, 1]], 4.0), atol=1e-4)


def test_phase3_criterion_runs():
    P, Hp, Wp = 2, 14, 14
    # d_hat / refined_layers carry grad in real training (they come from Phi_o);
    # mirror that here so total.backward() has a graph to traverse.
    d_hat = (torch.rand(P, 2, 1, Hp, Wp) + 1.0).requires_grad_(True)
    refined_layers = (torch.rand(P, 2) + 1.0).requires_grad_(True)
    base_depth = (torch.rand(1, 1, 32, 32) + 0.5).requires_grad_(True)
    output = RefinedDepthOutput(
        refined_depth=base_depth, base_depth=base_depth, image_hw=(32, 32),
        pair_batch_index=torch.zeros(P, dtype=torch.long),
        pair_query_idx=torch.zeros(P, 2, dtype=torch.long),
        pair_iou=torch.zeros(P), refined_layers=refined_layers,
        base_layers=torch.rand(P, 2) + 1.0, d_hat_roi=d_hat,
        e_obj_roi=torch.rand(P, 2, 1, Hp, Wp), d_obj_roi=d_hat.detach().clone(),
    )
    tgt = RefineTargets(
        dt_scalar=torch.rand(P, 2) + 1.0, pair_valid=torch.ones(P, dtype=torch.bool),
        dt_dense=torch.rand(P, 2, 1, Hp, Wp) + 1.0,
        dt_valid=torch.ones(P, 2, 1, Hp, Wp, dtype=torch.bool),
    )
    gt = torch.rand(1, 1, 32, 32) + 0.5
    crit = Phase3Criterion(Phase3LossConfig())
    losses = crit(output, tgt, gt)
    assert "l_obj" in losses and "l_dist" in losses and "total" in losses
    losses["total"].backward()
    # L_obj feeds gradient to D_hat, L_dist to the refined layers (in the real
    # model both trace back to Phase 1 via ROIAlign(depth_p1)).
    assert d_hat.grad is not None
    assert refined_layers.grad is not None


def test_phase3_criterion_empty_pairs():
    base_depth = (torch.rand(1, 1, 32, 32) + 0.5).requires_grad_(True)
    z = torch.zeros(0, 2, 1, 14, 14)
    output = RefinedDepthOutput(
        refined_depth=base_depth, base_depth=base_depth, image_hw=(32, 32),
        pair_batch_index=torch.zeros(0, dtype=torch.long),
        pair_query_idx=torch.zeros(0, 2, dtype=torch.long),
        pair_iou=torch.zeros(0), refined_layers=torch.zeros(0, 2),
        base_layers=torch.zeros(0, 2), d_hat_roi=z, e_obj_roi=z, d_obj_roi=z,
    )
    tgt = RefineTargets(torch.zeros(0, 2), torch.zeros(0, dtype=torch.bool),
                        z.clone(), z.clone().bool())
    losses = Phase3Criterion(Phase3LossConfig())(output, tgt, torch.rand(1, 1, 32, 32) + 0.5)
    total = losses["total"]
    total.backward()   # must still have a grad path (via base_depth) with zero pairs
    assert float(total.detach()) == 0.0


def test_config_loads_and_resolves_subphases():
    from instancedepth.configs.phase3_config import Phase3Config
    cfg = Phase3Config.from_yaml("instancedepth/configs/phase3.yaml")
    assert cfg.phase1 is not None and cfg.phase2 is not None
    assert cfg.head.refine_granularity == "dense"
    assert cfg.optim.lr == 1.0e-6 and cfg.optim.total_iters == 25000
    assert cfg.data.occlusion_only is False   # faithful default


def test_occlusion_frame_indices():
    from instancedepth.data.occlusion_index import occlusion_frame_indices

    class FakeDS:
        pass

    def frame(insts):
        return {"instances": insts}

    def inst(box, d=3.0):
        return {"bbox_xyxy": box, "depth_layer_m": d}

    man = {"frames": {
        "f_overlap": frame([inst([0, 0, 10, 10]), inst([5, 5, 15, 15])]),   # overlap -> selected
        "f_disjoint": frame([inst([0, 0, 5, 5]), inst([20, 20, 30, 30])]),  # no overlap
        "f_single": frame([inst([0, 0, 10, 10])]),                          # <2 instances
        "f_invalid_depth": frame([inst([0, 0, 10, 10], d=0.0), inst([5, 5, 15, 15], d=0.0)]),  # dropped (no valid depth)
    }}
    ds = FakeDS()
    ds.index = [(man, "f_overlap"), (man, "f_disjoint"), (man, "f_single"), (man, "f_invalid_depth")]
    sel = occlusion_frame_indices(ds, max_depth=10.0)
    assert sel == [0]   # only f_overlap qualifies


# --------------------------------------------------------------------------- #
# freeze_phase1 (docs/AUDIT_2026.md): pin the Phase-1 depth branch so ROI-only
# Phase-3 supervision can't drift the dense base (0.078 -> 0.139 abs_rel).
# --------------------------------------------------------------------------- #
def test_freeze_phase1_default_is_true():
    from instancedepth.configs.phase3_config import Phase3Config
    assert Phase3Config().freeze_phase1 is True


def test_phase3_optimizer_handles_frozen_phase1():
    """With Phase 1 frozen its params carry requires_grad=False, so the
    optimizer must build a SINGLE (head-only) group and never see the frozen
    depth params."""
    import torch.nn as nn
    from instancedepth.engine.train_phase3 import build_phase3_optimizer

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.phase1 = nn.Linear(4, 4)            # "depth branch"
            self.relation_head = nn.Linear(4, 2)     # Phi_o
    m = Fake()
    for p in m.phase1.parameters():
        p.requires_grad_(False)

    class OptimCfg:
        lr = 1e-6
        head_lr_mult = 10.0
        weight_decay = 0.01
    opt = build_phase3_optimizer(m, OptimCfg())
    assert len(opt.param_groups) == 1                      # depth group dropped
    trainable = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in trainable for p in m.phase1.parameters())
    assert all(id(p) in trainable for p in m.relation_head.parameters())
    assert opt.param_groups[0]["lr"] == 1e-6 * 10.0
