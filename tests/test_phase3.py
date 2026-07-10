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


def test_build_pairs_finds_overlap_pair():
    p2 = _synthetic_p2()
    cfg = Phase3CandidateConfig()
    pairs = build_pairs(p2, cfg)
    # q0<->q1 overlap both directions; q2 isolated -> exactly 2 directional pairs
    assert len(pairs) == 2
    mains = set(int(pairs.query_idx[p, 0]) for p in range(len(pairs)))
    guests = set(int(pairs.query_idx[p, 1]) for p in range(len(pairs)))
    assert mains == {0, 1} and guests == {0, 1}
    assert 2 not in mains and 2 not in guests
    assert (pairs.iou > cfg.overlap_iou_thresh).all()
    # normalized boxes in [0,1]
    assert (pairs.boxes_norm >= 0).all() and (pairs.boxes_norm <= 1).all()


def test_build_pairs_filters_low_confidence():
    p2 = _synthetic_p2()
    p2.class_logits[..., 0] = -3.0   # person conf now ~0.05 < 0.9 -> everything filtered
    pairs = build_pairs(p2, Phase3CandidateConfig())
    assert len(pairs) == 0


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


def test_composite_nearest_wins():
    from instancedepth.models.phase3.candidates import PairSet
    H = W = 32
    base = torch.full((1, 1, H, W), 5.0)
    pairs = PairSet(
        batch_index=torch.tensor([0]),
        query_idx=torch.tensor([[0, 1]]),
        boxes_norm=torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]]),
        iou=torch.tensor([0.5]),
    )
    d_hat = torch.full((1, 2, 1, 14, 14), 2.0)   # both members refine to 2.0
    mask_prob = torch.zeros(1, 2, H, W)
    mask_prob[0, 0, 0:16, 0:16] = 1.0            # main covers top-left quadrant
    refined = composite_refined_depth(base, pairs, d_hat, mask_prob, 0.5)
    assert torch.allclose(refined[0, 0, 0:16, 0:16], torch.full((16, 16), 2.0), atol=1e-4)
    assert torch.allclose(refined[0, 0, 16:, 16:], torch.full((16, 16), 5.0), atol=1e-4)


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
