"""Phase 3 qualitative debugging visualizer.

For each sampled test frame writes two kinds of PNGs to --out-dir:

  frame_<idx>_panel.png   the full refinement story in one grid:
      row 1: RGB | GT depth | Phase-1 base | Phase-3 refined (config composite)
      row 2: refined (scalar composite) | abs err base | abs err refined |
             refined-minus-base (signed: where & how refinement acted)
      row 3: GT instance masks | Phase-2 instances + Dep_i labels |
             pair-member predicted masks | candidate pairs (boxes +
             occluder->occludee link + IoU/Dep)

      The Phase-2 panel shows every instance the branch predicts above
      --inst-score-thresh; the pair-member panel shows only those that
      survived Phase 3's stricter 0.9/0.8 filter AND formed a pair -- the
      gap between the two panels is what Phase 3 declined to act on.

  frame_<idx>_pair<p>.png  per-pair ROI strip, one row per member
      (0 = nearer/occluder, 1 = farther):
      RGB crop | mask ROI | D_obj (ROI base depth) | E field (signed around
      0.5) | D_hat (refined ROI) | GT depth crop

Usage (server, project root):

    python -m scripts.visualize_phase3 \\
        --config instancedepth/configs/phase3_current.yaml \\
        --checkpoint runs/phase3_current/best.pth \\
        --out-dir viz/phase3_current --num-frames 12

By default samples occlusion frames (>=2 overlapping GT instances) -- the
frames where Phase 3 actually does something; --all-frames disables that.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset
from instancedepth.data.occlusion_index import occlusion_frame_indices
from instancedepth.models.phase3.inference import Phase3Inferencer
from instancedepth.models.phase3.relation_head import composite_refined_depth
from instancedepth.utils.viz import (
    colorize_depth, colorize_error, colorize_signed, denormalize_image,
    draw_instances_with_depth, draw_pairs, hstack_panels, overlay_masks, put_label,
    stack_grid,
)

log = logging.getLogger("scripts.visualize_phase3")


def _crop_norm(img: np.ndarray, box_norm) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_norm
    X1, Y1 = int(round(x1 * W)), int(round(y1 * H))
    X2, Y2 = max(int(round(x2 * W)), X1 + 1), max(int(round(y2 * H)), Y1 + 1)
    return img[Y1:Y2, X1:X2]


def visualize_frame(inferencer: Phase3Inferencer, sample: dict, cfg: Phase3Config,
                    out_dir: Path, tag: str, err_cap: float = 1.0,
                    inst_score_thresh: float = 0.5) -> None:
    max_d = cfg.data.max_depth
    img_t = sample["image"]                                   # (3,H,W) normalized
    rgb = denormalize_image(img_t.numpy())                    # BGR
    gt = sample["depth"][0].numpy()                           # (H,W)
    gt_valid = gt > 0

    pred = inferencer.predict(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    refined, base = pred["refined"], pred["base"]
    output, aux = pred["output"], pred["aux"]
    pairs, p2 = aux["pairs"], aux["p2"]
    P = len(pairs)

    # scalar-composite variant for direct comparison against the config mode
    if P > 0:
        scalar_map = composite_refined_depth(
            output.base_depth.float(), pairs, output.e_obj_roi.float(),
            output.refined_layers.float(), p2.mask_logits.sigmoid().float(),
            cfg.candidate.mask_binarize_thresh, ratio_mode="scalar",
        )[0, 0].cpu().numpy()
        if scalar_map.shape != refined.shape:
            scalar_map = cv2.resize(scalar_map, refined.shape[::-1], interpolation=cv2.INTER_LINEAR)
    else:
        scalar_map = refined

    gt_masks = sample["targets"]["masks"].numpy() > 0.5      # (G,H,W)
    pair_boxes = pairs.boxes_norm.cpu().numpy() if P else np.zeros((0, 2, 4))
    pair_ious = pairs.iou.cpu().numpy().tolist() if P else []
    pair_deps = None
    member_masks = []
    if P:
        dep_all = p2.depth_layers[0].float().cpu().numpy()
        pair_deps = np.stack([[dep_all[int(q)] for q in row] for row in pairs.query_idx.cpu().numpy()])
        mask_prob = p2.mask_logits.sigmoid()[0].float().cpu().numpy()
        member_qs = sorted({int(q) for row in pairs.query_idx.cpu().numpy() for q in row})
        member_masks = [mask_prob[q] >= cfg.candidate.mask_binarize_thresh for q in member_qs]

    # Everything Phase 2 predicts on this frame (not just the queries that
    # survived Phase 3's stricter 0.9/0.8 candidate filter and formed pairs),
    # each labelled with its Dep_i -- shows what the instance branch actually
    # sees vs. what Phase 3 chose to act on.
    p2_probs = p2.mask_logits.sigmoid()[0].float().cpu().numpy()
    p2_deps_all = p2.depth_layers[0].float().cpu().numpy()
    p2_keep = np.where(p2.scores()[0].float().cpu().numpy() > inst_score_thresh)[0]
    p2_masks, p2_deps = [], []
    for q in p2_keep.tolist():
        m = p2_probs[q] >= cfg.candidate.mask_binarize_thresh
        if m.any():
            p2_masks.append(m)
            p2_deps.append(float(p2_deps_all[q]))

    panels = [
        ("RGB", rgb),
        ("GT depth", colorize_depth(gt, max_d)),
        ("Phase-1 base", colorize_depth(base, max_d)),
        (f"Refined ({cfg.head.composite_ratio})", colorize_depth(refined, max_d)),
        ("Refined (scalar ratio)", colorize_depth(scalar_map, max_d)),
        (f"|base-GT| (cap {err_cap}m)", colorize_error(np.abs(base - gt), err_cap, gt_valid)),
        (f"|refined-GT| (cap {err_cap}m)", colorize_error(np.abs(refined - gt), err_cap, gt_valid)),
        ("refined-base (signed, cap 0.5m)", colorize_signed(refined - base, 0.5)),
        ("GT instance masks", overlay_masks(rgb, gt_masks)),
        (f"Phase-2 instances + Dep_i ({len(p2_masks)})",
         draw_instances_with_depth(rgb, p2_masks, p2_deps)),
        ("pair-member pred masks", overlay_masks(rgb, np.array(member_masks)) if member_masks else rgb),
        (f"candidate pairs (P={P})", draw_pairs(rgb, pair_boxes, pair_ious, pair_deps)),
    ]
    grid = stack_grid(panels, cols=4)
    cv2.imwrite(str(out_dir / f"{tag}_panel.png"), grid)

    # ---- per-pair ROI strips ------------------------------------------------
    for p in range(P):
        rows = []
        for k in range(2):
            box = pair_boxes[p, k]
            q = int(pairs.query_idx[p, k])
            d_obj = output.d_obj_roi[p, k, 0].float().cpu().numpy()
            e = output.e_obj_roi[p, k, 0].float().cpu().numpy()
            d_hat = output.d_hat_roi[p, k, 0].float().cpu().numpy()
            mask_roi = _crop_norm(p2.mask_logits.sigmoid()[0, q].float().cpu().numpy(), box)
            role = "occluder" if k == 0 else "occludee"
            strip = hstack_panels([
                put_label(cv2.resize(_crop_norm(rgb, box), (160, 160)), f"q{q} {role}"),
                put_label(cv2.resize((mask_roi * 255).astype(np.uint8), (160, 160)), "mask ROI"),
                put_label(cv2.resize(colorize_depth(d_obj, max_d), (160, 160)), "D_obj"),
                put_label(cv2.resize(colorize_signed(e - 0.5, 0.5), (160, 160)), "E-0.5"),
                put_label(cv2.resize(colorize_depth(d_hat, max_d), (160, 160)), "D_hat"),
                put_label(cv2.resize(colorize_depth(_crop_norm(gt, box), max_d), (160, 160)), "GT crop"),
            ])
            rows.append(strip)
        cv2.imwrite(str(out_dir / f"{tag}_pair{p}.png"), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num-frames", type=int, default=12)
    ap.add_argument("--all-frames", action="store_true",
                    help="sample from all frames, not just occlusion frames")
    ap.add_argument("--err-cap", type=float, default=1.0, help="error-map color cap in meters")
    ap.add_argument("--inst-score-thresh", type=float, default=0.5,
                    help="category-confidence cut for the Phase-2 instances panel (viz-oriented, "
                         "looser than Phase 3's own 0.9 candidate filter)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = Phase3Config.from_yaml_with_overrides(args.config, args.override)
    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=cfg.data.annotations_root, split=args.split,
        image_size=cfg.data.image_size, max_depth=cfg.data.max_depth,
        min_instance_px=cfg.data.min_instance_px, hflip_prob=0.0, color_jitter=0.0,
        size_divisor=cfg.data.size_divisor,
    ))
    pool = occlusion_frame_indices(ds, cfg.data.max_depth) if not args.all_frames else list(range(len(ds)))
    if not pool:
        raise RuntimeError("no frames to visualize (occlusion pool empty; try --all-frames)")
    picks = [pool[i] for i in np.linspace(0, len(pool) - 1, min(args.num_frames, len(pool)), dtype=int)]

    inferencer = Phase3Inferencer(cfg, args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for j, idx in enumerate(picks):
        sample = ds[idx]
        tag = f"frame_{idx:06d}"
        log.info("[%d/%d] %s (%s / %s)", j + 1, len(picks), tag,
                 sample["meta"]["sequence"], sample["meta"]["frame"])
        visualize_frame(inferencer, sample, cfg, out_dir, tag, err_cap=args.err_cap,
                        inst_score_thresh=args.inst_score_thresh)

    (out_dir / "index.txt").write_text("\n".join(
        f"frame_{i:06d}  {ds[i]['meta']['sequence']} / {ds[i]['meta']['frame']}" for i in picks))
    log.info("Wrote %d frame visualizations to %s", len(picks), out_dir)


if __name__ == "__main__":
    main()
