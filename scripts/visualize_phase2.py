"""Write Phase 2 (Instance Depth Layer Prediction) qualitative results to
TensorBoard for visual inspection against the ground truth, on the test
split.

For each sampled test frame it logs one side-by-side strip under the
IMAGES tab:

    [ RGB | GT instances | predicted instances ]

Each instance is drawn as a translucent coloured fill plus a bright
contour, with its depth-layer value (metres) printed at the mask centroid.
Instances are drawn far-to-near (nearest on top) so occlusion ordering
reads correctly -- the same ordering Phase 3 will reason about. GT uses the
dataset's per-instance masks/depths; predictions are the queries surviving
the score + mask-confidence thresholds (the same filtering
``evaluate_phase2.py`` applies).

``--occlusion-only`` restricts the output to frames with >=2 overlapping GT
instances -- the exact slice the Phase 2 redesign is judged on (plan
SS2.3/SS6), and where the original implementation's wrong-ownership /
fragmentation failures showed up. Start here when eyeballing whether the
redesign worked.

This is a read-only diagnostic: it never trains, and writes to its own
``viz/`` subdir so it can't collide with the training run's ``tb/`` scalars.

Usage (on the Backend.AI server, after Phase 2 finished):

    python -m scripts.visualize_phase2 \
        --config instancedepth/configs/phase2_mask2former.yaml \
        --checkpoint runs/phase2_mask2former/best.pth \
        --num-samples 24 --occlusion-only

Then point the Backend.AI TensorBoard extension at
``runs/phase2_mask2former`` (or the parent ``runs/``) and open the IMAGES
tab. The scalar metrics (eval/overall_*, eval/occlusion_*) logged during
training live in the same tree under SCALARS; for authoritative full-test
metrics run ``python -m instancedepth.engine.evaluate_phase2``.
"""

from __future__ import annotations

import argparse
import colorsys
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from instancedepth.configs.phase2_config import Phase2Config
from instancedepth.data.gid_dataset import IMAGENET_MEAN, IMAGENET_STD
from instancedepth.engine.train_phase2 import build_dataloader
from instancedepth.models.phase2.model import Phase2Model
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.phase2_metrics import has_overlapping_instances

log = logging.getLogger("scripts.visualize_phase2")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _palette(n: int) -> np.ndarray:
    """n visually distinct RGB uint8 colours (golden-ratio hue spacing)."""
    colors = []
    for i in range(max(n, 1)):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def _denorm_rgb_u8(img_norm: np.ndarray) -> np.ndarray:
    """(3,H,W) ImageNet-normalized -> (H,W,3) uint8."""
    mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, np.float32)[:, None, None]
    rgb = np.clip(img_norm * std + mean, 0.0, 1.0)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def _overlay_instances(base_u8: np.ndarray, masks: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """base_u8: (H,W,3) uint8 (copied inside). masks: (K,H,W) bool.
    depths: (K,) metres. Draws far-to-near so nearest instances sit on top."""
    img = base_u8.copy()
    k = masks.shape[0]
    if k == 0:
        return img
    palette = _palette(k)
    order = np.argsort(-depths)              # far (large depth) first, near last -> on top
    for draw_i, idx in enumerate(order.tolist()):
        m = masks[idx]
        if not m.any():
            continue
        color = palette[draw_i]              # colour by draw order for max separation
        # translucent fill
        img[m] = (0.55 * img[m] + 0.45 * color).astype(np.uint8)
        # bright contour
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, [int(c) for c in color], 2)
        # depth-layer label at centroid (dark halo + white text for legibility)
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = f"{float(depths[idx]):.1f}m"
        cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _hstack_u8(panels, sep: int = 8) -> np.ndarray:
    h = panels[0].shape[0]
    white = np.full((h, sep, 3), 255, dtype=np.uint8)
    out = []
    for i, p in enumerate(panels):
        out.append(p)
        if i != len(panels) - 1:
            out.append(white)
    return np.concatenate(out, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="e.g. instancedepth/configs/phase2_mask2former.yaml")
    ap.add_argument("--checkpoint", required=True, help="e.g. runs/phase2_mask2former/best.pth")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--max-per-sequence", type=int, default=None,
                    help="cap frames drawn from any one sequence/video (default: auto -- spread "
                         "num_samples as evenly as possible across all sequences). The test loader "
                         "is unshuffled and in-sequence order, so without this all samples come from "
                         "the first video.")
    ap.add_argument("--score-threshold", type=float, default=0.5, help="keep queries with foreground conf above this")
    ap.add_argument("--mask-threshold", type=float, default=0.5, help="binarize mask probability at this value")
    ap.add_argument("--occlusion-only", action="store_true",
                    help="only visualize frames with >=2 overlapping GT instances (the redesign's target slice)")
    ap.add_argument("--logdir", default=None, help="default: <run_root>/<run_name>/viz")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from torch.utils.tensorboard import SummaryWriter

    cfg = Phase2Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Phase2Model(
        checkpoint=cfg.model.checkpoint, checkpoint_dir=cfg.model.checkpoint_dir,
        allow_hub_download=cfg.model.allow_hub_download, num_classes=cfg.model.num_classes,
    ).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)
    model.eval()

    logdir = Path(args.logdir) if args.logdir else Path(cfg.run_root) / cfg.run_name / "viz"
    writer = SummaryWriter(log_dir=str(logdir))
    tag_prefix = f"{args.split}_occlusion" if args.occlusion_only else args.split
    log.info("writing up to %d %s-split visualizations to %s (occlusion_only=%s)",
             args.num_samples, args.split, logdir, args.occlusion_only)

    loader = build_dataloader(cfg, split=args.split)
    precision = cfg.optim.precision
    use_autocast = precision != "fp32" and device.type == "cuda"

    # Spread samples across sequences instead of draining the first video
    # (the test loader is unshuffled and walks one sequence at a time).
    num_sequences = len({man["sequence"] for man, _ in loader.dataset.index})
    per_seq_cap = args.max_per_sequence or max(1, math.ceil(args.num_samples / max(num_sequences, 1)))
    log.info("%s split has %d sequences; taking up to %d %sframe(s) per sequence",
             args.split, num_sequences, per_seq_cap, "occlusion " if args.occlusion_only else "")
    seq_count: dict = {}

    written = 0
    scanned = 0
    for batch in loader:
        if written >= args.num_samples:
            break
        metas = batch["meta"]
        scanned += 1

        # Decide which items are worth rendering using GT ONLY (sequence cap
        # + occlusion check) BEFORE the expensive Mask2Former forward pass.
        # Occlusion is a GT-only property, so filtering here means batches
        # with no eligible frames never pay for inference -- this is what
        # keeps --occlusion-only fast instead of running the model over the
        # whole split (the earlier version ran the forward first, then
        # filtered, so a rare-occlusion split cost a full-dataset inference).
        needed = []
        for b in range(len(metas)):
            if seq_count.get(metas[b]["sequence"], 0) >= per_seq_cap:
                continue
            if args.occlusion_only and not has_overlapping_instances(batch["targets"][b]["masks"].bool()):
                continue
            needed.append(b)
        if not needed:
            continue

        image = batch["image"].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            out = model(image)
        scores = out.scores()               # (B,N)
        mask_probs = out.mask_confidence()  # (B,N,H,W)

        for b in needed:
            if written >= args.num_samples:
                break
            seq = metas[b]["sequence"]
            if seq_count.get(seq, 0) >= per_seq_cap:
                continue
            tgt = batch["targets"][b]
            gt_masks = tgt["masks"].bool()          # (G,H,W) on CPU
            gt_depths = tgt["depths"].cpu().numpy()

            rgb_u8 = _denorm_rgb_u8(image[b].cpu().numpy())

            keep = scores[b] > args.score_threshold
            pred_masks = (mask_probs[b, keep] > args.mask_threshold).cpu().numpy()   # (P,H,W)
            pred_depths = out.depth_layers[b, keep].cpu().numpy()                    # (P,)

            gt_panel = _overlay_instances(rgb_u8, gt_masks.numpy(), gt_depths)
            pred_panel = _overlay_instances(rgb_u8, pred_masks, pred_depths)

            seq_short = str(seq).replace("/", "-")
            strip = _hstack_u8([rgb_u8, gt_panel, pred_panel])   # (H, 3W+seps, 3) uint8
            writer.add_image(f"{tag_prefix}/{written:03d}_{seq_short}_rgb-gt-pred",
                             strip.transpose(2, 0, 1), global_step=0)   # uint8 CHW
            seq_count[seq] = seq_count.get(seq, 0) + 1
            written += 1

    writer.close()
    if written == 0 and args.occlusion_only:
        log.warning("wrote 0 strips after scanning %d batches -- no frames with >=2 bounding-box-"
                    "overlapping GT instances were found. Re-run without --occlusion-only, or lower the "
                    "overlap threshold in phase2_metrics.has_overlapping_instances.", scanned)
    log.info("done: wrote %d strips across %d sequences. Panel order: RGB | GT instances | predicted "
             "instances. Each instance: translucent fill + contour + depth-layer label; drawn "
             "far-to-near (nearest on top). Predictions filtered at score>%.2f, mask prob>%.2f.",
             written, len(seq_count), args.score_threshold, args.mask_threshold)


if __name__ == "__main__":
    main()
