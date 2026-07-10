"""Write Phase 1 (Holistic Depth Initialization) qualitative results to
TensorBoard for visual inspection against the ground truth, on the test
split.

For each sampled test frame it logs one side-by-side strip under the
IMAGES tab:

    [ RGB | GT depth | predicted depth | abs error ]

GT depth and predicted depth share the *same* colour range per frame (the
GT valid-pixel 2nd..98th percentile), so they are directly comparable by
eye; invalid GT pixels (depth == 0) render black. This is purely a
read-only diagnostic -- it never trains, and writes to its own ``viz/``
subdir so it can't collide with the training run's ``tb/`` scalars.

Usage (on the Backend.AI server, after Phase 1 finished):

    python -m scripts.visualize_hdi \
        --config instancedepth/configs/hdi_enhanced.yaml \
        --checkpoint runs/hdi_enhanced/best.pth \
        --num-samples 24

Then point the Backend.AI TensorBoard extension at ``runs/hdi_enhanced``
(or the parent ``runs/``) and open the IMAGES tab. The scalar metrics
(eval/abs_rel, eval/rms, ...) logged during training live in the same tree
under the SCALARS tab; for the authoritative full-test-set metrics run
``python -m instancedepth.engine.evaluate_hdi`` (see its docstring).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch

from instancedepth.configs.config import HDIConfig
from instancedepth.data.gid_dataset import IMAGENET_MEAN, IMAGENET_STD
from instancedepth.engine.train_hdi import build_dataloader
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.utils.checkpoint import load_checkpoint

log = logging.getLogger("scripts.visualize_hdi")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _get_cmap(name: str):
    """Perceptual colormap if matplotlib is available; else None (grayscale)."""
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except Exception:
        try:
            from matplotlib import cm
            return cm.get_cmap(name)
        except Exception:
            log.warning("matplotlib not available -- depth maps will render as grayscale.")
            return None


def _denorm_rgb(img_norm: np.ndarray) -> np.ndarray:
    """(3,H,W) ImageNet-normalized -> (3,H,W) float in [0,1]."""
    mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, np.float32)[:, None, None]
    return np.clip(img_norm * std + mean, 0.0, 1.0)


def _colorize(depth: np.ndarray, mask: np.ndarray, vmin: float, vmax: float, cmap) -> np.ndarray:
    """(H,W) depth -> (3,H,W) float in [0,1]; masked-out pixels are black."""
    d = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    if cmap is not None:
        rgb = np.asarray(cmap(d))[..., :3]           # (H,W,3)
    else:
        rgb = np.stack([d, d, d], axis=-1)
    rgb = rgb.copy()
    rgb[~mask] = 0.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32)


def _hstack(panels, sep: int = 8) -> np.ndarray:
    """Concatenate (3,H,W) panels left-to-right with white separators."""
    h = panels[0].shape[1]
    white = np.ones((3, h, sep), dtype=np.float32)
    out = []
    for i, p in enumerate(panels):
        out.append(p)
        if i != len(panels) - 1:
            out.append(white)
    return np.concatenate(out, axis=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="e.g. instancedepth/configs/hdi_enhanced.yaml")
    ap.add_argument("--checkpoint", required=True, help="e.g. runs/hdi_enhanced/best.pth")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--max-per-sequence", type=int, default=None,
                    help="cap frames drawn from any one sequence/video (default: auto -- spread "
                         "num_samples as evenly as possible across all sequences). The test loader "
                         "is unshuffled and in-sequence order, so without this all samples come from "
                         "the first video.")
    ap.add_argument("--cmap", default="turbo", help="matplotlib colormap for depth (turbo/viridis/magma)")
    ap.add_argument("--logdir", default=None,
                    help="TensorBoard output dir (default: <run_root>/<run_name>/viz)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from torch.utils.tensorboard import SummaryWriter

    cfg = HDIConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmap = _get_cmap(args.cmap)
    err_cmap = _get_cmap("magma")

    model = HolisticDepthModel(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)
    model.eval()

    logdir = Path(args.logdir) if args.logdir else Path(cfg.run_root) / cfg.run_name / "viz"
    writer = SummaryWriter(log_dir=str(logdir))
    log.info("writing %d %s-split visualizations to %s", args.num_samples, args.split, logdir)

    loader = build_dataloader(cfg, split=args.split)
    precision = cfg.optim.precision
    use_autocast = precision != "fp32" and device.type == "cuda"

    # Spread samples across sequences instead of draining the first video
    # (the test loader is unshuffled and walks one sequence at a time).
    num_sequences = len({man["sequence"] for man, _ in loader.dataset.index})
    per_seq_cap = args.max_per_sequence or max(1, math.ceil(args.num_samples / max(num_sequences, 1)))
    log.info("%s split has %d sequences; taking up to %d frame(s) per sequence",
             args.split, num_sequences, per_seq_cap)
    seq_count: dict = {}

    written = 0
    for batch in loader:
        metas = batch["meta"]
        needed = [b for b in range(len(metas)) if seq_count.get(metas[b]["sequence"], 0) < per_seq_cap]
        if written >= args.num_samples:
            break
        if not needed:
            continue   # whole batch is from already-capped sequences -> skip the forward pass

        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            out = model(image)
        pred = out.depth_final.float()

        for b in needed:
            if written >= args.num_samples:
                break
            seq = metas[b]["sequence"]
            if seq_count.get(seq, 0) >= per_seq_cap:
                continue
            rgb_np = _denorm_rgb(image[b].cpu().numpy())
            gt_np = gt_depth[b, 0].cpu().numpy()
            pred_np = pred[b, 0].cpu().numpy()
            valid = gt_np > 0

            if valid.any():
                vmin, vmax = np.percentile(gt_np[valid], [2, 98])
            else:
                vmin, vmax = 0.0, float(cfg.bins.max_depth)

            gt_panel = _colorize(gt_np, valid, vmin, vmax, cmap)
            pred_panel = _colorize(pred_np, np.ones_like(valid), vmin, vmax, cmap)

            err = np.abs(pred_np - gt_np)
            err_vmax = float(np.percentile(err[valid], 95)) if valid.any() else 1.0
            err_panel = _colorize(err, valid, 0.0, max(err_vmax, 1e-6), err_cmap)

            seq_short = str(seq).replace("/", "-")
            strip = _hstack([rgb_np, gt_panel, pred_panel, err_panel])
            writer.add_image(f"{args.split}/{written:03d}_{seq_short}_rgb-gt-pred-err", strip, global_step=0)
            seq_count[seq] = seq_count.get(seq, 0) + 1
            written += 1

    writer.close()
    log.info("done: wrote %d strips across %d sequences. Panel order per strip: RGB | GT depth | "
             "predicted depth | abs error. GT and prediction share the same colour scale; invalid "
             "GT pixels are black.", written, len(seq_count))


if __name__ == "__main__":
    main()
