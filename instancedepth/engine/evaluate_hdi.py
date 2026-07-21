"""Phase 1 evaluation: paper Table 2/3 metrics (RMS/REL/RMSlog/Log10/
sigma1-3) on the held-out test split, plus optional disparity diagnostics
.

Usage (on the Backend.AI server):

    python -m instancedepth.engine.evaluate_hdi \\
        --config instancedepth/configs/hdi.yaml \\
        --checkpoint runs/hdi_faithful/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.config import HDIConfig
from instancedepth.engine.train_hdi import build_dataloader
from instancedepth.models.hdi.model import HolisticDepthModel
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.metrics import compute_depth_metrics, compute_disparity_diagnostics

log = logging.getLogger("instancedepth.engine.evaluate_hdi")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@torch.no_grad()
def evaluate(
    model: HolisticDepthModel,
    loader: DataLoader,
    device: torch.device,
    cfg: HDIConfig,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    disp_totals: Dict[str, float] = {}
    n_batches = 0
    n_disp_batches = 0

    # Match Trainer.train_step's precision exactly (trainer.py's
    # _autocast_ctx) -- without this, the forward pass here runs in full
    # fp32 while training runs it in bf16/fp16, roughly doubling this
    # model's activation memory for the same batch size/resolution. That
    # mismatch is what OOM'd on the very first periodic eval call even
    # though training itself had already run thousands of steps cleanly.
    precision = cfg.optim.precision
    use_autocast = precision != "fp32" and device.type == "cuda"

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Batches hold independent (shuffle-ordered) frames: a temporal model
        # must treat each as a length-1 sequence, not carry state across them.
        if hasattr(model, "reset_temporal_state"):
            model.reset_temporal_state()
        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        with torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            out = model(image)
        mask = gt_depth > 0

        m = compute_depth_metrics(out.depth_final, gt_depth, mask)
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        disp_m = compute_disparity_diagnostics(out.depth_final, gt_depth, mask, cfg.camera)
        if disp_m is not None:
            for k, v in disp_m.items():
                disp_totals[k] = disp_totals.get(k, 0.0) + v
            n_disp_batches += 1

    result = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if n_disp_batches:
        result.update({k: v / n_disp_batches for k, v in disp_totals.items()})
    else:
        log.info("Camera intrinsics not set (instancedepth/configs/*.yaml -> camera.*) -- "
                  "skipping disparity diagnostics (this never blocks the standard metrics above).")
    return result


@torch.no_grad()
def evaluate_streaming(model: HolisticDepthModel, cfg: HDIConfig, device: torch.device,
                       split: str = "test", max_frames: Optional[int] = None) -> Dict[str, float]:
    """Sequence-ordered, stateful evaluation:
    frames processed one at a time in order, temporal state carried within
    each sequence and reset at sequence boundaries. Reports the standard
    dense metrics under streaming plus the temporal-alignment-error (TAE)
    diagnostic -- the metric the temporal module is meant to improve. Also
    meaningful for a per-frame model (its TAE is the flicker baseline)."""
    from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset
    from instancedepth.utils.metrics import temporal_alignment_error

    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=cfg.data.annotations_root, split=split,
        image_size=cfg.data.image_size, min_instance_px=cfg.data.min_instance_px,
        hflip_prob=0.0, color_jitter=0.0,
    ))
    model.eval()
    precision = cfg.optim.precision
    use_autocast = precision != "fp32" and device.type == "cuda"

    totals: Dict[str, float] = {}
    n = 0
    tae_sum, tae_n = 0.0, 0
    gt_d_sum = pred_d_sum = 0.0
    prev_seq = None
    prev_pred = prev_gt = None

    for i in range(len(ds) if max_frames is None else min(len(ds), max_frames)):
        sample = ds[i]
        seq = sample["meta"]["sequence"]
        if seq != prev_seq:
            if hasattr(model, "reset_temporal_state"):
                model.reset_temporal_state()
            prev_seq, prev_pred, prev_gt = seq, None, None

        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["depth"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            out = model(image)
        pred = out.depth_final.float()
        mask = gt > 0

        for k, v in compute_depth_metrics(pred, gt, mask).items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

        if prev_pred is not None:
            both_valid = mask & (prev_gt > 0)
            tae = temporal_alignment_error(pred, prev_pred, gt, prev_gt, both_valid)
            if tae == tae:   # not NaN
                tae_sum += tae
                tae_n += 1
                # Context for TAE, without which it is uninterpretable:
                #   gt_temporal_delta   = mean|Δgt|   -- how much GT itself moves
                #     frame to frame (real motion + sensor noise). It is also the
                #     TAE a *constant* predictor would score, i.e. the trivial
                #     upper reference.
                #   pred_temporal_delta = mean|Δpred| -- how much the prediction
                #     moves. Compare the two: a prediction that tracks GT has
                #     pred≈gt delta AND low TAE; one that flickers has
                #     pred delta > gt delta.
                # If TAE ≈ gt_temporal_delta, the prediction's frame-to-frame
                # changes carry no information about GT's -- the metric is then
                # dominated by GT noise and cannot resolve real improvements.
                gt_d_sum += float((gt - prev_gt).abs()[both_valid].mean())
                pred_d_sum += float((pred - prev_pred).abs()[both_valid].mean())
        prev_pred, prev_gt = pred, gt

    result = {k: v / max(n, 1) for k, v in totals.items()}
    result["temporal_alignment_error"] = tae_sum / max(tae_n, 1)
    result["gt_temporal_delta"] = gt_d_sum / max(tae_n, 1)
    result["pred_temporal_delta"] = pred_d_sum / max(tae_n, 1)
    result["num_frames"] = n
    return result


def make_eval_fn(cfg: HDIConfig, max_batches: int = 50):
    """Factory used by ``Trainer`` for periodic in-training eval (a cheap
    subset, not the full held-out set)."""
    val_loader = build_dataloader(cfg, split="test")

    def eval_fn(model: torch.nn.Module) -> Dict[str, float]:
        device = next(model.parameters()).device
        return evaluate(model, val_loader, device, cfg, max_batches=max_batches)

    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-batches", type=int, default=None, help="default: evaluate the full split")
    ap.add_argument("--streaming", action="store_true",
                    help="sequence-ordered stateful eval (adds temporal_alignment_error); "
                         "the intended mode for temporal models, also gives the flicker "
                         "baseline for per-frame models")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = HDIConfig.from_yaml_with_overrides(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HolisticDepthModel(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)

    if args.streaming:
        max_frames = args.max_batches * cfg.optim.batch_size if args.max_batches else None
        metrics = evaluate_streaming(model, cfg, device, split=args.split, max_frames=max_frames)
    else:
        loader = build_dataloader(cfg, split=args.split)
        metrics = evaluate(model, loader, device, cfg, max_batches=args.max_batches)

    log.info("Evaluation results (%s split, %s):\n%s", args.split, args.checkpoint, json.dumps(metrics, indent=2))
    suffix = "_streaming" if args.streaming else ""
    out_path = Path(cfg.run_root) / cfg.run_name / f"eval_{args.split}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
