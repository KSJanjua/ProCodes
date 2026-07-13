"""Phase 2 evaluation: overall mask precision/recall/IoU + depth-layer MAE,
**and** the same metrics restricted to the occlusion-focused slice (frames
with >=2 overlapping GT instances). This slice is the primary signal for
whether the redesign actually fixed
the wrong-ownership/fragmentation problems, since whole-dataset averages
would dilute exactly the cases this rebuild targets.

Usage:

    python -m instancedepth.engine.evaluate_phase2 \\
        --config instancedepth/configs/phase2_mask2former.yaml \\
        --checkpoint runs/phase2_mask2former/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from instancedepth.configs.phase2_config import Phase2Config
from instancedepth.engine.train_phase2 import build_dataloader
from instancedepth.models.phase2.model import Phase2Model
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.phase2_metrics import aggregate, evaluate_frame, has_overlapping_instances

log = logging.getLogger("instancedepth.engine.evaluate_phase2")

_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@torch.no_grad()
def evaluate(
    model: Phase2Model,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    max_batches: Optional[int] = None,
    precision: str = "fp32",
) -> Dict[str, Dict[str, float]]:
    model.eval()
    all_frames, occlusion_frames = [], []

    # Match Trainer.train_step's precision (see evaluate_hdi.py's identical
    # fix) -- forwarding a Swin-L Mask2Former + ViT-scale query heads in
    # full fp32 during eval, while training runs bf16/fp16, would double
    # eval's activation memory for no reason and can OOM a process that
    # trained cleanly for many steps.
    use_autocast = precision != "fp32" and device.type == "cuda"

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        image = batch["image"].to(device)
        with torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            out = model(image)
        scores = out.scores()             # (B, N)
        mask_probs = out.mask_confidence()  # (B, N, H, W)

        for b in range(image.shape[0]):
            keep = scores[b] > score_threshold
            pred_masks = (mask_probs[b, keep] > mask_threshold)
            pred_scores = scores[b, keep]
            pred_depths = out.depth_layers[b, keep]

            tgt = batch["targets"][b]
            gt_masks = tgt["masks"].bool().to(device)
            gt_depths = tgt["depths"].to(device)

            frame_result = evaluate_frame(pred_masks, pred_scores, pred_depths, gt_masks, gt_depths, iou_threshold)
            all_frames.append(frame_result)
            if has_overlapping_instances(gt_masks):
                occlusion_frames.append(frame_result)

    return dict(
        overall=aggregate(all_frames),
        occlusion_slice=aggregate(occlusion_frames) if occlusion_frames else
        dict(precision=float("nan"), recall=float("nan"), f1=float("nan"),
             mean_iou=float("nan"), depth_mae=float("nan"), num_frames=0),
    )


def make_eval_fn(cfg: Phase2Config, max_batches: int = 50):
    val_loader = build_dataloader(cfg, split="test")

    def eval_fn(model: torch.nn.Module) -> Dict[str, float]:
        device = next(model.parameters()).device
        result = evaluate(model, val_loader, device, max_batches=max_batches, precision=cfg.optim.precision)
        # flatten for TensorBoard scalar logging (Trainer expects Dict[str, float])
        flat = {f"overall_{k}": v for k, v in result["overall"].items()}
        flat.update({f"occlusion_{k}": v for k, v in result["occlusion_slice"].items()})
        # Trainer's best-checkpoint selection looks for "abs_rel" (Phase 1
        # convention); Phase 2's analogous "lower is better" signal is
        # depth_mae -- exposed under both names so Trainer's generic logic
        # (instancedepth/engine/trainer.py) works unmodified.
        flat["abs_rel"] = flat.get("overall_depth_mae", float("nan"))
        return flat

    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = Phase2Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Phase2Model(
        checkpoint=cfg.model.checkpoint, checkpoint_dir=cfg.model.checkpoint_dir,
        allow_hub_download=cfg.model.allow_hub_download, num_classes=cfg.model.num_classes,
    ).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=str(device), restore_rng=False)

    loader = build_dataloader(cfg, split=args.split)
    result = evaluate(model, loader, device, args.score_threshold, args.mask_threshold,
                       args.iou_threshold, args.max_batches, precision=cfg.optim.precision)

    log.info("Evaluation results (%s split, %s):\n%s", args.split, args.checkpoint, json.dumps(result, indent=2))
    out_path = Path(cfg.run_root) / cfg.run_name / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
