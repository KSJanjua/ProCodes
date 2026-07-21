"""Per-sequence Phase-3 occlusion-slice abs_rel, for significance testing.

Mirrors instancedepth/engine/evaluate_phase3.py's occlusion-slice definition
(frames with >=2 overlapping GT instances; metrics over instance-union pixels)
but records base and refined abs_rel PER SEQUENCE. Sequences are the
independent unit for significance (frames within a sequence are correlated),
so downstream tests pair across sequences, not frames.

The head is auto-selected from the checkpoint (BoundedPairAttentionHead vs the
MLP Phi_o), so this evaluates both the vanilla and bounded runs unchanged.

Writes results/<run_name>_per_sequence.json:
    {"sequences": {seq_id: {base_abs_rel, refined_abs_rel, n_occ_frames}, ...},
     "aggregate": {occ_base_abs_rel, occ_refined_abs_rel, n_occ_frames, n_sequences}}

Run on the server (needs GPU + data):
    python -m scripts.eval_phase3_per_sequence \
        --config videodepth/configs/phase3_dav2_p2run.yaml \
        --checkpoint runs/phase3_video_dav2/best.pth --run-name phase3_video_dav2
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from instancedepth.configs.phase3_config import Phase3Config
from instancedepth.engine.train_phase3 import build_dataloader
from instancedepth.models.phase3.model import Phase3Model
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.metrics import compute_depth_metrics
from instancedepth.utils.phase2_metrics import has_overlapping_instances

log = logging.getLogger("eval_phase3_per_sequence")
_PRECISION_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def build_model(cfg: Phase3Config, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    raw = torch.load(str(ckpt_path), map_location="cpu")
    sd = raw.get("model", raw)
    from videodepth.models.phase3_video import is_bounded_relation_head_checkpoint
    if is_bounded_relation_head_checkpoint(sd.keys()):
        from videodepth.models.phase3_video import Phase3VideoModel
        model = Phase3VideoModel(cfg)
        log.info("checkpoint uses BoundedPairAttentionHead (bounded run)")
    else:
        model = Phase3Model(cfg)
        log.info("checkpoint uses MLP Phi_o (vanilla relation-head run)")
    model = model.to(device)
    load_checkpoint(ckpt_path, model, map_location=str(device), restore_rng=False)
    return model


@torch.no_grad()
def evaluate_per_sequence(model, loader, device, precision) -> Dict[str, Dict]:
    model.eval()
    use_autocast = precision != "fp32" and device.type == "cuda"
    base: Dict[str, List[float]] = defaultdict(list)
    refined: Dict[str, List[float]] = defaultdict(list)

    for batch in loader:
        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        with torch.autocast(device_type=device.type, dtype=_PRECISION_DTYPE[precision], enabled=use_autocast):
            output, _ = model(image)
        ref = output.refined_depth.float()
        bas = output.base_depth.float()

        for b in range(image.shape[0]):
            gt_masks = batch["targets"][b]["masks"].bool().to(device)
            if gt_masks.shape[0] < 2 or not has_overlapping_instances(gt_masks):
                continue
            gt = gt_depth[b:b + 1]
            inst_valid = (gt > 0) & gt_masks.any(0)[None, None]     # (1,1,H,W) instance-union, valid
            rb = compute_depth_metrics(bas[b:b + 1], gt, inst_valid)["abs_rel"]
            rr = compute_depth_metrics(ref[b:b + 1], gt, inst_valid)["abs_rel"]
            if rb == rb and rr == rr:                               # drop NaN (no valid px)
                seq = batch["meta"][b]["sequence"]
                base[seq].append(rb)
                refined[seq].append(rr)

    seqs = {}
    tot_b = tot_r = tot_n = 0.0
    for seq in sorted(base):
        bs, rs = base[seq], refined[seq]
        seqs[seq] = {"base_abs_rel": sum(bs) / len(bs),
                     "refined_abs_rel": sum(rs) / len(rs),
                     "n_occ_frames": len(bs)}
        tot_b += sum(bs); tot_r += sum(rs); tot_n += len(bs)
    aggregate = {"occ_base_abs_rel": tot_b / max(tot_n, 1),
                 "occ_refined_abs_rel": tot_r / max(tot_n, 1),
                 "n_occ_frames": int(tot_n), "n_sequences": len(seqs)}
    return {"sequences": seqs, "aggregate": aggregate}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-name", required=True, help="output file stem: results/<run-name>_per_sequence.json")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = Phase3Config.from_yaml_with_overrides(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, Path(args.checkpoint), device)
    loader = build_dataloader(cfg, split=args.split)
    result = evaluate_per_sequence(model, loader, device, cfg.optim.precision)

    log.info("aggregate: %s", json.dumps(result["aggregate"], indent=2))
    out_path = Path("results") / f"{args.run_name}_per_sequence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("wrote %s (%d sequences)", out_path, result["aggregate"]["n_sequences"])


if __name__ == "__main__":
    main()
