"""Batch inference over held-out sequences -> sharded on-disk artifacts for
downstream (batch eval / visualization) use. This is the *secondary*
integration path (plan SS6/SS10); Phase 2/3 training itself should prefer
``instancedepth.models.hdi.inference.HDIInferencer`` directly (avoids
persisting a multi-hundred-GB dense feature cache).

Serialization, one file **per sequence** (not per frame -- ~297 files, not
~55,000):

    <out_root>/<batch>/<sequence>.npz
        depth_final : (T,H,W) float16, meters
        seg_argmax  : (T,H,W) uint8      -- argmax bin index of seg_final
                                             (dense soft seg_final is NOT
                                             saved by default; pass
                                             --save-dense-seg to keep it)
        [seg_final] : (T,rd,H,W) float16 -- only if --save-dense-seg

    <out_root>/manifest.json
        {contract_version, produced_by: {config_hash, checkpoint, git_commit},
         sequences: [{sequence, batch, name, num_frames, path, image_hw}]}

Usage:

    python -m scripts.infer_sequence \\
        --config instancedepth/configs/hdi.yaml \\
        --checkpoint runs/hdi_faithful/best.pth \\
        --annotations-root gid_custom \\
        --split test \\
        --out-root hdi_artifacts/hdi_faithful
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from instancedepth.configs.config import HDIConfig
from instancedepth.data.gid_dataset import GIDInstanceDepthDataset
from instancedepth.models.hdi.inference import HDIInferencer
from instancedepth.models.hdi.output import CONTRACT_VERSION
from instancedepth.utils.manifest import git_commit_hash  # reuse, not duplicate (plan: avoid duplicated logic)

log = logging.getLogger("scripts.infer_sequence")


def _load_rgb(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read rgb {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def infer_one_sequence(
    inferencer: HDIInferencer, manifest: dict, save_dense_seg: bool
) -> dict:
    frame_keys = sorted(manifest["frames"].keys())
    depth_stack: List[np.ndarray] = []
    seg_argmax_stack: List[np.ndarray] = []
    seg_dense_stack: List[np.ndarray] = []

    for fk in frame_keys:
        rgb = _load_rgb(manifest["frames"][fk]["rgb"])
        out = inferencer.predict(rgb)
        depth_stack.append(out.depth_final[0, 0].cpu().numpy().astype(np.float16))
        seg = out.seg_final[0].cpu().numpy()   # (rd,H,W)
        seg_argmax_stack.append(seg.argmax(axis=0).astype(np.uint8))
        if save_dense_seg:
            seg_dense_stack.append(seg.astype(np.float16))

    payload = dict(
        depth_final=np.stack(depth_stack, axis=0),
        seg_argmax=np.stack(seg_argmax_stack, axis=0),
    )
    if save_dense_seg:
        payload["seg_final"] = np.stack(seg_dense_stack, axis=0)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--annotations-root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--save-dense-seg", action="store_true",
                    help="also save the full (rd,H,W) soft seg_final, not just the argmax bin index")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N sequences (debugging)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = HDIConfig.from_yaml(args.config)
    inferencer = HDIInferencer(cfg, args.checkpoint)

    ann_root = Path(args.annotations_root)
    seq_ids = [s for s in (ann_root / f"{args.split}.txt").read_text().splitlines() if s.strip()]
    if args.limit:
        seq_ids = seq_ids[: args.limit]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for i, sid in enumerate(seq_ids):
        man_path = ann_root / sid / "annotations.json"
        with open(man_path) as f:
            man = json.load(f)
        log.info("[%d/%d] %s (%d frames)", i + 1, len(seq_ids), sid, man["num_frames"])

        payload = infer_one_sequence(inferencer, man, args.save_dense_seg)
        out_path = out_root / f"{sid.replace('/', '__')}.npz"
        np.savez_compressed(out_path, **payload)

        manifest_entries.append(dict(
            sequence=sid, batch=man["batch"], name=man["name"],
            num_frames=man["num_frames"], path=str(out_path), image_hw=man["image_hw"],
        ))

    manifest = dict(
        contract_version=CONTRACT_VERSION,
        produced_by=dict(
            config=args.config,
            checkpoint=args.checkpoint,
            git=git_commit_hash(Path(__file__).resolve().parents[1]),
        ),
        sequences=manifest_entries,
    )
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote %d sequence artifacts + manifest to %s", len(manifest_entries), out_root)


if __name__ == "__main__":
    main()
