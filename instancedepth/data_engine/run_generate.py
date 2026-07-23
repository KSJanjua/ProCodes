"""
run_generate.py is the one command that runs the whole data engine. It discovers all sequences, builds SAM3 once and reuses it, annotates every video, then splits into train/test at the video level — 20% to test, stratified by batch and seeded so it's reproducible and there's no frame leakage between splits. It also computes the paper's Figure-3 dataset statistics — object counts per depth bucket and per-video averages — and writes train.txt, test.txt, and meta.json. Those three plus the per-sequence files are exactly what the training Dataset reads.

Split + statistics + CLI for the GID-style data engine.

split_sequences : video-level split with a 20% test fraction (paper Sec. 3
                  assigns "a larger proportion (20%) to the test set"),
                  stratified by Batch so every recording condition appears in
                  both splits.  [Paper Specified: fraction & video-level;
                  Reasonable Assumption: batch stratification.]
compute_statistics : Fig. 3 analogs — object counts per depth-range bucket
                  (3a) and per-video (avg objects, num frames) scatter data (3b).
main            : `python -m instancedepth.data_engine.run_generate --config cfg.yaml`

Usage: python -m instancedepth.data_engine.run_generate --config instancedepth/configs/gid_custom.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .annotate import annotate_sequence
from .config import DataEngineConfig
from .discover import SequenceRecord, discover_dataset
from .sam3_engine import build_segmenter

log = logging.getLogger("data_engine.run")

# decides which whole videos go to test (default 20%). Done at the video level. it takes 20% from each batch, so every recording condition appears in both splits
# --------------------------------------------------------------------------- #
def split_sequences(
    sequences: List[SequenceRecord], cfg: DataEngineConfig
) -> Tuple[List[str], List[str]]:
    rng = random.Random(cfg.split.seed)
    train_ids: List[str] = []
    test_ids: List[str] = []

    if cfg.split.stratify_by_batch:
        by_batch: Dict[str, List[SequenceRecord]] = {}
        for s in sequences:
            by_batch.setdefault(s.batch, []).append(s)
        groups = list(by_batch.values())
    else:
        groups = [list(sequences)]

    for group in groups:
        ids = sorted(s.seq_id for s in group)
        rng.shuffle(ids)
        n_test = max(1, round(cfg.split.test_fraction * len(ids))) if len(ids) > 1 else 0
        test_ids += ids[:n_test]
        train_ids += ids[n_test:]
    return sorted(train_ids), sorted(test_ids)

# how many objects fall in each 2 m depth bucket (0–2 m, 2–4 m, …), and per-video stats (frames, tracks, avg objects/frame), plus totals and averages.
# --------------------------------------------------------------------------- #
def compute_statistics(manifests: List[Dict], bucket_m: float = 2.0) -> Dict:
    """Depth-range population (Fig. 3a) and per-video object/frame stats (Fig. 3b)."""
    depth_buckets: Counter = Counter()
    per_video = []
    for man in manifests:
        counts = []
        for fr in man["frames"].values():
            counts.append(len(fr["instances"]))
            for inst in fr["instances"]:
                d = inst["depth_layer_m"]
                if d > 0:
                    lo = int(d // bucket_m) * bucket_m
                    depth_buckets[f"{lo:.0f}-{lo + bucket_m:.0f}m"] += 1
        per_video.append(dict(
            sequence=man["sequence"],
            num_frames=man["num_frames"],
            num_tracks=man["num_tracks"],
            avg_objects_per_frame=round(sum(counts) / max(len(counts), 1), 2),
        ))
    return dict(
        depth_range_object_counts=dict(sorted(depth_buckets.items())),
        per_video=per_video,
        total_frames=sum(m["num_frames"] for m in manifests),
        total_videos=len(manifests),
        avg_video_len=round(
            sum(m["num_frames"] for m in manifests) / max(len(manifests), 1), 1
        ),
        avg_objects=round(
            sum(v["avg_objects_per_frame"] for v in per_video) / max(len(per_video), 1), 2
        ),
    )

# discover_dataset → build_segmenter (once) → loop annotate_sequence over every video (in a try/finally that always closes the segmenter) → split_sequences → write train.txt/test.txt → compute_statistics → write meta.json. Returns the meta dict.
# --------------------------------------------------------------------------- #
def generate(cfg: DataEngineConfig, limit: int | None = None) -> Dict:
    sequences = discover_dataset(cfg)
    if limit:
        sequences = sequences[:limit]
    # Building SAM3 is expensive (loads a big model onto the GPU), so it's created once and reused for every video
    segmenter = build_segmenter(cfg.sam3)
    manifests: List[Dict] = []
    try:
        for i, seq in enumerate(sequences):
            log.info("=== [%d/%d] %s (%d frames) ===", i + 1, len(sequences),
                     seq.seq_id, len(seq))
            manifests.append(annotate_sequence(seq, cfg, segmenter, segmenter))
    finally:
        segmenter.close()

    train_ids, test_ids = split_sequences(sequences, cfg)
    out_root = Path(cfg.output.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train.txt").write_text("\n".join(train_ids) + "\n")
    (out_root / "test.txt").write_text("\n".join(test_ids) + "\n")

    stats = compute_statistics(manifests)
    meta = dict(
        categories={p: cid for p, cid in cfg.category_ids.items()},
        max_depth_m=cfg.depth.max_depth_m,
        num_train=len(train_ids),
        num_test=len(test_ids),
        statistics=stats,
    )
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    log.info("Done: %d videos, %d frames, train/test = %d/%d.",
             stats["total_videos"], stats["total_frames"], len(train_ids), len(test_ids))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GID-style annotations with SAM3.")
    ap.add_argument("--config", required=True, help="YAML DataEngineConfig")
    ap.add_argument("--limit", type=int, default=None, help="annotate first N sequences")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    generate(DataEngineConfig.from_yaml(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
