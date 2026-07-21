"""Paired significance test over per-sequence Phase-3 occlusion abs_rel.

Consumes the per-sequence JSONs from scripts.eval_phase3_per_sequence and tests
whether a depth improvement is beyond per-sequence noise. Sequences are the
independent unit (frames within a sequence are correlated), so the test pairs
across sequences. Lower abs_rel is better; a POSITIVE delta = improvement.

Modes:
  within  : base vs refined within ONE run    -> does refinement help?  (G2)
  between : refined of run A vs refined of run B, matched by sequence id
            -> is the bounded head better than the vanilla relation head?  (G3)

Reports n sequences, mean & median per-sequence delta, 95% bootstrap CI,
one-sided bootstrap p (P[mean delta <= 0]), an exact sign-test p, and the count
of sequences improved. Pure numpy/stdlib -- no GPU, no scipy.

    python -m scripts.paired_significance within  --file results/phase3_video_dav2_per_sequence.json
    python -m scripts.paired_significance between --file-a results/phase3_relation_dav2_per_sequence.json \
                                                  --file-b results/phase3_video_dav2_per_sequence.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _sign_test_p(improved: int, total: int) -> float:
    """Two-sided exact binomial p vs H0: P(improve)=0.5 (ties excluded upstream)."""
    if total == 0:
        return float("nan")
    k = min(improved, total - improved)
    tail = sum(math.comb(total, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** total))


def _bootstrap(deltas: np.ndarray, n_boot: int = 20000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_one_sided = float((means <= 0.0).mean())     # P[mean improvement <= 0]
    return float(lo), float(hi), p_one_sided


def _report(name: str, deltas: np.ndarray, weights: np.ndarray) -> None:
    n = len(deltas)
    improved = int((deltas > 0).sum())
    tied = int((deltas == 0).sum())
    mean_d, med_d = float(deltas.mean()), float(np.median(deltas))
    frame_weighted = float(np.average(deltas, weights=weights)) if weights.sum() else float("nan")
    lo, hi, p_boot = _bootstrap(deltas)
    p_sign = _sign_test_p(improved, n - tied)

    print(f"\n=== {name} ===")
    print(f"sequences (paired)          : {n}")
    print(f"improved / tied / worse     : {improved} / {tied} / {n - improved - tied}")
    print(f"mean per-seq delta          : {mean_d:+.5f}   (positive = improvement)")
    print(f"median per-seq delta        : {med_d:+.5f}")
    print(f"frame-weighted mean delta   : {frame_weighted:+.5f}")
    print(f"95% bootstrap CI (mean)     : [{lo:+.5f}, {hi:+.5f}]")
    print(f"one-sided bootstrap p       : {p_boot:.4f}   (P[mean improvement <= 0])")
    print(f"exact sign-test p (2-sided) : {p_sign:.4f}")
    verdict = "significant at 0.05" if (lo > 0 and p_sign < 0.05) else "NOT significant at 0.05"
    print(f"verdict                     : {verdict}")


def _load_seqs(path: str) -> Dict[str, Dict]:
    return json.loads(Path(path).read_text())["sequences"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    pw = sub.add_parser("within", help="base vs refined in one run (G2)")
    pw.add_argument("--file", required=True)
    pb = sub.add_parser("between", help="run A refined vs run B refined (G3)")
    pb.add_argument("--file-a", required=True, help="baseline run (e.g. vanilla relation head)")
    pb.add_argument("--file-b", required=True, help="candidate run (e.g. bounded head)")
    pb.add_argument("--field", default="refined_abs_rel")
    args = ap.parse_args()

    if args.mode == "within":
        seqs = _load_seqs(args.file)
        ids = sorted(seqs)
        deltas = np.array([seqs[s]["base_abs_rel"] - seqs[s]["refined_abs_rel"] for s in ids])
        weights = np.array([seqs[s]["n_occ_frames"] for s in ids], dtype=float)
        _report(f"WITHIN: base - refined  ({Path(args.file).name})", deltas, weights)
    else:
        a, b = _load_seqs(args.file_a), _load_seqs(args.file_b)
        ids = sorted(set(a) & set(b))
        if not ids:
            raise SystemExit("no shared sequence ids between the two files")
        deltas = np.array([a[s][args.field] - b[s][args.field] for s in ids])   # A - B, positive = B better
        weights = np.array([min(a[s]["n_occ_frames"], b[s]["n_occ_frames"]) for s in ids], dtype=float)
        _report(f"BETWEEN: {Path(args.file_a).name} - {Path(args.file_b).name}  [{args.field}]", deltas, weights)


if __name__ == "__main__":
    main()
