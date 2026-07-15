#!/usr/bin/env bash
# Full InstanceDepth pipeline, unattended and fault-tolerant:
#
#   Phase 1  train -> eval (+streaming/TAE) -> viz -> videos
#   Phase 1b temporal fine-tune (optional)  -> streaming eval -> videos
#   Phase 2  train -> eval -> viz
#   Phase 3  train -> eval -> viz -> videos
#
# One command:   bash scripts/run_full_pipeline.sh
#
# Fault tolerance: TRAINING steps are critical -- a phase's training failure
# skips only that phase's own downstream steps and any phase that needs its
# checkpoint (Phase 3 needs BOTH Phase 1 and Phase 2; Phase 2 is independent
# of Phase 1, so it runs regardless). Eval / visualization / video steps are
# non-critical: their failures are recorded and everything else continues.
# A pass/fail summary prints at the end; exit code is non-zero if anything failed.
#
# Configuration via environment variables (defaults in parentheses):
#   P1_CONFIG (hdi_enhanced.yaml)  P2_CONFIG (phase2_mask2former.yaml)
#   P3_CONFIG (phase3_current.yaml)  TEMPORAL (1: run the temporal stage)
#   TEMPORAL_CONFIG (hdi_temporal.yaml)  VIDEO_SEQS (4: sequences per video step)
#
# Eval JSONs land in each run's folder (runs/<run_name>/eval_*.json) and are
# copied into results/ at the end for the record.

set -uo pipefail
cd "$(dirname "$0")/.."

P1_CONFIG=${P1_CONFIG:-instancedepth/configs/hdi_enhanced.yaml}
P2_CONFIG=${P2_CONFIG:-instancedepth/configs/phase2_mask2former.yaml}
P3_CONFIG=${P3_CONFIG:-instancedepth/configs/phase3_current.yaml}
TEMPORAL=${TEMPORAL:-1}
TEMPORAL_CONFIG=${TEMPORAL_CONFIG:-instancedepth/configs/hdi_temporal.yaml}
VIDEO_SEQS=${VIDEO_SEQS:-4}

run_name() { python -c "import yaml,sys; print(yaml.safe_load(open('$1'))['run_name'])"; }
P1_RUN=$(run_name "$P1_CONFIG"); P2_RUN=$(run_name "$P2_CONFIG"); P3_RUN=$(run_name "$P3_CONFIG")
TEMPORAL_RUN=$(run_name "$TEMPORAL_CONFIG")

declare -a PASSED FAILED SKIPPED
banner() { echo -e "\n########## $(date '+%Y-%m-%d %H:%M:%S')  $1 ##########\n"; }

# critical: on failure, record and return 1 so callers can gate dependents
critical() { local name=$1; shift; banner "$name"
    if "$@"; then PASSED+=("$name"); return 0
    else FAILED+=("$name (CRITICAL)"); return 1; fi; }
# optional: on failure, record and continue
optional() { local name=$1; shift; banner "$name"
    if "$@"; then PASSED+=("$name"); else FAILED+=("$name (non-critical)"); fi; return 0; }
skip() { SKIPPED+=("$1"); echo "SKIP: $1"; }

# =========================== PHASE 1 =========================================
P1_OK=0
if critical "P1 train" python -m instancedepth.engine.train_hdi --config "$P1_CONFIG"; then
    P1_OK=1
    optional "P1 eval" python -m instancedepth.engine.evaluate_hdi \
        --config "$P1_CONFIG" --checkpoint "runs/$P1_RUN/best.pth"
    optional "P1 eval (streaming/TAE)" python -m instancedepth.engine.evaluate_hdi \
        --config "$P1_CONFIG" --checkpoint "runs/$P1_RUN/best.pth" --streaming
    optional "P1 viz" python -m scripts.visualize_hdi \
        --config "$P1_CONFIG" --checkpoint "runs/$P1_RUN/best.pth"
    optional "P1 videos" python -m scripts.make_sequence_videos \
        --phase 1 --config "$P1_CONFIG" --checkpoint "runs/$P1_RUN/best.pth" \
        --out-dir "videos/$P1_RUN" --include-gt --limit-seq "$VIDEO_SEQS"
else
    skip "P1 eval/viz/videos (P1 training failed)"
fi

# ======================= PHASE 1b: TEMPORAL ==================================
if [ "$TEMPORAL" = "1" ]; then
    if [ "$P1_OK" = "1" ] || [ -f "runs/$P1_RUN/best.pth" ]; then
        if critical "P1b temporal train" python -m instancedepth.engine.train_hdi_temporal \
                --config "$TEMPORAL_CONFIG" \
                --override "temporal.init_checkpoint=runs/$P1_RUN/best.pth"; then
            optional "P1b eval (streaming/TAE)" python -m instancedepth.engine.evaluate_hdi \
                --config "$TEMPORAL_CONFIG" --checkpoint "runs/$TEMPORAL_RUN/best.pth" --streaming
            optional "P1b videos" python -m scripts.make_sequence_videos \
                --phase 1 --config "$TEMPORAL_CONFIG" --checkpoint "runs/$TEMPORAL_RUN/best.pth" \
                --out-dir "videos/$TEMPORAL_RUN" --include-gt --limit-seq "$VIDEO_SEQS"
        else
            skip "P1b eval/videos (temporal training failed)"
        fi
    else
        skip "P1b temporal stage (no Phase-1 checkpoint)"
    fi
fi

# =========================== PHASE 2 =========================================
# Independent branch: runs even if Phase 1 failed.
P2_OK=0
if critical "P2 train" python -m instancedepth.engine.train_phase2 --config "$P2_CONFIG"; then
    P2_OK=1
    optional "P2 eval" python -m instancedepth.engine.evaluate_phase2 \
        --config "$P2_CONFIG" --checkpoint "runs/$P2_RUN/best.pth"
    optional "P2 viz" python -m scripts.visualize_phase2 \
        --config "$P2_CONFIG" --checkpoint "runs/$P2_RUN/best.pth"
else
    skip "P2 eval/viz (P2 training failed)"
fi

# =========================== PHASE 3 =========================================
if [ -f "runs/$P1_RUN/best.pth" ] && [ -f "runs/$P2_RUN/best.pth" ]; then
    P3_OVERRIDES=("phase1_checkpoint=runs/$P1_RUN/best.pth" "phase2_checkpoint=runs/$P2_RUN/best.pth")
    if critical "P3 train" python -m instancedepth.engine.train_phase3 \
            --config "$P3_CONFIG" --override "${P3_OVERRIDES[@]}"; then
        optional "P3 eval" python -m instancedepth.engine.evaluate_phase3 \
            --config "$P3_CONFIG" --override "${P3_OVERRIDES[@]}" \
            --checkpoint "runs/$P3_RUN/best.pth"
        optional "P3 viz" python -m scripts.visualize_phase3 \
            --config "$P3_CONFIG" --override "${P3_OVERRIDES[@]}" \
            --checkpoint "runs/$P3_RUN/best.pth" --out-dir "viz/$P3_RUN"
        optional "P3 videos" python -m scripts.make_sequence_videos \
            --phase 3 --config "$P3_CONFIG" --override "${P3_OVERRIDES[@]}" \
            --checkpoint "runs/$P3_RUN/best.pth" \
            --out-dir "videos/$P3_RUN" --include-gt --limit-seq "$VIDEO_SEQS"
    else
        skip "P3 eval/viz/videos (P3 training failed)"
    fi
else
    skip "P3 entirely (needs both runs/$P1_RUN/best.pth and runs/$P2_RUN/best.pth)"
fi

# ====================== RESULTS COLLECTION + SUMMARY =========================
optional "collect eval JSONs into results/" bash -c '
    mkdir -p results
    for run in "'"$P1_RUN"'" "'"$TEMPORAL_RUN"'" "'"$P2_RUN"'" "'"$P3_RUN"'"; do
        for f in runs/$run/eval*.json; do
            [ -f "$f" ] && cp "$f" "results/${run}_$(basename "$f")"
        done
    done; true'

banner "PIPELINE SUMMARY"
printf 'PASSED (%d):\n' "${#PASSED[@]}";  printf '  + %s\n' "${PASSED[@]:-none}"
printf 'FAILED (%d):\n' "${#FAILED[@]}";  printf '  - %s\n' "${FAILED[@]:-none}"
printf 'SKIPPED (%d):\n' "${#SKIPPED[@]}"; printf '  ~ %s\n' "${SKIPPED[@]:-none}"
[ "${#FAILED[@]}" -eq 0 ]
