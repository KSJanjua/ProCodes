#!/usr/bin/env bash
# Full videodepth pipeline, one command:
#
#   bash videodepth/run_pipeline.sh            # the real runs
#   SMOKE=1 bash videodepth/run_pipeline.sh    # 5-10 min end-to-end error check
#
# SMOKE=1 exercises every step's REAL code path — checkpoint loading, data
# loading, forward, backward, checkpoint writing, eval JSON writing — with
# ~3 training iterations and a few eval frames, in separate runs/smoke_*
# directories (never touches, skips, or pollutes the real run dirs). If the
# smoke pass completes, the real pipeline differs only in iteration counts.
#
# Steps (each skipped with a clear message if its prerequisite is missing,
# and skipped if its output already exists so an interrupted run resumes
# where it left off — use FORCE=1 to redo everything; SMOKE implies FORCE):
#
#   1. Baseline streaming eval of the per-frame Phase-1 checkpoint
#      (the comparison row: abs_rel + TAE before the temporal stage).
#   2. Train the temporal stage (TGM) on top of that checkpoint.
#   3. Streaming eval of the temporal model (compare with step 1).
#   4. Train Phase-3 with the bounded pair-attention head (needs Phase-2).
#   5. Eval Phase-3 (refined vs base).
#   6. [opt-in] Fine-tune full DAv2 (encoder+DPT head) — the AbsRel lever.
#      Enable by setting DAV2_WEIGHTS=/path/to/depth_anything_v2_metric_hypersim_vitl.pth
#
# Overridable env vars (defaults match the checkpoints actually on this
# server: Phase 1 = hdi_dav2, Phase 2 = phase2_run):
#   P1_CKPT=runs/hdi_dav2/best.pth          Phase-1 spatial checkpoint
#   P2_CKPT=runs/phase2_run/best.pth        Phase-2 checkpoint (Phase 3 needs it)
#   TEMPORAL_CFG=videodepth/configs/video_temporal_dav2.yaml
#   PHASE3_CFG=videodepth/configs/phase3_dav2_p2run.yaml
#   DAV2_WEIGHTS=                            (unset = skip step 6)
#   FORCE=                                   (1 = rerun steps whose outputs exist)
#   SMOKE=                                   (1 = tiny error-check run, see above)

set -euo pipefail
cd "$(dirname "$0")/.."     # repo root, wherever the script is called from

P1_CKPT="${P1_CKPT:-runs/hdi_dav2/best.pth}"
P2_CKPT="${P2_CKPT:-runs/phase2_run/best.pth}"
TEMPORAL_CFG="${TEMPORAL_CFG:-videodepth/configs/video_temporal_dav2.yaml}"
PHASE3_CFG="${PHASE3_CFG:-videodepth/configs/phase3_dav2_p2run.yaml}"
DAV2_WEIGHTS="${DAV2_WEIGHTS:-}"
FORCE="${FORCE:-}"
SMOKE="${SMOKE:-}"

if [[ -n "$SMOKE" ]]; then
    FORCE=1
    PREFIX="smoke_"
    LABEL="SMOKE TEST"
    # tiny but complete: 3 iters, eval at iter 2 (so best.pth is written and
    # the next step has a checkpoint to load), 1-sample batches, no workers
    TINY=(optim.total_iters=3 optim.warmup_iters=0 optim.log_every=1
          optim.ckpt_every=2 optim.eval_every=2 optim.batch_size=1
          optim.num_workers=0)
    EVAL_BATCHES=(--max-batches 5)
    EVAL_FRAMES=(--max-frames 20)
else
    PREFIX=""
    LABEL="REAL RUN"
    TINY=()
    EVAL_BATCHES=()
    EVAL_FRAMES=()
fi

BASE_RUN="runs/${PREFIX}hdi_dav2_baseline"
TEMPORAL_RUN="runs/${PREFIX}video_temporal_dav2"
PHASE3_RUN="runs/${PREFIX}phase3_video_dav2"
DAV2_RUN="runs/${PREFIX}dav2_full"
LOGDIR="runs/${PREFIX}pipeline_logs"
mkdir -p "$LOGDIR"

banner() { printf '\n============================================================\n %s\n============================================================\n' "$*"; }
skip()   { printf ' >> SKIP: %s\n' "$*"; }
run()    { local name="$1"; shift
           printf ' >> %s\n >> log: %s/%s.log\n' "$*" "$LOGDIR" "$name"
           "$@" 2>&1 | tee "$LOGDIR/$name.log"; }
fresh()  { [[ -n "$FORCE" || ! -e "$1" ]]; }   # true if we should (re)produce $1

banner "$LABEL — P1=$P1_CKPT  P2=$P2_CKPT"

# ---------------------------------------------------------------- step 1
banner "[1/6] Baseline streaming eval (per-frame $P1_CKPT)"
BASE_EVAL="$BASE_RUN/eval_test_streaming.json"
if [[ ! -f "$P1_CKPT" ]]; then
    skip "no Phase-1 checkpoint at $P1_CKPT (set P1_CKPT=...)"
elif ! fresh "$BASE_EVAL"; then
    skip "baseline eval exists: $BASE_EVAL"
else
    run baseline_eval python -m instancedepth.engine.evaluate_hdi \
        --config instancedepth/configs/hdi_dav2.yaml \
        --checkpoint "$P1_CKPT" --streaming "${EVAL_BATCHES[@]}" \
        --override "run_name=$(basename "$BASE_RUN")"
fi

# ---------------------------------------------------------------- step 2
banner "[2/6] Temporal stage training ($TEMPORAL_CFG)"
if [[ ! -f "$P1_CKPT" ]]; then
    skip "needs $P1_CKPT"
elif ! fresh "$TEMPORAL_RUN/best.pth"; then
    skip "already trained: $TEMPORAL_RUN/best.pth"
else
    # reuse the (model-independent) motion-score cache from any earlier run
    mkdir -p "$TEMPORAL_RUN"
    if [[ ! -f "$TEMPORAL_RUN/motion_scores.json" ]]; then
        found="$(ls runs/*/motion_scores.json 2>/dev/null | head -1 || true)"
        if [[ -n "$found" ]]; then
            cp "$found" "$TEMPORAL_RUN/motion_scores.json"
            echo " >> reused motion-score cache from $found"
        fi
    fi
    run temporal_train python -m videodepth.engine.train_video \
        --config "$TEMPORAL_CFG" --run-name "$(basename "$TEMPORAL_RUN")" \
        --override "init_checkpoint=$P1_CKPT" "${TINY[@]}" \
        ${SMOKE:+eval.max_frames=20}
fi

# ---------------------------------------------------------------- step 3
banner "[3/6] Temporal model streaming eval (vs step 1)"
if [[ ! -f "$TEMPORAL_RUN/best.pth" ]]; then
    skip "no temporal checkpoint at $TEMPORAL_RUN/best.pth"
elif ! fresh "$TEMPORAL_RUN/eval_streaming_test.json"; then
    skip "eval exists: $TEMPORAL_RUN/eval_streaming_test.json"
else
    run temporal_eval python -m videodepth.engine.evaluate_video \
        --config "$TEMPORAL_CFG" --checkpoint "$TEMPORAL_RUN/best.pth" \
        "${EVAL_FRAMES[@]}" \
        --override "run_name=$(basename "$TEMPORAL_RUN")"
fi

# ---------------------------------------------------------------- step 4
banner "[4/6] Phase-3 training, bounded pair-attention head ($PHASE3_CFG)"
if [[ ! -f "$P1_CKPT" || ! -f "$P2_CKPT" ]]; then
    skip "needs both $P1_CKPT and $P2_CKPT (set P1_CKPT=/P2_CKPT=... to your checkpoints)"
elif ! fresh "$PHASE3_RUN/best.pth"; then
    skip "already trained: $PHASE3_RUN/best.pth"
else
    run phase3_train python -m videodepth.engine.train_phase3_video \
        --config "$PHASE3_CFG" \
        --override "phase1_checkpoint=$P1_CKPT" "phase2_checkpoint=$P2_CKPT" "${TINY[@]}" \
        --run-name "$(basename "$PHASE3_RUN")"
fi

# ---------------------------------------------------------------- step 5
banner "[5/6] Phase-3 eval (refined vs base)"
if [[ ! -f "$PHASE3_RUN/best.pth" ]]; then
    skip "no Phase-3 checkpoint at $PHASE3_RUN/best.pth"
elif ! fresh "$PHASE3_RUN/eval_phase3_test.json"; then
    skip "eval exists: $PHASE3_RUN/eval_phase3_test.json"
else
    run phase3_eval python -m videodepth.engine.evaluate_phase3_video \
        --config "$PHASE3_CFG" --checkpoint "$PHASE3_RUN/best.pth" \
        "${EVAL_BATCHES[@]}" \
        --override "run_name=$(basename "$PHASE3_RUN")"
fi

# ---------------------------------------------------------------- step 6
banner "[6/6] Full-DAv2 Phase-1 fine-tune (the AbsRel lever)"
if [[ -z "$DAV2_WEIGHTS" ]]; then
    skip "opt-in: set DAV2_WEIGHTS=/path/to/depth_anything_v2_metric_hypersim_vitl.pth"
elif [[ ! -f "$DAV2_WEIGHTS" ]]; then
    skip "DAV2_WEIGHTS file not found: $DAV2_WEIGHTS"
elif ! fresh "$DAV2_RUN/best.pth"; then
    skip "already trained: $DAV2_RUN/best.pth"
else
    run dav2_train python -m videodepth.engine.train_dav2 \
        --config videodepth/configs/dav2_full.yaml \
        --run-name "$(basename "$DAV2_RUN")" \
        --override "dav2_checkpoint=$DAV2_WEIGHTS" \
                   "backbone.checkpoint_path=$DAV2_WEIGHTS" "${TINY[@]}" \
        ${SMOKE:+eval_max_frames=20}
    run dav2_eval python -m videodepth.engine.train_dav2 --evaluate \
        --config videodepth/configs/dav2_full.yaml \
        --run-name "$(basename "$DAV2_RUN")" \
        --checkpoint "$DAV2_RUN/best.pth" \
        ${SMOKE:+--override eval_max_frames=20}
fi

# ---------------------------------------------------------------- summary
banner "$LABEL complete — result files"
for f in "$BASE_EVAL" \
         "$TEMPORAL_RUN/eval_streaming_test.json" \
         "$PHASE3_RUN/eval_phase3_test.json" \
         "$DAV2_RUN/eval_streaming_test.json"; do
    if [[ -f "$f" ]]; then printf '  %s\n' "$f"; else printf '  (missing) %s\n' "$f"; fi
done
if [[ -n "$SMOKE" ]]; then
    echo "Smoke artifacts live under runs/smoke_* (safe to delete: rm -rf runs/smoke_*)."
    echo "All executed steps passed -> the real pipeline runs the same code with more iterations:"
    echo "    bash videodepth/run_pipeline.sh"
else
    echo "Compare step-1 vs step-3 JSONs: abs_rel must hold, TAE/flicker_ratio should drop."
fi
