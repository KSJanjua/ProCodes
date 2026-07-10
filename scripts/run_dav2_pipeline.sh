#!/usr/bin/env bash
# Runs the full DAv2-encoder pipeline: Phase 1 -> Phase 2 -> Phase 3, with a
# standalone eval (JSON saved to disk) after each training stage. One command:
#
#   bash scripts/run_dav2_pipeline.sh
#
# All three phases write to their own, previously-unused run folders --
# runs/hdi_dav2/, runs/phase2_dav2/, runs/phase3_dav2/ -- so nothing from
# earlier runs (hdi_enhanced, phase2_run, phase3_smoke, ...) is touched.
#
# `set -e` stops the whole pipeline on the first failing step (e.g. Phase 1
# training crashing) rather than silently continuing into Phase 2/3 on top
# of a broken checkpoint.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

log() { echo -e "\n========== $(date '+%Y-%m-%d %H:%M:%S')  $1 ==========\n"; }

# ---- Phase 1: Holistic Depth Initialization (DAv2 encoder) -----------------
log "PHASE 1 TRAIN  (hdi_dav2, disparity_aux_weight=1.2)"
python -m instancedepth.engine.train_hdi \
    --config instancedepth/configs/hdi_dav2.yaml

log "PHASE 1 EVAL  -> runs/hdi_dav2/eval_test.json"
python -m instancedepth.engine.evaluate_hdi \
    --config instancedepth/configs/hdi_dav2.yaml \
    --checkpoint runs/hdi_dav2/best.pth

# ---- Phase 2: Instance Depth Layer Prediction ------------------------------
log "PHASE 2 TRAIN  (phase2_dav2)"
python -m instancedepth.engine.train_phase2 \
    --config instancedepth/configs/phase2_dav2.yaml

log "PHASE 2 EVAL  -> runs/phase2_dav2/eval_test.json"
python -m instancedepth.engine.evaluate_phase2 \
    --config instancedepth/configs/phase2_dav2.yaml \
    --checkpoint runs/phase2_dav2/best.pth

# ---- Phase 3: Occlusion-Aware Depth Refinement -----------------------------
log "PHASE 3 TRAIN  (phase3_dav2, occlusion_only=true)"
python -m instancedepth.engine.train_phase3 \
    --config instancedepth/configs/phase3_dav2.yaml

log "PHASE 3 EVAL  -> runs/phase3_dav2/eval_phase3_test.json"
python -m instancedepth.engine.evaluate_phase3 \
    --config instancedepth/configs/phase3_dav2.yaml \
    --checkpoint runs/phase3_dav2/best.pth

log "PIPELINE COMPLETE"
echo "Results:"
echo "  runs/hdi_dav2/eval_test.json"
echo "  runs/phase2_dav2/eval_test.json"
echo "  runs/phase3_dav2/eval_phase3_test.json"
