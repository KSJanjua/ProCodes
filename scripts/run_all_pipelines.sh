#!/usr/bin/env bash
# Stage 1: Phase 3 on the CURRENT checkpoints (phase3_current.yaml).
# Stage 2: the full DAv2 retrain pipeline (scripts/run_dav2_pipeline.sh),
#          which itself chains Phase 1 -> eval -> Phase 2 -> eval -> Phase 3
#          -> eval into runs/hdi_dav2/, runs/phase2_dav2/, runs/phase3_dav2/.
#
# Meant to be started once and left running unattended -- see the
# nohup/tmux launch instructions in the header comment this script's caller
# should use (not repeated here); this file only defines the *sequence*.
#
# Stage 1's train->eval pair uses a hard `&&`: evaluating a checkpoint that
# training itself failed to produce is pointless.
#
# Stage 2 is intentionally NOT gated on Stage 1's success (no `&&` between
# them): Phase 3 is the only stage with two full backbones resident at once
# (its own heavier memory footprint), so a Phase-3-specific failure there
# (e.g. an OOM) says nothing about whether Phase 1/2's much lighter,
# single-backbone DAv2 training will succeed. Hard-blocking a multi-day
# unattended retrain on an unrelated Stage-1 hiccup would waste the whole
# window for no reason -- so Stage 2 always runs, and its own outcome is
# reported/exits with its own status.

set -uo pipefail   # deliberately NOT -e at this top level -- see above
cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

log() { echo -e "\n########## $(date '+%Y-%m-%d %H:%M:%S')  $1 ##########\n"; }

log "STAGE 1: PHASE 3 ON CURRENT CHECKPOINTS (phase3_current.yaml)"
if python -m instancedepth.engine.train_phase3 \
       --config instancedepth/configs/phase3_current.yaml \
   && python -m instancedepth.engine.evaluate_phase3 \
       --config instancedepth/configs/phase3_current.yaml \
       --checkpoint runs/phase3_current/best.pth
then
    log "STAGE 1 COMPLETE -- runs/phase3_current/eval_phase3_test.json written"
else
    log "STAGE 1 FAILED (see above) -- continuing to Stage 2 anyway (fully independent checkpoints/config)"
fi

log "STAGE 2: FULL DAv2 RETRAIN PIPELINE (hdi_dav2 -> phase2_dav2 -> phase3_dav2)"
bash scripts/run_dav2_pipeline.sh
stage2_status=$?

if [ "$stage2_status" -eq 0 ]; then
    log "ALL STAGES COMPLETE"
else
    log "STAGE 2 FAILED (exit $stage2_status) -- check the banner above for which phase"
fi
exit "$stage2_status"
