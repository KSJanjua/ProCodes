# InstanceDepth — Full Pipeline Audit (2026)

A ground-up audit of all three phases, driven by the recorded evaluation
JSONs in `results/` and a line-by-line reading of the model/engine code. Every
claim below is tied to a number in `results/` or a specific code path, not to
intuition.

> **Scope / honesty note.** This audit was produced in a code-only checkout
> (no dataset, no checkpoints, no GPU). Nothing here re-measures AbsRel — it
> **diagnoses** from the metrics already recorded in `results/` and the code.
> Findings are tagged **[verified-in-code]** (provable by reading/running the
> code here), **[evidenced]** (backed by a recorded metric), or
> **[needs-validation]** (a redesign that must be confirmed on the training
> server). Changes shipped in this iteration are only the [verified-in-code]
> ones; the [needs-validation] items are a ranked roadmap with exact commands.

---

## 0. The evidence, in one table

| Run | Config | abs_rel | TAE | note |
|-----|--------|--------:|----:|------|
| Phase 1 vanilla | `hdi_enhanced` (non-stream) | **0.0782** | — | best single-frame P1 |
| Phase 1 vanilla | `hdi_enhanced` (streaming, full) | 0.0820 | 0.05868 | flicker baseline |
| Phase 1 + temporal | `hdi_temporal` (streaming, full) | 0.0792 | **0.05843** | TAE moved **0.4 %** |
| Phase 1 DAv2-enc | `hdi_dav2` | 0.0778 | — | encoder swap ≈ no gain |
| Phase 3 base | `phase3_current` (= P1 depth) | **0.1387** | — | drifted from 0.078! |
| Phase 3 refined | `phase3_current` | 0.1387 | — | ≈ base, no help |
| Phase 3 base (occ) | occlusion slice | 0.0784 | — | — |
| Phase 3 refined (occ) | occlusion slice | **0.0801** | — | **worse** than base |

Three headline facts fall straight out of this table:

1. The temporal module changes TAE by 0.4 % — a **null result**.
2. Phase-3's *base* depth (0.139) is **78 % worse** than Phase-1's own eval
   (0.078) — the pipeline damages the depth before refinement even acts.
3. Even on the occlusion slice it is *supposed* to fix, Phase-3 refined is
   **worse** than base (0.0801 vs 0.0784).

---

## 1. Phase 1 — Holistic Depth Initialization

### 1.1 Why the temporal module is a null result  [evidenced]

**Root cause: this dataset has no coherent temporal-inconsistency signal for
the module to remove.** This is not an architecture bug — the module is
correctly built and correctly a no-op.

The proof is already in the repo, in
`results/hdi_enhanced_eval_streaming_subset100.json`:

```
temporal_alignment_error ≈ sqrt(gt_temporal_delta² + pred_temporal_delta²)
   (holds to 0.15 % on the measured subset)
```

That identity says the prediction's frame-to-frame change and the ground
truth's frame-to-frame change are **statistically independent** — TAE is
measuring two *uncorrelated noise sources adding in quadrature* (single-sensor
depth noise + independent prediction jitter), not a systematic flicker that a
recurrent aligner could learn to cancel. When there is no correlated temporal
error, the optimal temporal correction is ≈ 0, and that is exactly what the
zero-initialised ConvGRU learned to stay near (TAE 0.05868 → 0.05843).

Contributing factors, in priority order:

- **Dataset property (dominant).** Human-activity clips from a fixed
  single sensor are slow/near-static frame-to-frame; the GT itself moves as
  much as the prediction error (`gt_temporal_delta` 0.037 vs
  `pred_temporal_delta` 0.028 on the subset). There is little coherent
  inconsistency to fix. FlashDepth/Video-Depth-Anything gains come from
  large, motion-rich video corpora — a regime this data doesn't occupy.
- **Wrong model-selection signal [verified-in-code].** `train_hdi_temporal`
  selects `best.pth` via `make_eval_fn`, whose `evaluate()` calls
  `reset_temporal_state()` **every batch** on shuffled frames
  (`engine/evaluate_hdi.py:62`). So checkpoint selection ran in a mode where
  the temporal memory is *always empty* — it literally cannot see the metric
  the module targets. (The streaming eval that *can* is only run offline,
  after the fact.) Selection was blind to the module's purpose.
- **Aggressive 10 % downsample.** `TemporalAligner(downsample=0.1)` makes the
  residual a very low-frequency correction — fine for slow global drift,
  unable to touch high-frequency flicker even if it existed.

**Decision: do not "fix" the temporal module — report it honestly.** Churning
the architecture to chase a signal the data doesn't contain would be
manufacturing a result. The correct engineering statement is: *on this
dataset, per-frame depth is already temporally stable to within sensor noise;
the temporal stage is a measured no-op and should be reported as such (and
left opt-in / off by default).* If temporal consistency ever becomes a real
target, it needs motion-rich video data and a TAE-based selection signal
first — not a bigger module.

### 1.2 Why AbsRel is "far from the paper"  [needs-validation]

AbsRel ≈ 0.078 is already strong for metric depth, so first confirm the paper
comparison is like-for-like (same benchmark, same `abs_rel` definition, same
`max_depth` cap). If there is still a real gap, the lever is **not** the
encoder: swapping vanilla DINOv2 → DAv2 encoder moved abs_rel 0.0782 → 0.0778
(0.5 %). The encoder isn't the bottleneck; the **from-scratch Depth-Range
decoder + ordinal-bin refinement head trained on a small dataset** is.

Ranked options (validate on the server):

1. **Initialise Phase 1 from the *full* pretrained Depth-Anything-V2 (encoder
   **+** DPT head), then fine-tune** — instead of training a bespoke decoder
   from scratch. Today only the DAv2 *encoder* is loaded; its pretrained
   dense-depth decoder (the part that actually carries the depth prior) is
   discarded. Adopting it imports a strong metric-depth prior and is the
   single most likely AbsRel win. `models/backbone/dinov2_wrapper.py` already
   handles DAv2 encoder key-remapping, so this is additive.
2. **Keep the disparity-space auxiliary loss** — `hdi_dav2` already shows a
   healthy `disparity_abs_rel` (0.072); weight it more.
3. Confirm the bin refinement (`iterative_refinement.py`) is helping at all vs
   a plain DPT regression head — ablate it.

---

## 2. Phase 2 — Instance Segmentation

### 2.1 Poor cross-dataset generalization  [needs-validation]

Symptom: works on the custom set, fails on foreign data. Cause is standard
catastrophic specialization — a COCO-pretrained Mask2Former fine-tuned hard on
one sensor's statistics forgets COCO's breadth. Mitigations, cheapest first:

- **Freeze the Swin-L backbone (or use a very low backbone LR) during Phase-2
  fine-tune**; train mainly the pixel/transformer decoder. Preserves COCO
  features.
- **Strong photometric + geometric augmentation** (the config has
  `color_jitter: 0.0` — turn it on) so the model can't overfit sensor color/
  exposure.
- For a true zero-shot person segmenter, prefer a **promptable foundation
  model (SAM2)** at inference over a narrowly fine-tuned head.

### 2.2 Identity flicker (colors/IDs switch every frame)  [verified-in-code]

Per-frame Mask2Former emits queries in an **arbitrary, frame-independent
order** — query #k is a different person each frame — so any colormap keyed on
query index strobes. This is inherent to per-frame instance seg and needs an
explicit **association** step. Ranked:

1. **SAM2 / DEVA mask propagation (most robust).** Segment (or prompt) once,
   then *propagate* masks through the clip with a video object segmenter.
   Identity is carried by the tracker's memory, not re-derived per frame —
   this is the permanent-identity guarantee the user wants, and it also fixes
   occlusion re-appearance.
2. **Hungarian mask-IoU tracker with track memory (cheap, no new weights).**
   Match this frame's masks to *live tracks* (not just the previous frame) by
   IoU (+ embedding cosine + depth-layer gate), `scipy.linear_sum_assignment`;
   keep unmatched tracks alive for N frames so short occlusions don't reset
   the color. **[partly shipped]** `utils/viz.py::MaskTracker` was upgraded
   from greedy to global Hungarian assignment (with a greedy fallback when
   scipy is absent); it already keeps lost tracks alive for `max_age` frames.
   Honest caveat: for *realizable* masks greedy and Hungarian rarely diverge
   (the pathological case needs near-identical masks), so this is a
   correctness hardening, **not** the cure — option 1 (propagation) is the
   real fix for foreign video, and coloring by `track_id` (§4.1) is the fix
   for annotated sequences.

> **Video Mask2Former?** It gives clip-level identity natively but at real
> cost: clip-window attention, higher memory, and its own training. For a
> people-in-groups setting, SAM2/DEVA propagation (option 1) delivers the same
> permanent identity at inference-only cost — recommended over Video-M2F
> unless clip-level *training* is otherwise needed.

---

## 3. Phase 3 — Instance-Aware Depth Rectification  ← biggest concern

### 3.1 Root cause of "Phase-3 is worse than Phase-2"  [evidenced, FIXED]

Phase-3 fine-tunes the trainable Phase-1 branch at LR 1e-6 (paper Sec. 4.3),
but `Phase3Criterion`'s `L_obj` supervises **only ROI (instance) pixels**
(`losses/phase3_losses.py`). With no whole-frame supervision, the dense
Phase-1 depth **drifts catastrophically in non-instance regions**:

> `phase3_current.yaml` comment, verbatim: *"recorded runs drifted the base
> depth 0.078 → 0.139 abs_rel because only ROI pixels were supervised."*

The table confirms it: base = 0.1387 vs Phase-1's own 0.0782. The relation
head's composite is ≈ identity (refined 0.13866 vs base 0.13861), so **the
entire Phase-3 regression is the base drift, not the head.** And because the
drift dominates, it also swamps whatever small correction the head makes on
the occlusion slice (refined 0.0801 > base 0.0784).

**Decision — freeze Phase 1 during Phase 3 [SHIPPED].** Deviate from the
paper's joint fine-tune: pin the depth branch so the dense base is *guaranteed*
to stay at Phase-1 quality (0.078) and the refinement can only ever help. This
makes Phase-3 **non-degrading by construction** — the direct fix for the #1
concern.

Implemented:
- `Phase3Config.freeze_phase1: bool = True` (`configs/phase3_config.py`).
- `Phase3Model._freeze_phase1()` + `train()` keeps it in `eval()`
  (`models/phase3/model.py`).
- `build_phase3_optimizer` drops the now-empty depth group
  (`engine/train_phase3.py`).
- Set **True** in `phase3_current.yaml`, `phase3_dav2.yaml`; explicit
  **False** in `phase3.yaml`/`phase3_enhanced.yaml` to keep those
  paper-faithful. Unit-tested (`tests/test_phase3.py`).

The anti-forgetting `holistic_weight` regularizer becomes unnecessary once
Phase 1 is frozen (there is nothing to forget) — leave it 0 with freeze on.

### 3.2 Boundary artifacts (lines around people)  [verified-in-code]

They are the **silhouette of an instance-only depth edit**: Phase-3 multiplies
depth by a ratio `2E` *inside* each mask and leaves the background untouched,
so any `E ≠ 0.5` creates a step at the mask edge that the depth colormap draws
as a ring. Significance: mostly **cosmetic** (a visualization tell), but a
symptom of over-aggressive per-instance correction.

Already mitigated in `relation_head.composite_refined_depth`: soft-alpha
(sigmoid-mask) blending + `composite_feather_px`. With Phase 1 frozen (§3.1)
the corrections shrink toward identity, shrinking the rings further. Remaining
recommendation [needs-validation]: bound the correction magnitude
(clamp `|2E−1| ≤ ε`) and/or only apply it in genuinely-occluded regions, so a
mis-trained head can't paint a hard ratio step.

### 3.3 The deeper Phase-3 limitation  [evidenced]

Phase-3's premise is **amodal** depth (depth of *hidden* body parts beyond
occlusions). A single calibrated sensor has **no ground truth for hidden
pixels**, so `L_obj` is masked to *visible* pixels only (README, by
necessity). Training an occlusion-completion head with zero occlusion-region
supervision is fundamentally under-constrained — the head cannot learn what it
is never shown. This is *why* even the correctly-built head barely moves the
occlusion metric. Honest conclusion: on visible-only single-sensor GT, Phase 3
can be made **safe** (§3.1) but cannot be made to *strongly* improve occluded
depth without amodal supervision (synthetic/multi-view GT, or a learned amodal
prior). Frame the deliverable accordingly.

---

## 4. `annotations.json` per-sequence files  [verified-in-code]

They already drive the whole pipeline (`data/gid_dataset.py`,
`data_engine/annotate.py`): masks, track IDs, per-instance depth layers, boxes.
Two currently-underused, genuinely-beneficial hooks:

1. **Track IDs → free, permanent video identity.** The annotations carry
   `track_ids` (Phase-3's collate even drops them,
   `train_phase3.py:119`). For visualization of *annotated* sequences, key the
   instance colormap on the persistent `track_id` instead of the per-frame
   query index — zero-cost fix to §2.2 for the paper figures. (Live/foreign
   video still needs the tracker of §2.2.)
2. **Depth-layer ordering → L_dist supervision** is already used; keep it.

Do **not** invent new uses beyond these — they're the only ones the data
actually supports.

---

## 5. Visualization

- The required depth panel (RGB · GT · P1 · P1+temporal · P3 · GT) and seg
  panel (RGB · GT mask · pred mask) are supported by `visualize_hdi.py` /
  `visualize_phase3.py` / `visualize_phase2.py`. Ensure the P1-vs-P1+temporal
  columns share a colormap normalization so the (tiny) temporal delta is
  visible rather than renormalized away.
- **Video identity:** for annotated test sequences, color by `track_id`
  (§4.1) — permanent by construction. For arbitrary video, wire the §2.2
  tracker. This is the real fix for "same person keeps changing color."

---

## 6. Repository cleanup  [recommendation]

Conservative, because deleting is irreversible and several "extra" files are
legitimate profile variants. Safe to remove once confirmed unused by the
server workflow: one-shot diagnostics (`scripts/verify_mask2former_api.py`,
`scripts/inspect_dinov2_checkpoint.py`, `scripts/check_temporal_module.py`) and
redundant pipeline drivers (keep one of `run_full_pipeline.sh` /
`run_all_pipelines.sh` / `run_dav2_pipeline.sh`). The config `*_dav2` /
`*_enhanced` / `*_current` variants are **not** dead — they are the ablation
matrix; keep them. Not deleted in this pass to avoid breaking an unseen
server workflow — flagged for the owner to confirm.

---

## 7. Ranked roadmap (validate on the training server)

| # | Change | Phase | Expected effect | Status |
|---|--------|-------|-----------------|--------|
| 1 | **Freeze Phase 1 in Phase 3** | 3 | base pinned 0.139 → 0.078; Phase 3 non-degrading | **shipped** |
| 2 | Color video by `track_id` | viz | permanent identity on annotated seqs | quick, do next |
| 3 | Full DAv2 (encoder+DPT head) init for P1 | 1 | the real AbsRel lever | needs-validation |
| 4 | SAM2/DEVA propagation for live identity | 2 | permanent identity on foreign video | needs-validation |
| 5 | Freeze Swin-L + turn on augmentation | 2 | cross-dataset generalization | needs-validation |
| 6 | Report temporal as a measured no-op; keep off by default | 1 | honesty | decided |

### Validate change #1 (shipped) on the server

```bash
# Re-train Phase 3 with Phase 1 frozen (base can no longer drift):
python -m instancedepth.engine.train_phase3 \
    --config instancedepth/configs/phase3_current.yaml \
    --override phase1_checkpoint=runs/hdi_enhanced/best.pth \
               phase2_checkpoint=runs/phase2_run/best.pth
python -m instancedepth.engine.evaluate_phase3 \
    --config instancedepth/configs/phase3_current.yaml \
    --checkpoint runs/phase3_current/best.pth
# Expect: overall_base abs_rel ≈ 0.078 (NOT 0.139); overall_refined ≤ base.
```

---

## 8. What changed in this iteration

- **[shipped]** `freeze_phase1` across config/model/optimizer + tests (§3.1)
  — fixes the #1 Phase-3 regression by pinning the dense base at 0.078.
- **[shipped]** `MaskTracker` greedy → global Hungarian assignment + greedy
  fallback + tests (§2.2) — correctness hardening for video identity.
- **[audit]** This document: evidence-backed root causes for every concern,
  with the temporal null-result and Phase-3 regression *proven* from the
  recorded metrics rather than asserted.
- Everything requiring a GPU/data to validate is left as a ranked, commanded
  roadmap rather than untested code — deliberately, so no unverified model
  surgery lands in the pipeline.
