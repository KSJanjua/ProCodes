# Results Analysis & Improvement Plan (post first full runs)

Evidence base: `results/*.json` (recorded runs), `docs/PHASE3_DIAGNOSIS.md`
(implementation defects found after those runs), FlashDepth integration
(`docs/TEMPORAL_DESIGN.md`, implemented alongside this document).

## 1. Phase 1 — why abs_rel ≈ 0.078 and what to do

Recorded: `hdi_enhanced` 0.0782, `hdi_dav2` 0.0778 (paper's holistic-stage
rows sit near 0.051). Analysis, in order of estimated contribution:

1. **Different dataset, noisier GT (not a bug).** The paper's numbers are on
   GID (RealSense D455 / Azure Kinect). This project's GT is ZED **stereo**
   depth at 720p: stereo disparity error grows quadratically with range, so
   the 4–10 m band — where most people in this data stand — carries
   substantially noisier supervision *and* noisier evaluation targets. Part
   of the 0.078-vs-0.051 gap is irreducible protocol difference; matching the
   paper's absolute number was never the right target. The right targets are
   internal: beat the per-frame baseline with the temporal stage, and beat
   Phase-1 with Phase-3.
2. **No LR warmup** — `warmup_iters: 0` let full-LR gradients hit the
   pretrained DINOv2 from step 0. **Fixed:** warmup 500 added to
   `hdi_enhanced` / `hdi_dav2` (faithful `hdi.yaml` untouched).
3. **Per-frame estimation noise** shows up both as flicker and as metric
   error. **Addressed:** the FlashDepth temporal stage (below) attacks
   exactly this; stage 2b (joint fine-tune at tiered LRs) historically also
   improves per-frame accuracy slightly by letting features co-adapt.
4. **Untuned hyperparameters carried from conventions** — SigLog λ, head LR
   ×10, bin count are DAv2/paper conventions, not swept on this dataset.
   Recommended sweep order (cheapest signal first): `silog_lambda`
   {0.3, 0.5, 0.85} → `head_lr_mult` {5, 10} → `rd` {4, 5, 6}.
5. **Augmentation is minimal** (hflip only). A scale/crop augmentation is the
   standard next lever for metric depth; deferred (touches the shared
   dataset), recommended after the temporal stage lands.

## 2. Phase 3 — the smoking gun and the fixes

Recorded evidence (`results/phase3_current_eval.json`):

| Quantity | Value |
|---|---|
| Phase-1 standalone abs_rel | **0.0782** |
| Same branch after Phase-3's 25k fine-tune (`overall_base`) | **0.1386** (+77%) |
| `occ_base` (instance regions, which WERE supervised) | 0.0784 (unchanged) |

**Diagnosis: catastrophic forgetting.** Phase 3 fine-tunes the entire depth
branch (paper Sec. 4.3) but its losses (Eq. 10–11) touch only paired-instance
ROI pixels — everything outside instances received 25k steps of gradient-free
drift. Instance regions stayed accurate; background/ground collapsed. On top
of that, the recorded runs trained under the since-fixed defects D1–D5
(`PHASE3_DIAGNOSIS.md`) — most damagingly D4, which actively taught depth
suppression at instance boundaries — and composited with the identity-breaking
D1 paste, which alone explains `occ_refined` (0.0801) landing *worse* than
`occ_base` (0.0784).

**Fixes applied (run profiles `phase3_current` / `phase3_dav2`; the faithful
`phase3.yaml` keeps the paper recipe for comparison):**

- `loss.holistic_weight: 0.3` — the anti-forgetting full-frame SigLog term.
  Directly targets the 0.078→0.139 drift; weight chosen so the dense term
  anchors the ~97% of pixels the ROI losses never see without drowning the
  refinement signal (λ_obj=1.0, λ_dist=0.5). *Deviation from Eq. 12,
  justified by recorded evidence.*
- `composite_ratio: "scalar"` + `composite_feather_px: 5` — one coherent
  correction per instance, blended in over ~5 px at mask boundaries.
  Addresses the observed color-mixing inside static objects and the
  ring/outline artifacts around instances (a hard ratio≠1 edge at the mask
  boundary is a step discontinuity in the depth map — the "outlines" were
  composite seams, not model output). Compositing is not paper-specified,
  so these are implementation choices, not method changes.
- `warmup_iters: 500`.
- **A retrain is mandatory**: every recorded Phase-3 number reflects the
  pre-fix code (broken pairing for part of training, diluted GT targets,
  identity-breaking composite). `scripts/run_full_pipeline.sh` performs it.

## 3. FlashDepth temporal stage — dataset-specific hyperparameters

Implemented per `docs/TEMPORAL_DESIGN.md` (module: `models/hdi/temporal.py`,
training: `engine/train_hdi_temporal.py`, profile: `hdi_temporal.yaml`).
Values chosen for **this** dataset (297 sequences; 50–300 frames each;
~44k train frames):

| Hyperparameter | Value | Reasoning |
|---|---|---|
| clip_len | 5 | FlashDepth's `video_length`; BPTT memory stays trivial with the frozen backbone |
| clip_strides | {1,2,4,8} | max clip span 4·8+1 = 33 frames — fits even the shortest (~50-frame) sequences while teaching large-motion bridging; wider strides would starve short sequences of clips |
| lr | 1e-4 | FlashDepth's temporal-module LR; only the ~2M-param aligner trains |
| total_iters / batch | 8000 / 2 | ≈16k clips ≈ 80k frame-visits for a 2M-param module over a frozen backbone — ample coverage of 237 train sequences without overfitting; the clip index (~170k clips) is never exhausted |
| warmup | 500 | fresh module, fast LR |
| d_model / blocks / downsample | 128 / 2 / 0.1 | ≈2M params (~0.5% of model, matching FlashDepth's ~1% footprint claim); 0.1 is FlashDepth's value |
| levels | [2] | F_2 = last decoder feature before the heads (FlashDepth's placement analogue); [0] and [0,1,2] remain config ablations |
| freeze_spatial | true | stage 2a: clean attribution + cheap; stage 2b (tiered unfreeze) after 2a proves the module |

Measurement: `evaluate_hdi --streaming` reports the standard metrics under
sequence-ordered stateful inference **plus `temporal_alignment_error`**
(mean |Δpred−Δgt| between consecutive frames — 0 for a prediction that tracks
GT perfectly, motion-invariant, flow-free). Run it on the per-frame baseline
first: that number is the flicker baseline the temporal stage must beat, at
unchanged dense metrics.

## 4. Visualization fixes (this change)

- **Phase 1 far-range mismatch**: GT renders black beyond the sensor range
  (returns 0), predictions stayed dark-blue. `colorize_depth` gained
  `far_thresh`; both video tools pass `far_thresh = max_depth`, so predicted
  depth ≥ 10 m now renders black exactly like GT.
- **Blurry videos / color mixing**: two sources fixed — encoder default
  quality (now `-crf 18` / `-q:v 3` on every ffmpeg path; codec-rate blur was
  smearing the fine TURBO gradients) and the composite artifacts above
  (scalar ratio + feathering).
- **Instance outlines**: composite seams, fixed by feathering (§2). The
  instance *panel*'s contour drawing is intentional (it's an annotation
  overlay); the depth panels now carry no boundary markings.

## 5. Loss-function audit (is SigLog wrong for a single-source dataset?)

**Short answer: no — the concern conflates two different losses.** But the
audit did surface a real, separate gap, now fixed (§5.3).

### 5.1 SSI vs SILog — the distinction that matters

| | **SSI / `L_ssi`** (MiDaS, DPT, DAv2-*relative*) | **SILog / SigLog** (Eigen 2014; what we use) |
|---|---|---|
| Form | least-squares align pred to GT in **scale and shift**, then compare | `sqrt(mean(d²) − λ·mean(d)²)`, `d = log(gt) − log(pred)` |
| Absolute scale | **discarded entirely** | **penalized** (for λ<1) |
| Why it exists | training on **mixed sources** with unknown/inconsistent scale | training on **one calibrated sensor** with metric GT |

The loss the concern describes — the multi-source one that throws away metric
scale — is **SSI**, and this project does **not** use it. `SigLogLoss`'s
docstring explicitly rejects it for exactly that reason.

SILog with λ<1 is *not* scale-invariant. Substituting `pred → α·pred`
(i.e. `d → d − log α`):

```
mean(d²) − λ·mean(d)²  →  [mean(d²) − λ·mean(d)²] + (1−λ)·[(log α)² − 2·log α·mean(d)]
```

The extra term vanishes **only at λ = 1**. At the configured **λ = 0.5 a pure
metric-scale error is penalized** — verified as an executable test
(`test_siglog_penalizes_absolute_scale`: λ=0.5 scores >0.05 on a 25% scale
error, λ=1.0 scores ~0, and a larger scale error costs strictly more).

Corroboration: SILog λ∈[0.5, 0.85] is the standard loss of essentially every
**single-sensor** metric-depth model — Eigen (NYU), BTS, AdaBins, NeWCRFs,
ZoeDepth, DAv2-metric — all trained on one sensor (KITTI or NYU). It is the
right family here. Phase 3 additionally has no choice: Eq. 10 *specifies*
SigLog, citing Eigen [16].

### 5.2 λ is worth sweeping (the legitimate version of the concern)

λ trades absolute-scale fidelity against relative structure. We use λ=0.5
(Eigen/DAv2); BTS/AdaBins/NeWCRFs use **0.85** (more structure-focused).
Given ZED stereo GT is noisiest exactly where a systematic scale bias would be
inferred (4–10 m), λ=0.85 is a plausible win. Kept as the top sweep item —
`loss.silog_lambda` is already config-exposed, so it costs one override.

### 5.3 The real gap: nothing penalized blurry edges → gradient matching added

SigLog is a **pointwise** statistic: it is indifferent to *where* error sits,
so a prediction that smears a depth discontinuity across many pixels can score
as well as a sharp one. That is a direct contributor to the soft/blurry depth
seen in the visualizations. **Added** `GradientMatchingLoss` — multi-scale
gradient matching on the log residual (Eigen & Fergus 2015; MiDaS/DPT's
`L_reg`), minimized only when predicted depth *edges* coincide with GT edges.
Gradients count only where both neighbouring pixels have valid GT, so sensor
holes cannot manufacture edges. `loss.gradient_matching_weight: 0.5` in
`hdi_enhanced` / `hdi_dav2` / `hdi_temporal`; **0 in faithful `hdi.yaml`**.
Phase 1's loss is unspecified by InstanceDepth, so this is a free, well-cited
choice — and it is the improvement most likely to sharpen object separation
without any architectural change.

Phase 2's losses (Eq. 5–7: CE + point-sampled BCE/Dice + smooth-L1 depth) and
Phase 3's (Eq. 10–12) are paper-specified and stay unchanged.

## 6. Mask2Former's own metrics (COCO mask AP)

Phase 2 eval now reports **AP, AP50, AP75, APs, APm, APl** — exactly what
Mask2Former's tables report — alongside the existing project metrics.
Implemented standalone (`utils/phase2_metrics.compute_mask_ap`) against the
COCO protocol rather than depending on `pycocotools` (absent here, and it
would need a COCO-format conversion step): 10 IoU thresholds .50:.05:.95,
101-point interpolated PR, global score ranking across images, COCO's ignore
semantics for the area bands. Detections are **not** score-filtered (AP is
rank-based — a cut would silently truncate the PR curve) and are scored the
way Mask2Former's own `instance_inference` does: **class confidence × mask
quality**.

Why keep both suites: AP is rank-based and externally comparable (it answers
"how good is the instance branch in the field's standard currency"); the
existing fixed-threshold P/R/F1/IoU + **depth-layer MAE** answer "how good is
it at the operating point Phase 3 actually consumes" — and depth-layer MAE has
no standard equivalent, since the depth layer is this paper's own addition.
Both are reported for the whole test split and the occlusion slice.
Correctness is pinned by 7 tests against analytically-known cases
(`tests/test_coco_ap.py`).

## 7. Orchestration

`scripts/run_full_pipeline.sh` — one command for P1 → P1b(temporal) → P2 → P3,
each with train / eval / viz / videos; training failures gate only their
dependents (P2 runs even if P1 fails; P3 needs both checkpoints), all other
failures are recorded and skipped past; eval JSONs are copied into `results/`;
a pass/fail/skip summary ends the run and sets the exit code.
