# Results Analysis — DAv2 Pipeline + Phase 3 Current Run

Raw numbers saved alongside this file: `hdi_enhanced_eval.json`, `hdi_dav2_eval.json`,
`phase2_run_eval.json`, `phase2_dav2_eval.json`, `phase3_current_eval.json`,
`phase3_dav2_eval.json`, `hdi_enhanced_eval_streaming.json`,
`hdi_temporal_eval_streaming.json`, `hdi_enhanced_eval_streaming_subset100.json`,
`hdi_dav2_berhu_eval.json`.

## 1. Phase 1 — no issue, works as expected

| | abs_rel | rms | sigma1 |
|---|---|---|---|
| hdi_enhanced (vanilla DINOv2) | 0.0782 | 0.3925 | 0.9333 |
| hdi_dav2 (DAv2 encoder) | 0.0778 | 0.3810 | 0.9383 |

DAv2 encoder is marginally better across every metric (RMS −2.9%). Expected: DAv2's encoder is already fine-tuned for depth, so it's a better initialization than vanilla self-supervised DINOv2. **Nothing wrong here.**

## 2. Phase 2 — the DAv2-pipeline rerun is meaningfully worse, and this needs verifying before drawing conclusions

| | precision | recall | f1 | depth_mae |
|---|---|---|---|---|
| phase2_run (original) — overall | 0.945 | 0.941 | 0.943 | 0.0597 |
| phase2_dav2 (rerun) — overall | 0.907 | 0.931 | 0.919 | 0.0799 |
| phase2_run — occlusion slice | 0.952 | 0.909 | 0.930 | 0.0672 |
| phase2_dav2 — occlusion slice | 0.881 | 0.897 | 0.889 | 0.0906 |

`phase2_dav2.yaml` is **architecturally byte-identical** to `phase2_mask2former.yaml` (same Swin-L COCO checkpoint, same hyperparameters, same seed=2026, same 20k iterations, same data) — it's meant to be a plain rerun, not a different experiment. Yet precision dropped ~4pts overall (~7.4pts on the occlusion slice), and depth MAE got **34% worse**. That's larger than ordinary GPU/cuDNN run-to-run noise should produce on its own.

**Most likely explanation, in order of probability:**
1. **The run was interrupted/restarted.** You killed a background pipeline process earlier in this session (`kill -- -307384` etc.) while iterating on the occlusion-sampling fix. If that kill landed mid-way through `phase2_dav2`'s training (Stage 2 of `run_all_pipelines.sh` runs Phase 1 → Phase 2 → Phase 3 in sequence, and Phase 1 alone is 55k iterations), a restart could have produced a `best.pth` from an under-converged or oddly-scheduled run rather than a clean full 20k-iteration training.
2. **Ordinary training variance** — real, but usually smaller than this gap for a 20k-iteration run at fixed seed.

**Recommended check (no code changes needed):** look at `runs/phase2_dav2/manifest.json` and the TensorBoard curves (`runs/phase2_dav2/tb/`) — confirm training actually ran the full 20k iterations without a gap/restart in the loss curve. If it was interrupted, a clean rerun of just Phase 2 (`train_phase2.py --config phase2_dav2.yaml`, after clearing `runs/phase2_dav2/`) would settle this.

## 3. Phase 3 — the central finding: refinement isn't helping, and is measurably worse exactly where it should help most

This is the same pattern in **both independent runs** (different Phase 1/Phase 2 checkpoints underneath each), which makes it a real, reproducible effect rather than noise in one run:

| | overall: refined vs base | occlusion slice: refined vs base |
|---|---|---|
| **phase3_current** | abs_rel +0.04% (a wash) | abs_rel **+2.2% worse**, sigma1 **−1.1% worse** |
| **phase3_dav2** | abs_rel +0.06% (a wash) | abs_rel **+3.1% worse**, sigma1 **−1.1% worse** |

Two things stand out:
- **Overall metrics are essentially unchanged** (differences in the 4th decimal place) — the refined map is nearly identical to the base Phase-1 map almost everywhere.
- **On the occlusion slice specifically — the one region Phase 3 exists to improve — refined is consistently worse than base**, not just neutral.

### Why this is happening

**a) Φo is zero-initialized on purpose** (`relation_head.py`: `nn.init.zeros_` on the output conv), so at initialization `E_obj = 0.5` exactly and `D̂ = D_obj` — a pure no-op on Phase-1 depth. The "overall ≈ base" result is exactly what you'd see if the model barely moved from that starting point. That points to **under-training**, not a broken mechanism.

**b) The pair-formation bug (mask IoU → box IoU) was fixed *during* this same session.** If `runs/phase3_current/` or `runs/phase3_dav2/` weren't fully cleared before the final run, part of the 25k iterations could have trained under the old, broken mask-IoU logic (`num_pairs ≈ 0` — almost no real gradient to Φo) before the fix took effect. Even after the fix, this leaves comparatively few real learning steps out of 25k to work with — plausibly not enough to converge to a net-beneficial correction, only enough to nudge slightly, and noisily, away from identity.

**c) The occlusion slice is inherently the *harder*, noisier-supervised region.** A single RGB-D sensor has no ground truth for hidden pixels, so the "guest" (farther, more-occluded) member of every pair is trained with sparser, noisier visible-only depth targets than an isolated, unoccluded person. With limited training (b), it's plausible Φo learned a small, net-negative correction specifically here — noise dominating before the model had enough signal to learn the right direction.

**d) Secondary, architectural:** the composite step upsamples a coarse 28×28 correction back into the true bounding-box size via plain bilinear interpolation (`composite_refined_depth`). This can introduce a small amount of boundary blur right at instance edges — exactly the pixels the occlusion-slice metric is measuring — independent of whether Φo's underlying correction is directionally sound.

**e) Phase 2 quality feeds Phase 3 directly.** `phase3_dav2` sits on top of the weaker `phase2_dav2` (Section 2), so its candidate/pair quality is also somewhat degraded — but `phase3_current` (on the *better* `phase2_run`) shows the **same** negative-occlusion-slice pattern, which means (b)/(c) are the dominant cause, not simply "worse Phase 2 in one of the two runs."

### Recommended next steps (analysis only — not yet implemented)

1. **Check `train/num_pairs` and `train/num_valid_pairs` in TensorBoard across the *entire* training curve** (not just early iterations) for both `phase3_current` and `phase3_dav2`. If the curve shows a step-change partway through (near-zero, then healthy), that confirms (b) — the run mixed pre-fix and post-fix training, and a clean retrain (clear the run folder, restart fresh, now that box_iou is the default) would likely tell a very different story.
2. **If the curves already look healthy throughout**, then 25k iterations may simply not be enough at LR=1e-6 for Φo to converge past noise — worth a longer run or a slightly higher Φo-specific LR before concluding the approach itself is flawed.
3. Separately verify Phase 2's `phase2_dav2` training wasn't interrupted (Section 2).

I haven't changed any training code for this — happy to implement whichever of these you want to chase first (e.g., a clean from-scratch Phase 3 retrain now that pair formation is fixed, or a longer run, or investigating the compositing step) once you've had a chance to look at the TensorBoard curves.

## 4. Temporal module (FlashDepth stage 2a) — full-split streaming results: correcting my earlier "null result" read

Earlier I called the temporal module a null result based on (a) a misleading 100-frame subset and (b) TAE alone. With the **full 10,400-frame streaming split** now in hand, that read needs correcting.

| metric | hdi_enhanced (baseline) | hdi_temporal (stage 2a) | Δ |
|---|---|---|---|
| abs_rel | 0.08200 | 0.07925 | **−3.35%** |
| rms_log | 0.11479 | 0.11244 | **−2.04%** |
| log10 | 0.03463 | 0.03355 | **−3.10%** |
| rms | 0.40707 | 0.40649 | −0.14% (flat) |
| sigma1 | 0.93143 | 0.93474 | +0.36% (flat) |
| temporal_alignment_error | 0.058678 | 0.058434 | −0.42% (noise, see below) |

**This is a real, if modest, positive result — not a null one.** Three independent dense-accuracy metrics (abs_rel, rms_log, log10) all improve by 2–3.4%, consistently in the same direction. That's convergent evidence, not one metric wobbling. It's also consistent with the earlier `check_temporal_module.py` finding that `proj_out` moved measurably away from its zero-init (`absmax=0.037`): the ConvGRU aligner learned a genuine per-frame correction to F_2, and that correction improves single-frame depth accuracy — even though **stage 2a only trains the ~1.84 M-parameter aligner** with the rest of the model frozen.

**TAE is still inconclusive, and that's a metric-floor problem, not a model problem.** The subset diagnostic (`hdi_enhanced_eval_streaming_subset100.json`) shows `temporal_alignment_error ≈ sqrt(gt_temporal_delta² + pred_temporal_delta²)` to within 0.15% — i.e. the prediction's frame-to-frame changes are statistically independent of the ground truth's. TAE against ZED stereo GT is floored by GT's own sensor noise; it cannot currently resolve whether temporal *consistency* improved, only that the −0.42% shift is within that floor. The dense-metric gain above did **not** need TAE to be visible — it shows up directly in per-frame accuracy against GT, which isn't subject to the same delta-of-two-noisy-signals problem.

**Revised recommendation:** report this as a genuine (modest) improvement in dense accuracy from the temporal aligner, with the caveat that its effect on frame-to-frame *smoothness* specifically remains unmeasured — TAE needs replacing with a GT-free (optical-flow self-consistency) metric before that specific question can be answered. Also worth running Stage 2b (unfreeze the spatial model, tiered LRs) given 2a already shows a real gain from a frozen backbone — 2b is where FlashDepth's own recipe expects the larger effect.

One open item: `hdi_temporal`'s `best.pth` was checkpoint-selected during training on the same misleading 100-frame per-frame subset (it doesn't measure temporal consistency). The full-split numbers above are still meaningful (dense abs_rel doesn't care how the checkpoint was selected), but `latest.pth` (iter 8000) hasn't been evaluated on the full streaming split yet and may show a different — possibly better — TAE/dense-accuracy balance.

## 5. Phase 1 loss ablation: BerHu underperforms SigLog — keep SigLog

An ad hoc run substituting BerHu for SigLog regression on the DAv2 profile (`hdi_dav2_berhu` — see that file's `config_note` for the assumption this documents about how the run was produced) comes out clearly worse on every metric:

| | abs_rel | rms | sigma1 |
|---|---|---|---|
| hdi_dav2 (SigLog) | 0.07781 | 0.38100 | 0.93833 |
| hdi_dav2_berhu (BerHu) | 0.08640 | 0.41091 | 0.92682 |
| Δ | **+11.0% worse** | **+7.9% worse** | −1.2% worse |

This is a clean, unambiguous negative ablation, consistent with the loss audit in `docs/IMPROVEMENTS.md` §5: BerHu is an L1/L2 hybrid tuned for less noisy, smaller-scale indoor depth (its original NYU-context); it isn't a better fit than SigLog for this dataset's noisier, longer-range ZED stereo GT. **Verdict: no reason to switch off SigLog** — this closes that item on the sweep list rather than opening a new direction.
