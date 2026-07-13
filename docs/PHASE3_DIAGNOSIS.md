# Phase 3 Diagnosis — Why Refined Depth Shows Patchwork Inside a Single Person

**Symptom (qualitative, TensorBoard):** inside one person's mask, different body
regions get noticeably different depth values ("multiple colors per person").
**Symptom (quantitative, `results/`):** refined ≈ base overall; refined
consistently *worse* than base on the occlusion slice, in both independent runs.

**Verdict: implementation issues, not a paper/architecture problem.** The audit
below walks every category; four concrete defects were found (D1–D4), all in
this project's own inferred glue (compositing + GT extraction), none in the
paper-specified math (Eq. 8–12 are implemented correctly). One config field
was documented but never wired (D5). The remaining contribution is expected
noise from an undertrained dense field, which the defects amplify into exactly
the observed patchwork.

## Audit by category

| Category | Status | Notes |
|---|---|---|
| Gradient flow | ✅ correct | `L_obj → d_hat → (Φo, d_obj → ROIAlign → depth_p1)`: both Φo and Phase 1 receive gradient. `L_dist → refined_layers → d_hat` likewise. Compositing is correctly outside the loss (no_grad). |
| Pair→GT mapping / matcher use | ✅ correct | `build_dense_gt_rois` LUT indexing (`m//2, m%2` vs `reshape(2P,4)` ordering) verified consistent; Hungarian indices applied per image. |
| ROI alignment | ✅ correct | Normalized boxes reconcile the 728×1288 / 736×1280 frames; batch-index flattening (`repeat_interleave(2)`) matches member interleaving. `aligned=True` used throughout. |
| Loss math (Eq. 10–12) | ✅ correct | SigLog form matches Phase 1's; masking to visible GT; Eq. 11 gap² form correct. |
| Inference path | ✅ correct | No matcher/GT dependency; empty-pair fallback returns base. |
| Feature extraction | ✅ correct | F_obj/G_obj channel assembly as designed. |
| **Compositing** | ❌ **D1, D2, D3** | See below — primary cause of the visual patchwork. |
| **Supervision** | ❌ **D4** | GT boundary dilution biases targets low at instance edges. |
| Loss config | ❌ **D5** | `min_valid_roi_px` documented but never enforced. |

## The defects

### D1 — Compositing pasted low-resolution *depth* instead of applying the correction *ratio* to full-resolution depth  *(primary)*

Eq. 9 algebraically simplifies: `D̂ = (2E−1)·D + D = 2E·D` — the refinement is a
**multiplicative ratio field `2E`** on the base depth. The old
`composite_refined_depth` instead ROIAligned the base depth down to 28×28,
multiplied there, bilinearly upsampled the *depth values* back to box size
(often several hundred pixels), and pasted them over the mask. Consequences:

- **The identity property was destroyed.** At `E = 0.5` (Φo's zero-init, and
  approximately its state when undertrained) the refinement is *supposed* to be
  a no-op — but the composite still replaced every masked pixel with a
  28×28-blurred reconstruction of its own depth. Refined could never equal
  base inside masks, only a degraded copy of it. This alone explains
  "occlusion slice: refined worse than base" even for a perfectly-behaved Φo.
- **All genuine within-person geometry (the thing the user wants preserved)
  was thrown away** and replaced by low-res blobs, modulated by whatever noise
  the E-field carried.

**Fix:** composite the upsampled **ratio** field: `refined = upsample(2E) × base`
inside the mask. Identity is now exact at `E=0.5`; base geometry is preserved
and merely modulated; corrections remain faithful to Eq. 9 (same math, applied
at full resolution instead of 28×28).

### D2 — Duplicate directional pairs wrote the same person twice with disagreeing fields

`build_pairs` followed the paper's wording literally ("for each main instance …
retain the nearest guest"), producing *directional* pairs — for two mutually
overlapping people both (A,B) **and** (B,A) survive. Person A then gets **two
different** `d_hat` patches (Φo's output is pair-context-dependent), both
written into A's mask and arbitrated **per-pixel by minimum value** (D3). Two
disagreeing noisy fields min-mixed per pixel is a direct patchwork generator —
seams appear wherever the two fields cross. It also double-counted every such
instance in the loss.

**Fix:** deduplicate to unordered pairs; canonical member order = nearer
(smaller `Dep`) first, giving Φo a consistent "channel 0 = occluder" semantic.
`[Reasonable Assumption]`, empirically motivated.

### D3 — Per-pixel min was the wrong arbitration semantic

Nearest-wins should arbitrate **between different instances** (occluder
overwrites occludee — exactly `data_engine/annotate.py::_flatten_id_map`'s
convention, which compares scalar per-instance *layers*). The old code compared
**raw per-pixel depth values across every write**, which (a) mixed multiple
estimates of the *same* person per-pixel (see D2), and (b) biased contested
pixels systematically nearer (min of noisy estimates < their mean).

**Fix:** aggregate all of an instance's pair-appearances into one ratio field
(mean), then write instances with **per-instance scalar layer** arbitration —
mirroring `_flatten_id_map` exactly. Within one person there is now exactly one
coherent correction field.

### D4 — GT ROI targets were diluted toward zero at instance boundaries

`build_dense_gt_rois` ROIAligned `depth × valid` (zeros outside the instance)
and thresholded the aligned valid-fraction at 0.5. Any boundary cell with
valid-fraction in [0.5, 1) kept `dt_valid=True` but had its depth target
scaled down by that fraction — e.g. a 3 m person's edge cells got targets of
1.6–2.9 m. SigLog then actively trained Φo to push depth *down* near instance
boundaries — i.e. it **taught the model to produce depth inconsistency at the
edges of body regions**, compounding D1–D3's patchwork.

**Fix:** normalized convolution — `dt_dense = ROIAlign(depth·valid) /
ROIAlign(valid)` — so every valid cell's target is the true local mean depth,
undiluted.

### D5 — `min_valid_roi_px` was documented but never enforced

`Phase3LossConfig.min_valid_roi_px = 16` claimed ROIs with fewer valid GT
pixels are skipped; the criterion never read it. ROIs with 1–2 valid pixels
contributed maximally noisy gradients. **Fix:** enforced in `Phase3Criterion`.

## Remaining (non-defect) contribution: undertrained dense E-field

Where GT is invalid (sensor holes — common on people at exactly the occluded
regions), the dense E-field is unconstrained by `L_obj` and drifts; with few
effective training steps (zero-init identity start + the mask-IoU pair
starvation that was only fixed mid-way through the recorded runs) the field is
mostly noise around 0.5. Post-fix this noise now modulates (not replaces)
full-res geometry, and a new `head.composite_ratio: "scalar"` switch composites
each instance with its single mask-mean ratio — maximal within-person coherence
while preserving all base geometry (compositing is not paper-specified, so this
is a legitimate implementation choice; `"dense"` remains the default and the
loss always trains the dense field either way).

## Expected outcome

- With **no retraining**, existing checkpoints should already look
  substantially better: identity regions are now truly untouched, one field per
  person, no min-mixing, and `composite_ratio: scalar` available for maximum
  coherence.
- A **clean retrain** (run folders cleared, box-IoU pairing + D1–D5 fixes all
  active from step 0) is still recommended before judging the method: the
  recorded runs trained partly under the broken pairing and the boundary-diluted
  targets.
