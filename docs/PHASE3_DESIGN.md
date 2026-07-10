# Phase 3 — Occlusion-Aware Depth Refinement: Research-Level Design Blueprint

**Paper:** *Instance-Level Video Depth in Groups Beyond Occlusions* (ICCV 2025), Sec. 4.2.2, Eq. 8–12; training recipe Sec. 4.3.
**Status:** Design only — no code committed. To be reviewed before implementation.
**Provenance tags:** `[Paper Specified]` · `[Strongly Inferred]` · `[Reasonable Assumption]`

The primary source of truth for this design is the paper itself; the pre-Phase-3 summary is treated as *implementation history*, not ground truth. Where the two disagree, the paper wins and the disagreement is called out explicitly in §0.

---

## 0. Corrections to the pre-Phase-3 summary (verified against the paper)

Before designing Phase 3, five points in the accumulated summary must be corrected or sharpened, because they change Phase 3's design. Each is verified directly against the paper text.

**0.1 `[Paper Specified]` — Phase 3 has an explicit training recipe the summary omits.**
Sec. 4.3, "Occlusion-Aware Joint Refinement": *"we reverse the freezing strategy—fixing the instance decoder while fine-tuning both the depth encoder and decoder for 25k iterations with a lower learning rate of 1×10⁻⁶."* The summary's Phase-3 preview never states this. It is central:
- Phase 2 (Mask2Former instance decoder) is **frozen** in Phase 3.
- Phase 1 (DINOv2 depth encoder + Depth Range Decoder + Eq. 1–4 refinement) is **fine-tuned** at LR `1e-6`.
- The new occlusion MLP `Φo` is trained from scratch.
- Duration: **25k iterations** (same as Phase 2).
This is not optional and is baked into the training-pipeline section (§8).

**0.2 `[Strongly Inferred]` — `r_d = 5` and `max_depth = 10` are better-supported than "[Reasonable Assumption]".**
Fig. 2 caption: GID depth is visualized *"within a range of 0.01 to 10.0 meters"* → `max_depth = 10 m` is the GID sensor range (paper-grounded for GID; still the user's own value for their sensor). Table 5 ablates depth-range partitioning and finds **2-meter partitioning is best**. With `max_depth = 10`, 2-meter bins ⇒ `r_d = 5`. So the current `rd: 5` is the paper's *best* configuration, not a guess. **Recommendation:** parameterize as `r_d = round(max_depth / 2.0)` so the "2-meter partition" invariant survives any change to the user's `max_depth`. Phase 3 does not change `r_d`, but this matters because Phase 3 fine-tunes the same bin-refinement module.

**0.3 `[Paper Specified] + physics` — there is no ground truth for truly-hidden pixels.**
A single calibrated RGB-D sensor records depth of the *front-most* surface only. In an occluded region the sensor sees the **occluder**, never the occludee's hidden geometry. Therefore:
- `DT_obj` (GT object depth) is defined only on each instance's **visible** pixels (exactly how `data_engine/annotate.py::_depth_layer` computes the GT depth layer — mean of *valid* depth in the mask).
- `L_obj` (Eq. 10) can only be supervised where GT is valid, i.e. `gt_depth > 0` ∧ inside the instance mask — the same masking convention Phase 1 already uses.
- The refinement's measurable benefit (Table 4: H 0.4188 RMS → H+I 0.397; Table 6) comes from **correcting depth-boundary bleed and visible-region instance-consistency**, not from hallucinating hidden geometry with direct supervision. This shapes the loss masking in §7 and the honest framing of what "beyond occlusions" can and cannot mean under this sensor.

**0.4 `[Reasonable Assumption]` — the independent-Swin-L choice (Phase 2 Option B) is *consistent with*, not in tension with, the Phase 3 freezing recipe.**
The paper's prose ("freeze the depth encoder and train the instance decoder") reads like a *shared* DINOv2 encoder feeding the instance decoder. The summary deviated to an independent COCO Swin-L. Re-examined against Phase 3: in Phase 3 the depth encoder is **fine-tuned** while the instance decoder is **frozen**. If the instance decoder shared the depth encoder, fine-tuning that encoder would silently shift the features under a frozen decoder → its masks would drift uncontrollably. The **independent-backbone choice removes this conflict**: fine-tuning Phase 1 in Phase 3 leaves Phase 2's masks bit-identical. So the Option-B deviation is not merely defensible — it is what makes the paper's own 3-phase freeze strategy coherent to reproduce. This is a positive finding; keep Option B.

**0.5 `[Paper Specified]` — the two Phase-3 sub-modules map to Fig. 4 labels precisely.**
Fig. 4's lower row has two named boxes: **"Instance Layer Decoder"** (= Phase 2's Mask2Former, already built) and **"Occlusion Pair Relation Reasoning"** (= Phase 3's new `Φo`). Phase 3 = *candidate/pair construction front-end* (reads frozen Phase-2 outputs) + *ROI-Align feature extraction* + *`Φo` relation reasoning* (Eq. 8–9) + *composite-back* + *losses* (Eq. 10–12) + *depth-branch fine-tuning*. "Instance Layer Decoder" is **not** new work in Phase 3.

---

## 1. Overall architecture

```
                         ┌──────────────── PHASE 1 (TRAINABLE, LR 1e-6) ────────────────┐
   RGB frame ───────────►│ DINOv2 ViT-L/14 → Depth Range Decoder → Eq.1–4 refinement    │
        │                │ OUT: depth_final (B,1,H,W)   feat_final F_2 (B,C,h2,w2)       │
        │                └──────────────────────────────────────────────────────────────┘
        │                         ┌──────────── PHASE 2 (FROZEN, no grad) ───────────────┐
        └────────────────────────►│ Swin-L Mask2Former + DepthLayerHead                  │
                                  │ OUT per query: mask_logits, class_logits, Dep_i, box │
                                  └──────────────────────────────────────────────────────┘
                                             │ (cacheable — frozen)
                    ┌────────────────────────┴─────────────────────────────────────────┐
                    │                    PHASE 3  (NEW)                                 │
                    │  (a) Candidate filter: cat-conf>0.9 ∧ mask-conf>0.8               │
                    │  (b) Pair build: for each MAIN, overlapping (IoU>0.1),            │
                    │        keep depth-nearest GUEST → P pairs                         │
                    │  (c) ROIAlign (per instance, normalized boxes):                   │
                    │        F_obj  = ROIAlign(feat_final, box)      → 2×C×Hp×Wp        │
                    │        D_obj  = ROIAlign(depth_final, box)     → 2×1×Hp×Wp        │
                    │        G_obj  = [mask_logits, norm-coords, global-depth] ROIAlign │
                    │  (d) Φo relation reasoning: E_obj = σ(Φo([F_obj,G_obj]))          │
                    │  (e) Refine:  D̂_obj = (2·E_obj − 1)·D_obj + D_obj                 │
                    │  (f) Composite D̂_obj back into depth_final (nearest-wins)         │
                    │  Losses: L_ref = λ1·SigLog(D̂,DT) + λ2·L_dist                      │
                    └──────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                        Refined dense metric depth  +  refined per-instance depth layers
```

`[Paper Specified]`: steps (a) thresholds, (b) IoU>0.1 + nearest-guest, (c) ROIAlign of F_obj/G_obj, (d) Eq. 8, (e) Eq. 9, losses Eq. 10–12, and the freeze/fine-tune split.
`[Strongly Inferred]`: the composite-back step (f) — required to produce Fig. 4's dense "Final Depth Pred" and to explain Table 4/6's dense-metric gains, but the paper never writes the compositing rule.
`[Reasonable Assumption]`: `Φo`'s internal architecture, `Hp×Wp`, `G_obj` channel layout, whether refinement is dense or scalar (§5 — the single most important open decision).

---

## 2. Data flow (one training iteration)

1. `image (B,3,H1,W1)` → **Phase 1 forward (grad on)** → `depth_final (B,1,H1,W1)`, `feat_final (B,C,h2,w2)`.
2. `image (B,3,H2,W2)` → **Phase 2 forward (no-grad / cached)** → `mask_logits (B,N,H2,W2)`, `class_logits (B,N,2)`, `depth_layers Dep (B,N)`. Boxes derived from `mask_logits.sigmoid()>0.5`.
3. **Hungarian matching** (`Phase2HungarianMatcher`, reused) between Phase-2 predictions and GT `targets` → `indices` (needed for supervision only; skipped at inference).
4. **Candidate filter** → per-image list of surviving query indices.
5. **Pair build** → `pair_query_idx (P,2)`, `batch_index (P,)`.
6. **ROIAlign** feature extraction on the P pairs (normalized boxes; see §6 for cross-resolution handling).
7. **`Φo`** → `E_obj`; **Eq. 9** → `D̂_obj`.
8. **Composite** `D̂_obj` into a refined copy of `depth_final`.
9. **`build_refine_targets`** (already implemented) → `dt (P,2)`, `valid (P,)` (GT depth layers for L_dist); **companion dense-GT-ROI extraction** → `DT_dense` per instance (for L_obj).
10. **Losses** Eq. 10–12 (+ optional anti-forgetting holistic term, §7.4) → backprop into Phase 1 + `Φo`.

At **inference** steps 3 and 9 are dropped; step 8's composited map plus the per-instance refined layers are the output.

---

## 3. Inputs required from Phase 1

Consumed **live** (Phase 1 is trainable in Phase 3, so nothing here can be cached):

| Field (from `HolisticDepthOutput`) | Shape | Role in Phase 3 | Provenance |
|---|---|---|---|
| `feat_final` (F_2) | `(B,C,h2,w2)` native | ROIAlign source for `F_obj` — the depth-feature evidence | `[Paper Specified]` ("Multi-scale depth features … via ROI alignment") |
| `depth_final` | `(B,1,H1,W1)` | (i) `D_obj` = ROIAlign source; (ii) the base map corrections are composited onto; (iii) `global depths` channel of `G_obj` | `[Paper Specified]` ("global depths" ∈ G_obj; Fig. 4 "Init Depth Map" → ROIAlign) |
| `feat_stride()` | `(sh,sw)` | ROIAlign `spatial_scale = 1/stride` for `feat_final` | `[Paper Specified]` mechanism (Mask R-CNN [23]) |

**Note on "multi-scale":** the paper says *multi-scale* depth features. `feat_final` exposes only the finest level F_2. Two readings:
- (A) `[Reasonable Assumption]` ROIAlign F_2 only (current contract). Simplest; keep for the faithful baseline.
- (B) `[Strongly Inferred]` ROIAlign all three decoder levels F_0/F_1/F_2 and concatenate (true multi-scale). This needs a one-line **contract bump to `HolisticDepthOutput` v1.2** to also expose `feat_levels: [F_0,F_1,F_2]` (they already exist inside `DepthRangeFeatures.levels`; only the dataclass drops them today). **Recommendation:** expose all three now (cheap, in-process), default the faithful profile to F_2-only, and ablate (B) — this directly honors the word "multi-scale" without committing compute.

## 4. Inputs required from Phase 2

Consumed as **fixed** tensors (Phase 2 frozen → cache to disk, §8.3). From `Phase2Output` + helpers:

| Field | Shape | Role | Provenance |
|---|---|---|---|
| `mask_logits` | `(B,N,H2,W2)` | (i) candidate region + IoU overlap detection; (ii) `mask_confidence()>0.8` filter; (iii) box derivation; (iv) `mask logits` channel of `G_obj` | `[Paper Specified]` |
| `scores()` (class conf) | `(B,N)` | `>0.9` category filter | `[Paper Specified]` |
| `depth_layers` (Dep_i) | `(B,N)` | (i) "nearest-in-depth" guest selection; (ii) optional `D_obj` seed (scalar reading, §5) | `[Paper Specified]` (Eq. 5–7 produce Dep_i; used for pair ordering) |
| derived `boxes` | `(B,N,4)` | ROIAlign boxes for both branches | `[Strongly Inferred]` (Fig. 4 ROIAlign needs a box; paper doesn't say box source → derive from predicted mask) |

Phase 3 does **not** consume `query_embeddings` (kept in the contract only for possible future use).

---

## 5. THE central design decision — dense vs. scalar refinement (`E_obj`, `D_obj`, `D̂_obj`)

This is the one genuinely ambiguous, high-impact choice. The paper writes `F_obj ∈ ℝ^{2×C×Hp×Wp}` (spatially resolved), Eq. 8 `E_obj = σ(Φo([F_obj,G_obj]))`, Eq. 9 `D̂_obj = (2E_obj−1)·D_obj + D_obj`, Eq. 10 `L_obj = SigLog(D̂_obj, DT_obj)`, Eq. 11 `L_dist = Σ‖(D̂_i−D̂_j)²−(DT_i−DT_j)²‖`.

**Reading D — dense per-pixel (RECOMMENDED, primary).** `Φo` is a small per-location head (1×1 convs) over the ROI grid. `E_obj ∈ (2×Hp×Wp)`, `D_obj = ROIAlign(depth_final, box)` dense, `D̂_obj` dense. `L_obj` = SigLog over valid GT pixels in the ROI. For `L_dist`, reduce `D̂_i` = mask-weighted mean of `D̂_obj[i]` (scalar). Composite dense `D̂_obj` back into the full map.
- *For:* `F_obj` is explicitly spatial; `G_obj` carries dense mask logits + dense global depth + per-pixel coordinates — pointless if `Φo` immediately pools; Table 4/6 report **dense** REL/RMS gains, which require a dense corrected map; "L_obj broadly optimizes depth values **in occluded regions**" (spatial language).
- *Against:* more moving parts (compositing, dense GT ROI extraction); the existing `build_refine_targets` (scalar) covers only `L_dist`.

**Reading S — scalar per-instance (fallback / ablation).** `Φo` pools `F_obj`→ scalar `E_obj` per instance; `D_obj = Dep_i` (Phase 2 scalar); `D̂_i` scalar; `L_obj` = SigLog over the batch's scalar refined layers. Exactly matches the current `build_refine_targets`.
- *For:* simplest; matches existing stub; differentiable by construction.
- *Against:* cannot by itself move the **dense** REL/RMS that the paper's ablations report, unless paired with a scalar-rescale-composite (Reading H).

**Reading H — hybrid (scalar `E`, dense `D`).** `E_obj` scalar per instance, but applied to the **dense** `D_obj` = ROIAlign(depth_final): `D̂_obj = (2E−1)·D_obj + D_obj` = per-instance multiplicative rescale of the dense depth. Improves dense metrics with a scalar head; `L_dist` naturally scalar.

**Decision:** implement **Reading D** as the faithful primary (most consistent with the spatial `F_obj` and the "occluded regions" language), keep **Reading H** as a first-class config switch (one flag: `refine_granularity: dense|scalar`), and log **Reading S**-style scalar corrections as a diagnostic. Rationale mirrors the project's existing philosophy (faithful primary + flagged ablation), and lets the eventual dense-vs-scalar question be settled empirically on the occlusion slice rather than by assertion. **Missing detail resolved by:** Mask R-CNN [23] (dense per-ROI heads are the norm for spatial ROI outputs) + Mask2Former's dynamic mask head (per-pixel query-conditioned prediction) as the architectural template for a dense `Φo`.

---

## 6. ROI feature extraction & feature-fusion strategy

**6.1 Boxes & the cross-resolution problem `[Strongly Inferred]`.** Three resolutions coexist: sensor-native (≈720×1280), Phase-1 `image_size` (728×1288), Phase-2 `image_size` (736×1280). `feat_final` and `mask_logits` live in *different* coordinate frames. Resolve by working in **normalized `[0,1]` box coordinates**: derive each box from `mask_logits.sigmoid()>0.5` (Phase-2 frame), normalize by `(H2,W2)`, then for each branch multiply by that branch's own feature-grid size. `torchvision.ops.roi_align` with `spatial_scale` per branch handles this; the box tensor is shared, expressed in a common normalized frame. **Recommendation:** in Phase 3, run **both branches at one common input resolution** (e.g. adopt Phase-2's `(736,1280)` for the whole model, or a shared `(H,W)` divisible by both 14 and 32) to eliminate a class of subtle alignment bugs; fall back to normalized-coord ROIAlign only if the two must differ.

**6.2 The pair dimension "2" `[Paper Specified] shape / [Reasonable Assumption] mechanics`.** `F_obj ∈ ℝ^{2×C×Hp×Wp}` = ROIAlign the *same* `feat_final` to the **two boxes** (main, guest): `F_obj[0]=ROIAlign(feat, box_main)`, `F_obj[1]=ROIAlign(feat, box_guest)`. Each instance is aligned to its **own** box (matches Fig. 4's two separate "ROI Align" arrows). The two ROIs cover different image regions but share the `Hp×Wp` index space; cross-instance relation is supplied to `Φo` through `G_obj`'s **normalized coordinates** (so the MLP knows where each ROI pixel actually is) plus each instance's mask + global depth. This is why the paper deliberately lists "normalized coordinates" inside `G_obj` — it is the geometric glue between two independently-cropped ROIs.

**6.3 `G_obj` channel layout `[Paper Specified] contents / [Reasonable Assumption] exact channels`.** Per ROI location, per instance:
- `mask logits` — `ROIAlign(mask_logits, box)`, 1 ch. `[Paper Specified]`
- `normalized coordinates` — the `(x,y)∈[0,1]` full-frame position of each ROI cell, 2 ch. `[Paper Specified]` (content) / channel-count `[Reasonable Assumption]`
- `global depths` — `ROIAlign(depth_final, box)`, 1 ch. `[Paper Specified]`
→ `G_obj ∈ ℝ^{2×4×Hp×Wp}`.

**6.4 Fusion into `Φo` `[Reasonable Assumption]`.** Concatenate along channel: per instance `[F_obj ; G_obj] ∈ ℝ^{(C+4)×Hp×Wp}`. To make it genuine *relation* reasoning (both instances jointly determine each other's correction), stack the pair into `(2·(C+4))×Hp×Wp` and let `Φo` (2–3 layers of 1×1 conv + GELU) emit a 2-channel error map `(2×Hp×Wp)` (main, guest). Sigmoid → `E_obj ∈ (0,1)`. This is the minimal design faithful to "MLP `Φo`" (not attention) while still coupling the pair. **Alternative to ablate:** a light cross-attention between the two ROI token sets (DETR/Mask2Former-style) if the 1×1-conv MLP under-reasons about the relation — resolved by reference to DETR [6] / Mask2Former [12] cross-attention if needed.

**6.5 `Hp×Wp` `[Reasonable Assumption]`.** Paper gives no value. Use `Hp=Wp=28` (Mask2Former's mask resolution is `H/4`; 28 is a common ROIAlign size, balances boundary detail vs. compute). Configurable; ablate {14,28,56}. **Missing detail resolved by:** Mask R-CNN (14×14 for boxes, 28×28 for masks) — 28 chosen because occlusion boundaries are the object of interest.

---

## 7. Losses & supervision strategy

**7.1 `L_obj` — Eq. 10 `[Paper Specified] form / [Strongly Inferred] masking`.** `SigLog(D̂_obj, DT_obj)`, reusing the **existing `SigLogLoss`** (`losses/hdi_losses.py`) unchanged (same Eigen scale-invariant-log the paper cites for Eq. 10). Masking (§0.3): valid only where `gt_depth>0` **and** inside the matched GT instance mask, per instance ROI. `DT_dense` = `ROIAlign(gt_depth_masked_to_gt_instance, box)`. Companion to `build_refine_targets`, to be added.

**7.2 `L_dist` — Eq. 11 `[Paper Specified]`.** Per pair: `‖(D̂_main−D̂_guest)² − (DT_main−DT_guest)²‖₁`. `D̂_*` = scalar mask-mean of the refined ROI depth; `DT_*` from `build_refine_targets`'s `dt (P,2)` (already implemented and correct for this). Averaged over `valid` pairs.

**7.3 Total — Eq. 12 `[Paper Specified] structure / [Reasonable Assumption] weights`.** `L_ref = λ1·L_obj + λ2·L_dist`. Paper gives no `λ` values; Table 6 shows `L_obj` dominates and `L_dist` is secondary. **Start `λ1=1.0, λ2=0.5` (weight `L_dist` below `L_obj`), sweep.** Flagged as a new hyperparameter, not assumed.

**7.4 Anti-forgetting holistic regularizer `[Reasonable Assumption] — opt-in`.** Fine-tuning the *entire* Phase-1 depth branch at 1e-6 under a loss that is only defined on paired-instance ROIs risks global drift in non-instance regions. Add optional `μ·L_holistic` (the existing `HDILoss` on the full map). **Default OFF in the faithful profile** (Eq. 12 verbatim), **ON at small `μ` (e.g. 0.1) in an "enhanced" profile** — the exact same faithful/enhanced split the project already uses for the disparity aux. This is a deviation, flagged as such.

**7.5 Supervision availability summary.**
- Dense `L_obj`: visible-GT pixels of each matched instance (no hidden-pixel GT — §0.3).
- Scalar `L_dist`: matched-instance GT depth layers (mean visible depth).
- Pairs with any unmatched member are dropped (`build_refine_targets.valid`) — no fabricated targets.

---

## 8. Training pipeline

**8.1 Freeze/train split `[Paper Specified]`.**
- Phase 2 (`Phase2Model`): `.eval()`, `requires_grad_(False)`.
- Phase 1 (`HolisticDepthModel`): trainable — **including DINOv2** (paper: "fine-tuning both the depth encoder and decoder"). Note this differs from Phase 1's own stage where the encoder is trainable too, but here at 10× lower LR.
- `Φo`: trainable (fresh).

**8.2 Optimizer/schedule `[Paper Specified] LR+iters / [Reasonable Assumption] rest`.** Reuse the generic `Trainer` + a Phase-3 `build_optimizer_fn` (like `train_phase2.py` already does for a non-`backbone`-prefixed model). Param groups: depth-encoder/decoder at `lr=1e-6` (paper), `Φo` at a higher head LR (e.g. `10×`, matching Phase 1's `head_lr_mult` convention — `[Reasonable Assumption]`). `total_iters=25000` (paper), AdamW + poly-decay `0.9`, bf16 — all inherited conventions.

**8.3 Efficiency — cache the frozen Phase-2 outputs `[Reasonable Assumption]`.** Because Phase 2 is frozen and its inputs (RGB) are fixed, precompute per training frame: top-`K` (K≈20) candidate `mask_logits` (fp16), `class_logits`, `depth_layers`, and derived boxes → sharded `.npz` (mirror `scripts/infer_sequence.py`). This removes Swin-L from the Phase-3 training loop (large memory + compute win) and is exactly valid *only because* Phase 2 is frozen. **Phase 1 outputs are NOT cacheable** (it's being fine-tuned) — recompute each step with grad. The Hungarian `indices` can also be precomputed against fixed GT since only Phase-2 predictions (frozen) and GT (fixed) enter the match.

**8.4 Batched variable-instance-count pairs `[Reasonable Assumption] — engineering`.** Instances/pairs per frame vary. Flatten all pairs across the batch into a single `(ΣP, ...)` tensor with a `batch_index` (already the signature `build_refine_targets` expects), ROIAlign in one call, run `Φo` once, scatter losses by `batch_index`. No python-loop over instances in the hot path.

**8.5 Compositing back `[Strongly Inferred] — engineering`.** Paste each refined ROI depth into a clone of `depth_final` using the instance's thresholded mask as the write stencil. Overlaps resolved **nearest-depth-wins** (occluder overwrites), reusing `data_engine/annotate.py::_flatten_id_map`'s exact convention for consistency between GT construction and inference output. Optional soft alpha (mask confidence) at boundaries to avoid seams. This composited map is what dense eval scores.

---

## 9. Inference pipeline

1. RGB → Phase 1 → `depth_final`, `feat_final`.
2. RGB → Phase 2 (frozen) → masks, classes, `Dep_i`, boxes.
3. Filter (cat>0.9 ∧ mask>0.8).
4. Build pairs (IoU>0.1, depth-nearest guest).
5. ROIAlign → `Φo` → `E_obj` → Eq. 9 → `D̂_obj` (main & guest).
6. Composite `D̂_obj` into `depth_final` (nearest-wins).
7. **Output:** `RefinedDepthOutput` = refined dense metric map + per-instance refined depth layers + the pairs used.
No matcher, no GT, no losses. Regions with no confident/paired instance keep Phase-1 depth verbatim (so non-crowded scenes degrade gracefully to Phase-1 quality — consistent with Table 3's marginal NYU gains).

**Temporal note `[Paper Specified] baseline`.** The paper's Sec. 4.2 refinement is **per-frame**; temporal coherence in the paper comes from the *dataset's* tracking identities, not a temporal module in the refinement. So per-frame Phase 3 is the faithful baseline. Joint spatio-temporal masked attention (Mask2Former-Video, Eq. 1–4 of that report) is a **deferred optional extension**, measured against the per-frame baseline on the occlusion slice — not part of the faithful reproduction.

---

## 10. Evaluation strategy

**10.1 Primary — dense depth, paper protocol.** Reuse `utils/metrics.py` (REL, RMS, RMSlog, Log10, σ1–3) on the **composited refined map** over the whole frame. Compare three rows to reproduce the paper's Table 4 ablation logic:
- Phase-1 only (`H`) — the existing HDI eval.
- Phase-1 + Phase-3 (`H+I`) — refined map.
- Target: paper's `Baseline+H+I` (REL 0.045 / RMS 0.397 on GID). **Compare against the ablation rows, not necessarily the full SOTA number**, since the user's dataset ≠ GID.

**10.2 Occlusion-focused slice (the decisive signal).** Restrict metrics to frames with ≥2 overlapping instances (IoU>0.1) and, within them, to instance/overlap pixels. This is where Phase 3 must beat Phase 1; it is also the decision signal for the deferred temporal module.

**10.3 Per-instance depth-layer error.** MAE/REL of refined `Dep_i` vs GT depth layer, split by "paired" vs "isolated" instances — isolates the relation-reasoning contribution.

**10.4 Loss ablations (reproduce Table 6).** `L_obj` only, `L_dist` only, both. Expect `L_obj` dominant, `L_dist` secondary — a faithfulness check on the whole stage.

**10.5 Design ablations.** dense (Reading D) vs scalar (Reading H); F_2-only vs multi-scale `F_obj`; `Hp×Wp∈{14,28,56}`; guest rule (nearest-to-main-depth vs frontmost); anti-forgetting `μ` on/off.

---

## 11. Expected outputs

- **`RefinedDepthOutput` contract (new, v1.0)** mirroring the existing dataclass+`contract_version` pattern: `refined_depth (B,1,H,W)`, `instance_depth_layers` (per surviving candidate), `pairs` (main/guest indices + IoU), `image_hw`, `contract_version`. Deliberately over-exposed, same philosophy as the Phase 1/2 contracts.
- **Refined dense metric depth map** (primary deliverable) — occlusion-corrected, otherwise = Phase-1 depth.
- **Per-instance refined depth layers** — for per-instance eval and downstream 3D use.
- **Trained artifacts:** fine-tuned Phase-1 weights + `Φo` weights (Phase-2 weights unchanged).

---

## 12. Integration with existing Phase 1 / Phase 2

**New modules (proposed layout, mirrors `models/hdi` & `models/phase2`):**
```
instancedepth/models/phase3/
  candidates.py     # filter (0.9/0.8) + pair build (IoU>0.1, nearest guest)
  roi_extract.py    # normalized-box ROIAlign of F_obj / D_obj / G_obj (torchvision.ops.roi_align)
  relation_head.py  # Φo (Eq. 8) + Eq. 9 refine + composite-back
  model.py          # Phase3Model: wires frozen Phase2Model + trainable HolisticDepthModel + Φo
  output.py         # RefinedDepthOutput (contract v1.0)
instancedepth/losses/phase3_losses.py   # L_obj (reuses SigLogLoss), L_dist (Eq. 11), L_ref (Eq. 12)
instancedepth/configs/phase3_config.py  # Phase3Config (freeze flags, λ1/λ2, Hp/Wp, thresholds, refine_granularity)
instancedepth/configs/phase3.yaml       # faithful profile (+ phase3_enhanced.yaml for μ / multi-scale)
instancedepth/engine/train_phase3.py    # build models, cache Phase-2, custom build_optimizer_fn, reuse Trainer
instancedepth/engine/evaluate_phase3.py # §10 metrics
```

**Reused as-is (no changes):** `Trainer`; `SigLogLoss`; `Phase2HungarianMatcher`; `build_refine_targets` (covers `L_dist`); `utils/metrics.py`; checkpoint/manifest/seed utils; `_flatten_id_map` convention (imported for compositing).

**Minimal changes to existing code:**
1. `HolisticDepthOutput` **v1.1→v1.2**: add `feat_levels: [F_0,F_1,F_2]` for optional multi-scale `F_obj` (they already exist inside the decoder; only the dataclass drops them). Backward compatible (additive) — bump the version string, keep `feat_final` as the F_2 alias.
2. `build_refine_targets` **companion**: add a sibling that also returns `DT_dense` (ROI GT depth) for `L_obj` in Reading D. Non-breaking (new function).
3. Config: a `common_input_hw` used by both branches in Phase 3 (§6.1) — no change to Phase-1/2 configs, just a Phase-3 wiring choice.

No change to Phase 1's or Phase 2's training entry points; Phase 3 composes them.

---

## 13. Provenance ledger (every major Phase-3 decision)

| # | Decision | Paper support | Ours | Tag |
|---|---|---|---|---|
| 1 | Freeze Phase 2, fine-tune Phase 1 @1e-6, 25k it | Sec. 4.3 explicit | same | `[Paper Specified]` |
| 2 | Candidate filter cat>0.9 ∧ mask>0.8 | Sec. 4.2.2 explicit | same | `[Paper Specified]` |
| 3 | Pair: overlap IoU>0.1, keep depth-nearest guest | Sec. 4.2.2 explicit | "nearest" = nearest to main's `Dep` | `[Paper Specified]` rule / `[Strongly Inferred]` "nearest-to-main" |
| 4 | ROIAlign F_obj / G_obj / D_obj | Eq. 8 + Fig. 4 + [23] | torchvision roi_align, normalized boxes | `[Paper Specified]` mechanism |
| 5 | `G_obj` = {mask logits, norm coords, global depths} | Sec. 4.2.2 explicit list | 1+2+1 ch layout | `[Paper Specified]` contents / `[Reasonable Assumption]` channels |
| 6 | `E_obj = σ(Φo([F_obj,G_obj]))` | Eq. 8 | dense 1×1-conv MLP, pair-coupled | `[Paper Specified]` form / `[Reasonable Assumption]` internals |
| 7 | `D̂ = (2E−1)·D + D` | Eq. 9 | same; `D` = dense ROI depth | `[Paper Specified]` |
| 8 | dense vs scalar refinement | ambiguous | dense primary + scalar ablation | `[Reasonable Assumption]` (§5) |
| 9 | `L_obj = SigLog(D̂,DT)` | Eq. 10 + [16] | reuse `SigLogLoss`, mask to visible GT | `[Paper Specified]` form / `[Strongly Inferred]` masking |
| 10 | `L_dist` squared-gap consistency | Eq. 11 | scalar, over valid pairs | `[Paper Specified]` |
| 11 | `L_ref = λ1 L_obj + λ2 L_dist` | Eq. 12 | λ1=1, λ2=0.5 start | `[Paper Specified]` form / `[Reasonable Assumption]` weights |
| 12 | Composite refined ROIs into dense map | implied by Fig.4 + Table 4/6 | nearest-wins paste | `[Strongly Inferred]` |
| 13 | Anti-forgetting holistic term | none | opt-in, off in faithful | `[Reasonable Assumption]` (deviation) |
| 14 | Multi-scale `F_obj` (F0/F1/F2) | "multi-scale" wording | expose all 3, ablate | `[Strongly Inferred]` |
| 15 | Cache frozen Phase-2 outputs | none | disk cache, K≈20 | `[Reasonable Assumption]` (efficiency) |
| 16 | Per-frame (no temporal module) | Sec. 4.2 is per-frame | per-frame baseline | `[Paper Specified]` baseline |
| 17 | Common input resolution across branches | none | unify to avoid alignment bugs | `[Reasonable Assumption]` (engineering) |

---

## 14. Open questions to settle empirically (not by assertion)

1. dense vs scalar `E_obj` (§5) — decided on the occlusion slice.
2. `λ1/λ2`, `Hp×Wp`, guest rule, multi-scale `F_obj` — sweeps in §10.5.
3. Does 1e-6 whole-branch fine-tuning drift global depth enough to need `μ>0`? — watch non-instance-region REL during Phase-3 training.
4. Is per-frame ownership sufficient, or is the temporal module (Mask2Former-Video) needed? — occlusion-slice per-instance error, paired vs isolated.

---

## 15. Engineering-risk register

| Risk | Mitigation |
|---|---|
| Two large backbones resident (Phase1 grad + Phase2 frozen) | Cache Phase-2 offline (§8.3) → only Phase 1 in the training graph |
| Compositing seams at mask boundaries | nearest-wins + soft-alpha; reuse `_flatten_id_map` rule |
| Cross-resolution ROI misalignment | unify input resolution (§6.1); assert box-frame consistency |
| Variable pair counts break batching | flat `(ΣP,…)` + `batch_index` (§8.4) |
| No GT for hidden pixels misread as a bug | supervision masked to visible GT by design (§0.3, §7.5) |
| Global-depth forgetting under narrow loss | optional `μ·L_holistic` (§7.4) |
