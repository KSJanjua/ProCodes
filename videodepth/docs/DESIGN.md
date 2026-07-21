# videodepth — Temporal Consistency + Video-Aware Occlusion Handling

The paper-facing extension package. **InstanceDepth (Liang et al., ICCV 2025)
is entirely per-frame** — a close reading of the paper (Sec. 4, Eq. 1–12)
confirms no mechanism ever connects two frames at inference: "video" refers to
the GID *dataset* (DEVA tracking IDs in the annotations) and to *within-frame*
instance consistency. Depth flicker between frames is therefore not a bug in
the reproduction — it is a gap in the method. This package fills that gap
with two contributions sized for a paper:

1. **A trained streaming temporal stage** (flicker in the *dense* depth).
2. **Temporal amodal completion + a bounded pair-attention Φo**
   (flicker and occlusion error in the *instance* depth).

Everything is additive: `instancedepth/` is untouched; every module here
wraps or subclasses it.

---

## 1. Why the previous temporal attempt measured nothing

A prior diagnosis established three compounding causes, all fixed here:

| # | Root cause | Fix in this package |
|---|------------|---------------------|
| 1 | **No temporal loss.** The ConvGRU had state but the training loss was purely per-frame — no gradient ever rewarded smoothness, so the zero-init module correctly stayed a no-op. | `losses/temporal_losses.py` — Temporal Gradient Matching (TGM). |
| 2 | **No motion in training clips.** Frame-to-frame motion in this data is tiny; it becomes visible over ~100-frame spans (user-confirmed), but clips spanned ≤33 frames and were sampled uniformly. | `data/motion_clips.py` — strides to 24 (span 97), motion-weighted sampling, `min_seq_len=60` drops the ~50-frame sequences. |
| 3 | **Selection blind to temporal quality.** `best.pth` was picked on shuffled per-frame abs_rel with state reset every batch — a mode where the temporal module is invisible. | `engine/train_video.py::make_eval_fn` — selection on **streaming** `abs_rel + 4·TAE`. |

## 2. The temporal loss: TGM (VDA's loss without VDA's architecture)

Constraint honoured: Video Depth Anything's temporal *architecture*
(clip-level attention through the DPT head over 32-frame windows) is too
heavy to adopt (mentor-confirmed). But VDA's central *insight* is its loss,
and it is ~15 lines:

```
L_tgm = mean_valid | (log d_t − log d_{t−1}) − (log dt_t − log dt_{t−1}) |
```

- A prediction that **tracks GT's motion scores 0** — moving objects are never
  punished for legitimately changing depth (the failure mode of optical-flow
  warping losses, and why we don't use them for training).
- Only **flicker** — change the GT does not have — is penalised.
- **Flow-free**: needs no RAFT, no extra models, zero inference cost.
- It is exactly the TAE metric turned differentiable: we optimise what we
  report.

`order: 2` optionally adds acceleration matching (suppresses residual jitter).
Total loss: `SigLog(per frame) + w·TGM(clip)` — single-frame accuracy stays
pinned while flicker is squeezed.

## 3. The temporal architecture: kept deliberately small

`models/temporal_head.py` — residual ConvGRU on one decoder level,
zero-init output (**exact no-op at init**: the model can never start worse
than the per-frame baseline; unit-tested), O(1) streaming state (arbitrary
video length at inference), `downsample: 0.25` (a quarter-side grid can fix
mid-frequency flicker; the old 0.10 could only shift global scale). A few
hundred K parameters; spatial weights frozen (stage-2a) for clean attribution.

`models/video_model.py` composes it with the trained Phase-1 checkpoint by
re-running `backbone → decoder → [stabilizer] → refinement` — no edits to
`instancedepth`.

## 4. Occlusion handling (Phase 3), two layers

### 4a. Bounded pair-attention Φo (`models/occlusion.py`) — trained, drop-in

Three defects of the paper's Eq. 8 MLP head, each fixed:

| Paper's Φo | Defect | `BoundedPairAttentionHead` |
|---|---|---|
| 1×1 convs only | no spatial context — can't see *where* the overlap is | 3×3 conv encoder |
| pair coupling = channel concat | no explicit reasoning over the partner | cross-attention: each member's ROI tokens attend to the *other* member's |
| E ∈ (0,1) ⇒ ratio 2E ∈ (0,2) | can halve/double depth; confident mistakes paint hard ratio steps = the **boundary rings** (AUDIT §3.2) | `ratio = 1 + max_corr·tanh(z)`, default ±15 % — rings bounded by construction, graceful degradation |

Interface-identical to `OcclusionRelationHead` (`e_obj = ratio/2`, `d_hat =
2·e_obj·d_obj`), so compositor, losses, matcher and trainer run unchanged —
train via `videodepth.engine.train_phase3_video` with `freeze_phase1: true`.

### 4b. Temporal amodal completion (`models/track_memory.py`) — training-free

The paper reasons about occlusion within one frame; but in video, the person
occluded *now* was fully visible *moments ago*. `TrackDepthMemory` keeps a
per-track visibility-modulated EMA with a constant-velocity term:

- fully visible → observation folds in at full momentum (output = low-passed,
  flicker-free layer);
- occluded (visibility from mask-area collapse) → update damps to zero, the
  track *coasts on its own velocity* (a person walking away keeps receding
  while hidden);
- `StreamingInstanceStabilizer` (`models/phase3_video.py`) chains the
  Hungarian `MaskTracker` (persistent IDs) with this memory — one
  `update(masks, layers)` call per frame.

Zero training, deployable on **any existing checkpoint** today — this is the
component that makes results *visible* immediately (stable colours, stable
per-person depth through occlusions).

## 4½. The AbsRel lever: full-DAv2 Phase 1 (`models/dav2_dpt.py`)

The audit's remaining headline gap is REL: paper 0.045 on GID vs 0.078 here
(note: different datasets, soft comparison — and RMS 0.38 here already *beats*
the paper's 0.397, so the error is concentrated at near-range pixels, which
REL weights most). The evidence chain for the fix:

| Model | REL |
|---|---|
| this repo's from-scratch Depth-Range decoder | 0.078 |
| swap ONLY the encoder to DAv2's | 0.0778 (−0.5 %) |
| plain DAv2 — pretrained encoder **+ DPT head** (paper Table 2, GID) | **0.053** |
| paper's full method (GID) | 0.045 |

The pretrained **decoder** carries the metric-depth prior; the encoder swap
alone proves the encoder was never the bottleneck. `DAV2MetricModel`
reproduces DAv2's DPT head **key-for-key compatible** with the official
checkpoints (unit-tested against the vitl key/shape schema), loads the full
pretrained model (`pretrained.*` → existing DINOv2 wrapper, `depth_head.*` →
fail-loud head loader), adds the metric `sigmoid·max_depth` output, and
fine-tunes with SigLog + gradient matching (encoder at 0.1× LR).

It exposes the same `backbone/decoder/refinement` contract, so the temporal
stage wraps it unchanged — **DAv2 spatial + TGM stabilizer compose for
free** (unit-tested). Recommended init: the *metric hypersim* checkpoint
(indoor range, sigmoid head matches 1:1).

```bash
python -m videodepth.engine.train_dav2 \
    --config videodepth/configs/dav2_full.yaml \
    --override dav2_checkpoint=/path/to/depth_anything_v2_metric_hypersim_vitl.pth \
               backbone.checkpoint_path=/path/to/depth_anything_v2_metric_hypersim_vitl.pth
python -m videodepth.engine.train_dav2 --evaluate \
    --config videodepth/configs/dav2_full.yaml --checkpoint runs/dav2_full/best.pth
```

## 4c. Video-Mask2Former identity, without the video training (`models/query_tracker.py`)

**Concept** (Cheng et al., *Mask2Former for Video Instance Segmentation*): in
the video model a single query is the same instance in every frame — identity
is a property of the query. The full video model buys that with clip-level
attention + video training (heavy, and it would obsolete the trained
`phase2_run` checkpoint). **MinVIS** (NeurIPS 2022) proved the concept works
with NO video training: the per-frame model's final decoder query embeddings
are already temporally consistent, so identity = Hungarian matching of query
embeddings across frames.

`QueryInstanceTracker` implements that, using the `query_embeddings` the
Phase-2 contract already exposes (cosine, weighted with mask IoU as a spatial
tie-break; EMA embedding memory with `max_age=30`). Wired into
`infer_video.py` / `make_sequence_videos.py` as the `--track-instances`
tracker (embeds auto-passed by `build_scene_predictor`; IoU-only fallback
when absent). What it adds over the old mask-IoU tracker:

- **crossings**: identity follows the *appearance embedding*, not the
  position, so two people passing each other keep their colours;
- **re-identification**: an occluded person keeps their embedding while
  unseen and re-matches on return — even many frames later, from a new
  position, which IoU tracking structurally cannot do.

Zero retraining, works on the existing checkpoints today. The trained
clip-level Video Mask2Former remains the (expensive) upgrade path if
embedding matching ever proves insufficient.

## 5. Run book (training server)

```bash
# 1) Temporal stage (Phase-1 checkpoint required):
python -m videodepth.engine.train_video \
    --config videodepth/configs/video_temporal.yaml \
    --override init_checkpoint=runs/hdi_enhanced/best.pth

# 2) Streaming eval — the paper table: per-frame baseline vs +temporal
python -m videodepth.engine.evaluate_video \
    --config videodepth/configs/video_temporal.yaml \
    --checkpoint runs/video_temporal/best.pth
#   report: abs_rel (must not regress), TAE, flicker_ratio (pred Δ / gt Δ)

# 3) Phase 3 with the bounded pair-attention head (Phase 1 frozen):
python -m videodepth.engine.train_phase3_video \
    --config instancedepth/configs/phase3_current.yaml \
    --override phase1_checkpoint=runs/hdi_enhanced/best.pth \
               phase2_checkpoint=runs/phase2_run/best.pth \
    --run-name phase3_video
python -m instancedepth.engine.evaluate_phase3 \
    --config instancedepth/configs/phase3_current.yaml \
    --checkpoint runs/phase3_video/best.pth
#   expect: overall_base ≈ 0.078 (frozen), occ_refined ≤ occ_base
```

### Ablation matrix for the paper

| Row | Config | Isolates |
|---|---|---|
| per-frame baseline | Phase-1 checkpoint via `evaluate_video` | the flicker floor |
| + temporal head, no TGM | `loss.temporal_weight=0` | architecture alone (expect ≈ null, reproducing AUDIT §1.1) |
| + TGM | default | **the loss is the contribution** |
| + motion weighting off | `clips.motion_weighting=false` | sampling contribution |
| Φo (paper) vs bounded pair-attention | `train_phase3` vs `train_phase3_video` | head contribution |
| ± track memory | viz/eval with & without `StreamingInstanceStabilizer` | training-free video gain |

## 6. Honest expectations

- TGM can only remove flicker that exists: on near-static clips the headline
  TAE gain will be modest; the *motion-weighted* subset and the
  `flicker_ratio` metric are where the improvement will show. Report both.
- `TrackDepthMemory` improves *instance-layer* stability and occluded-instance
  depth immediately and visibly; it does not change dense-map metrics.
- Dense amodal supervision remains impossible with single-sensor GT
  (AUDIT §3.3) — the temporal memory is precisely the workaround: it supplies
  the occluded instance's depth from *time* instead of from a second sensor.
