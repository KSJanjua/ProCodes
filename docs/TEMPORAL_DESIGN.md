# Temporal Consistency for Phase 1 — FlashDepth Integration Design

**Goal:** the first research improvement over the reproduced InstanceDepth
baseline: integrate the *temporal-consistency mechanism* of FlashDepth (Chou et
al., arXiv 2504.07093) into Phase 1, with the minimum necessary modifications
and no change to the InstanceDepth methodology elsewhere. FlashDepth's second
contribution (the ViT-S/ViT-L 2K hybrid) is explicitly **out of scope** —
already rejected for this 720p dataset in `docs/FLASHDEPTH_ANALYSIS.md`.

**Sources cross-referenced:** FlashDepth paper §3.2/§3.4/§3.5 + official code
(`flashdepth/model.py`, `flashdepth/mamba.py`, `configs/flashdepth-l/config.yaml`,
`train.py`); the InstanceDepth paper; this repo's Phase 1–3 implementations.
Provenance tags: `[Paper Specified]` (FlashDepth unless noted) ·
`[Strongly Inferred]` · `[Reasonable Assumption]`.

**Design document only — no code.**

---

## 0. Baseline facts this design builds on

- **Phase 1 is strictly per-frame.** `GIDInstanceDepthDataset` serves shuffled
  single frames; `HolisticDepthModel.forward(image)` is stateless;
  `HDIInferencer.predict()` is stateless. Nothing anywhere carries information
  across frames → per-frame estimation noise appears as temporal flicker in the
  sequence videos.
- **Phase 1's depth is built iteratively**: decoder features F_0 (1/8), F_1
  (1/4), F_2 (1/2) → `D_0 = InitialDepthHead(F_0)`; per level *i*, heads on F_i
  produce C_i/S_i → `E_i` → `D_{i+1} = up(D_i) + E_i`; `depth_final = up(D_3)`.
  F_2 feeds the **last** refinement level, whose correction E_2 has full
  authority over the final depth (R_2 ∈ (0, r_d) ⇒ E_2 spans several metres),
  and F_2 is also what produces `seg_final` and is exposed as `feat_final` to
  Phase 3.
- Phases 2 and 3 consume Phase 1 only through `HolisticDepthOutput`
  (`depth_final`, `feat_final`/`feat_levels`); Phase 2 is an independent
  branch.

## 1. What FlashDepth's temporal mechanism actually is (paper ∩ code)

Eq. 1–2 (paper §3.2), verbatim mechanism:

```
f, H_t   = Mamba(Flatten(Down(F)), H_{t−1})     # recur over downsampled tokens
F_align  = F + Up(UnFlatten(f))                 # residual add, then the final head
```

Code-verified specifics:

| Fact | Evidence |
|---|---|
| Placement: after the **last** DPT fusion block, before the final conv head | `mamba_in_dpt_layer: [3]` (config); paper §3.2 "after the DPT decoder, only before the final convolution head, to preserve the integrity of the pretrained features" |
| Downsample before recurrence | `downsample_mamba: [0.1]` — features shrunk to ~10 % per side before the RNN; paper: "dense pixel-level features are unnecessary for aligning depths" |
| Core: 4 blocks of (LayerNorm → Mamba → residual, LayerNorm → MLP(4×, GELU) → residual) | `num_mamba_layers: 4`, `mamba.py` block structure |
| Residual application mode | `mamba_type: "add"` (Eq. 2); a `modulation` (scale/shift) variant exists in code but is not the default |
| **Zero-initialized output projection** → the module contributes exactly 0 at step 0; the model *is* the pretrained per-frame model at initialization | `last_block ... weight.data.zero_(); bias.data.zero_()` (`mamba.py`); paper §3.2 |
| **Hidden state = zeros at every sequence start** — there is *no learned initial state* | `start_new_sequence()` re-creates fresh `InferenceParams`; state advances via `seqlen_offset` per frame (`forward_single_frame`) |
| Training: short ordered clips, full BPTT within the clip, state reset per clip | `video_length: 5` (config); paper §3.4 "backpropagation with input sequences containing a small number of frames" |
| **Stride augmentation** → generalization to ~1000-frame videos from 5-frame clips | paper §3.4 "augment the dataset by setting longer strides between frames" |
| Two-tier learning rates: temporal fast, pretrained slow | config: `mamba: 1e-4`, `vit: 5e-6`, `dpt/head: 5e-5`, warmup 1000 |
| **No temporal or optical-flow loss** — per-frame supervision only; consistency emerges from the recurrence | paper §3.4 "We do not use any additional supervision, such as optical flow or temporal losses" |
| ~1 % of total parameters; negligible latency | paper §3.2 |
| Recurrent core is swappable (Mamba/xLSTM/TransformerRNN/Hydra flags); "vanilla Mamba … is sufficient" | config `use_xlstm/use_hydra/use_transformer_rnn: false`; paper §3.5 |

## 2. The mentor's suggestion, evaluated

**Suggestion (as relayed):** Stage 1 train without the recurrent module; Stage 2
add it and continue training, so the temporal module learns from a well-trained
spatial backbone. Motivation: avoid problems from zero-initialized recurrent
states early in training.

**Verdict: sound — and it is precisely what FlashDepth itself does**, one level
removed: FlashDepth's "stage 1" is DAv2's own (spatial, per-frame) training —
they never train the spatial model jointly with an untrained temporal module
from scratch. They take the *finished* per-frame model, attach the zero-init
temporal module, and fine-tune with a fast temporal LR over a slow backbone
`[Paper Specified]`. For this project the mapping is even cleaner: **Stage 1 is
already complete** — the trained Phase-1 checkpoints (`runs/hdi_enhanced`,
`runs/hdi_dav2`) *are* the stage-1 product. The improvement reduces to **one new
fine-tuning stage**.

**One clarification the design must get right** (the mentor's memory conflated
two different "zero initializations"):

1. **Zero hidden state at sequence start** — *not* a problem to avoid. It is
   FlashDepth's actual behavior at every sequence boundary, in training and
   inference alike (no learned initial state exists in the code). It is safe
   *because of* point 2.
2. **Zero-initialized output projection of the temporal module** — this is the
   real stabilizer `[Paper Specified]`. It guarantees (a) at training step 0
   the model is exactly the pretrained per-frame model (no destructive early
   gradients through a random module), and (b) architecture-wise, an
   empty-memory first frame can never be *worse* than the per-frame model,
   because the module's contribution is a learned residual that starts at 0.

So the principled strategy is: pretrained spatial stage (done) → attach
zero-init residual module → fine-tune on clips; zero hidden state at every
sequence start is kept, in training and inference. The "different strategy at
inference" the mentor half-remembered is, with high probability, the
**streaming statefulness** difference: training uses short clips with a state
reset per clip, while inference carries the hidden state across the entire
sequence (hundreds of frames) and resets **only at sequence boundaries** —
§5 below.

## 3. Architectural integration

### 3.1 Insertion point `[Strongly Inferred]` (structural mapping of a [Paper Specified] choice)

**One temporal-alignment module on F_2, between the Depth Range Feature
Decoder and the Iterative Bin Refinement.** This is the exact structural
analogue of FlashDepth's placement: F_2 is the *last* decoder feature before
the heads that finalize depth (E_2 fully adjusts `depth_final`), just as
FlashDepth aligns the last DPT feature before its final head. Placing the
module *after* the pretrained-ish decoder and *before* the heads preserves "the
integrity of the pretrained features" — FlashDepth's stated reason.

Two consequences come free:
- `seg_final` (S_2 is computed from F_2) becomes temporally aligned too.
- `feat_final` — Phase 3's F_obj source — becomes the *aligned* F_2, so Phase 3
  inherits a temporally steadier evidence base with zero interface change.

Config exposes `temporal.levels: [2]` (default) with `[0]`, `[0,1,2]` as
ablations `[Reasonable Assumption]` — mirroring `mamba_in_dpt_layer`'s list
form. Rationale for the ablation: in this architecture the metric *scale* is
seeded by D_0 = f(F_0), so a coarse-level module is a plausible alternative
alignment point; FlashDepth's own evidence (one insertion at the last layer
suffices) makes `[2]` the default.

### 3.2 Module specification (mirrors Eq. 1–2)

```
F_2 (B,256,H/2,W/2)
  → Down(bilinear, factor 0.1 per side)        [Paper Specified value]
  → flatten to tokens, project to d_model=256
  → N=4 blocks: (LN → recurrent core → +res, LN → MLP(4×,GELU) → +res)   [Paper Specified]
  → output projection  — ZERO-INITIALIZED       [Paper Specified]
  → unflatten, Up(bilinear) to F_2's resolution
  → F_2_aligned = F_2 + correction              [Paper Specified: "add" mode]
```

State: carried across frames within a sequence; reset to zeros at sequence
boundaries (`start_new_sequence()` analogue). No learned initial state
`[Paper Specified]`.

### 3.3 Recurrent core `[Reasonable Assumption]` — the one deliberate deviation

FlashDepth defaults to Mamba, but `mamba-ssm` requires compiled CUDA kernels
(`causal-conv1d`, Triton) — fragile on the proxy-restricted Backend.AI server.
FlashDepth's own code demonstrates the core is swappable (xLSTM /
TransformerRNN / Hydra flags), and the paper needs only "a lightweight
recurrent network". **Default core: a ConvGRU** (convolutional gates over the
downsampled feature grid, hidden state (B, C_h, h', w'), per-frame update —
the standard recurrent update operator in vision, e.g. RAFT). Rationale:
dependency-free, per-*location* temporal memory is arguably a more natural
aligner than Mamba's rasterized space-time scan, and streaming/O(1)-state
properties are identical. `temporal.core: convgru | mamba` keeps a Mamba
backend available for a fidelity ablation if `mamba-ssm` installs. This is
flagged as the design's only substantive deviation from FlashDepth, motivated
by environment constraints, with the faithful option preserved.

### 3.4 Interface impact — none breaking

- `HolisticDepthOutput`: **no shape or field change**; `feat_final` semantics
  remain "the finest decoder feature that produced the depth" (now
  post-alignment when temporal is enabled).
- `HolisticDepthModel`: gains an internal, module-held temporal state and a
  `reset_temporal_state()`; `forward(image)` signature unchanged. With
  `temporal.enabled: false` (default in every existing profile) the module is
  absent and behavior is bit-identical to the baseline — the faithful
  reproduction is untouched.
- Phase 2: no change (independent branch). Phase 3: no structural change
  (consumes the same contract); interaction nuances in §5.3.

## 4. Training strategy

### 4.1 Stages

- **Stage 1 — spatial (already complete).** The existing Phase-1 training; use
  `runs/hdi_enhanced/best.pth` (or `hdi_dav2`) as the spatial base. No
  retraining.
- **Stage 2a — temporal module, frozen spatial `[Reasonable Assumption]`
  (primary first experiment).** Attach the zero-init module; freeze backbone,
  decoder, and heads; train only the temporal parameters. Cleanest attribution
  (any metric change is the module's), zero risk of degrading per-frame
  accuracy, and cheap: the backbone/decoder can run without gradient tracking,
  so clip training fits a single GPU comfortably.
- **Stage 2b — tiered-LR joint fine-tune `[Paper Specified]` (follow-up,
  FlashDepth-faithful).** Unfreeze with FlashDepth's LR tiers adapted to this
  repo's convention: temporal ~1e-4, backbone ~1e-6, decoder/heads ~5e-6–5e-5,
  short warmup. Run only if 2a shows the module helps; 2b lets the spatial
  features co-adapt (what FlashDepth actually does).

### 4.2 Clip construction `[Paper Specified]` mechanics / `[Reasonable Assumption]` values

- New **clip sampler** over the existing annotations: an index of
  (sequence, start_frame, stride) triples; each item yields T ordered frames +
  depths. T = 4–8 (FlashDepth: 5). Stride sampled from {1, 2, 4, 8} per clip —
  the stride augmentation that buys long-video generalization from short clips
  (paper §3.4; exact range unspecified → assumption). Augmentation (hflip,
  jitter) must be **consistent across the clip's frames**.
- Full backpropagation through the clip (truncated BPTT with truncation = clip
  length); temporal state reset at every clip start — identical to
  `start_new_sequence()` per training sequence in the official code.
- Batch = independent clips from different sequences; per-clip state is
  per-batch-element.

### 4.3 Loss — unchanged `[Paper Specified]` (that nothing new is needed)

Apply the **existing Phase-1 loss stack** (metric SigLog + deep supervision +
ordinal bin BCE, optional disparity aux) to *every frame of the clip*,
averaged. Do **not** adopt FlashDepth's plain L1 (their synthetic-metric
convenience; SigLog is this project's established metric-faithful choice), and
do **not** add a temporal/optical-flow loss — FlashDepth explicitly uses none;
consistency emerges from the recurrence + per-frame supervision. This keeps
the improvement minimal and the loss ablation-clean.

### 4.4 Cost estimate

Stage 2a: backbone/decoder forward under no-grad, gradients only through the
tiny module + (frozen) heads' activations → clip-of-5 training approximately
as memory-hungry as batch-5 inference; very feasible. Stage 2b: gradients
through 5 ViT-L forwards → batch 1–2 with gradient accumulation (FlashDepth
used 8×A100 for half a day at this stage; expect the single-GPU equivalent to
be the slow path). Total new parameters ≈ a few M (~1 % of the model), matching
the paper's footprint claim.

## 5. Inference strategy

### 5.1 Streaming (the intended mode) `[Paper Specified]`

Process each sequence's frames **in order**; carry the temporal state across
all frames; call `reset_temporal_state()` exactly at sequence boundaries.
`HDIInferencer` gains this reset + an explicit streaming path; the video tools
(`make_sequence_videos`, `infer_video`) already iterate frames in order and the
sequence-eval loaders already serve ordered frames — they only need the
per-sequence reset call and **batch_size 1** during temporal evaluation
(recurrent state is per-stream; mixing sequence positions in a batch is
incorrect).

### 5.2 Single-image semantics — calibrated by construction

A lone `predict()` resets state first, i.e. the frame is treated as a
length-1 sequence. This is exactly the distribution the module saw in training
(every clip's first frame has zero incoming state), so single-image behavior is
well-defined and *learned*, not an out-of-distribution hack. Note it is not
bit-identical to `temporal.enabled: false` after training (the output
projection is no longer zero) — both modes should be reported.

### 5.3 Long sequences and Phase 3

- **Length generalization:** trained on 5-frame clips, FlashDepth validates
  ~1000-frame videos with no drift handling `[Paper Specified evidence]`; this
  dataset's ~185-frame sequences are well inside that envelope. A
  `temporal.reset_every: 0` knob (periodic soft reset, off by default) is kept
  as a cheap safety valve `[Reasonable Assumption]`.
- **Phase 3 inference:** `Phase3Model` calls Phase 1 per frame; sequence tools
  reset Phase-1 state per sequence — Phase 3 then consumes temporally steadier
  `depth_final`/`feat_final` with zero interface change.
- **Phase 3 training (open question, documented not hidden):** Phase-3
  training samples random (occlusion-biased) frames — random access conflicts
  with recurrence. The minimal-change position: train Phase 3 against Phase 1
  in *reset-per-frame* mode (the calibrated length-1 behavior of §5.2) and run
  streaming at inference; evaluate whether the mild train/inference feature
  shift matters. The alternatives (clip-based Phase-3 training, or precomputing
  streamed Phase-1 features per sequence) are heavier and only justified if the
  shift measurably hurts.

## 6. Evaluation plan

1. **Dense per-frame metrics (must not regress):** the existing REL/RMS/σ
   protocol, temporal-on (streaming) vs. the frozen baseline checkpoint. The
   zero-init + frozen-spatial stage makes regression unlikely; verify anyway.
2. **Temporal consistency (the point of the change):** a new diagnostic —
   first-order **temporal alignment error**: mean |Δpred_t − Δgt_t| over valid
   pixels, where Δx_t = x_t − x_{t−1} on consecutive frames. If prediction
   tracks GT perfectly, the score is 0 regardless of camera/object motion; no
   optical flow needed because ZED GT exists per frame `[Reasonable
   Assumption]` (flow-warped TAE à la NVDS/VDA is the literature alternative;
   heavier, deferred). Report per-frame-mode vs. streaming-mode vs. baseline.
3. **Qualitative:** the existing sequence-video tool (RGB | GT | pred |
   instances) is the flicker eyeball test — the artifact that motivated this
   work.
4. **Downstream:** Phase-3 occlusion-slice metrics with a streaming Phase 1
   (no Phase-3 retraining) — does steadier base depth already help?

## 7. Expected benefits and risks

**Benefits:** visibly reduced frame-to-frame flicker at ~1 % parameter cost and
negligible latency; steadier `feat_final` for Phase 3; a clean, publishable
"baseline + temporal" ablation story; no impact on the faithful reproduction
(flag-gated).

**Risks & mitigations:**

| Risk | Mitigation |
|---|---|
| `mamba-ssm` uninstallable on the server | ConvGRU default core (§3.3); Mamba optional |
| Per-frame accuracy regression | zero-init start + frozen-spatial Stage 2a; must-not-regress gate (§6.1) |
| State drift on long sequences | FlashDepth's 1000-frame evidence; optional `reset_every` knob |
| Clip training memory | Stage 2a runs backbone no-grad; 2b uses batch 1–2 + grad accumulation |
| Batched eval with recurrent state | temporal eval at batch 1; per-sequence reset enforced in tools |
| Phase-3 train/inference feature shift | documented open question (§5.3); measured, not assumed |
| Augmentation breaking temporal coherence | clip-consistent hflip/jitter in the sampler |

## 8. Implementation roadmap (when approved — no code yet)

1. `TemporalAligner` module (ConvGRU core, zero-init output, down/up residual
   wrap) + `temporal.*` config block + unit tests (zero-init identity at step
   0; state carry changes output at t>0; reset restores length-1 behavior).
2. Clip sampler over existing annotations (ordered, stride-augmented,
   clip-consistent augmentation) + Stage-2a training entry
   (`hdi_temporal.yaml`: init from a trained Phase-1 checkpoint, frozen
   spatial, temporal LR 1e-4).
3. Streaming inference: `reset_temporal_state()` in the model,
   streaming path + per-sequence resets in `HDIInferencer` and the video
   tools; temporal-alignment-error diagnostic in the Phase-1 evaluator.
4. Train Stage 2a on `hdi_enhanced`'s checkpoint; evaluate §6; decide on
   Stage 2b (tiered unfreeze) and the `[0,1,2]`-levels / Mamba-core ablations.
5. Optional: Phase-3 streaming evaluation (§6.4).

## 9. Deviations from the InstanceDepth framework — declared

| Deviation | Justification |
|---|---|
| Temporal module itself (InstanceDepth is per-frame) | The research improvement under study; **opt-in flag, default off** — the faithful baseline remains intact and comparable |
| ConvGRU instead of Mamba as default core | Environment constraint; FlashDepth's own code proves the core is interchangeable; Mamba kept as optional backend |
| Clip-based training path alongside the frame-based one | Required by recurrence; additive, does not alter the baseline path |

Everything else — backbone, decoder, Eq. 1–4 refinement, losses, Phase 2,
Phase 3, all output contracts — is unchanged.
