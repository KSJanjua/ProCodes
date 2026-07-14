# FlashDepth for InstanceDepth — Analysis & Recommendation

**Question:** Should FlashDepth (Chou et al., arXiv 2504.07093, Netflix Eyeline /
Cornell) be incorporated into this RGB-D InstanceDepth reproduction, as a cheaper
alternative to Video Depth Anything for temporal consistency in Phase 1?

**Sources:** the paper (`2504.07093v2.pdf`) and the official implementation
(`github.com/Eyeline-Labs/FlashDepth`: `flashdepth/model.py`, `train.py`,
`configs/flashdepth-l/config.yaml`). Every claim below is cross-referenced
against the code, not the paper alone.

**One-line answer:** Adopt **one idea** — FlashDepth's lightweight, zero-init,
residual *streaming temporal-alignment module* — as an **opt-in** temporal
extension to Phase 1. Reject the hybrid 2K stream and reject replacing Phase 1's
architecture. Details, evidence, and trade-offs below.

---

## 1. What FlashDepth actually is (paper ∩ code)

FlashDepth = **Depth Anything V2** (DINOv2 ViT + DPT decoder) + two additions:

1. **A streaming temporal module** placed inside/after the DPT decoder, before
   the final head. Verified in `configs/flashdepth-l/config.yaml`:
   `use_mamba: true`, `mamba_type: "add"` (residual), `num_mamba_layers: 4`,
   `mamba_d_state: 256`, `downsample_mamba: [0.1]` (features downsampled to 10%
   before the RNN), `mamba_in_dpt_layer: [3]`. Paper Eq. 1–2:
   `f, H_t = Mamba(Flatten(Down(F)), H_{t-1})`; `F_align = F + Up(UnFlatten(f))`.
   Zero-initialized output → the first iteration is exactly the base DAv2 depth
   map (paper §3.2; ~1 % of total parameters).

2. **A hybrid high-resolution model** (only in "FlashDepth (Full)"): a ViT-S
   stream processes native 2K, cross-attending (Q=ViT-S, KV=ViT-L) to a ViT-L
   stream running at 518 px, to recover sharp boundaries at real-time speed
   (paper §3.3, `hybrid_fusion.py`). Disabled in the -L config
   (`use_hybrid: false`).

**Training** (`train.py`, config): two stages. Stage 1 trains the temporal
module on **5-frame clips** (`video_length: 5`) at 518×518 with **stride
augmentation** → generalizes to ~1000-frame videos; LR is two-tier — temporal
`mamba: 1e-4`, backbone `vit: 5e-6`, `dpt/head: 5e-5`. Loss is **L1**
(`loss_type: "l1"`) on GT scaled ×100 for numerical range-matching, with an
optional scale-shift-invariant alternative and an optional frames-500+ temporal
regularizer. Stage 2 trains the hybrid on ~16 k 2K frames.

**Depth type — the decisive detail.** Despite the ×100 metric-GT scaling in
training, FlashDepth is fundamentally a **relative / affine-invariant** model:
the paper's evaluation (§4, "Metrics") aligns *the entire predicted sequence* to
ground truth with a **least-squares global scale and shift** before computing
AbsRel/δ1. That alignment step is the signature of relative depth — the raw
output is not reliably metric.

**Temporal module is swappable.** The config exposes `use_hydra`,
`use_transformer_rnn`, `use_xlstm` (all `false` by default) — Mamba is one of
four interchangeable recurrent backends (`flashdepth/{mamba,hydra,xlstm_block,
rnn_transformer}.py`). This matters for dependency management (§6).

---

## 2. Comparison with the current Phase 1

| Dimension | Current Phase 1 (this repo) | FlashDepth | Verdict for this project |
|---|---|---|---|
| Base architecture | DINOv2 ViT-L + **Depth Range Feature Decoder** + iterative bin refinement (Eq. 1–4) | DINOv2 ViT + **DPT decoder** | Different decoders; FlashDepth's temporal module targets DPT features, so its *placement* must be adapted, not dropped in |
| Depth type | **Metric** (SigLog on ZED metric GT) | **Relative** (scale+shift-aligned at eval) | **Core incompatibility** — see §4 |
| Temporal consistency | **None** (per-frame; flickers) | **Yes** (streaming RNN alignment) | FlashDepth's genuine advantage |
| Per-frame accuracy | Competitive metric depth | Competitive relative (2nd/3rd in Tab. 2; VDA higher) | ≈ parity; not a reason to switch |
| Boundary sharpness | Decoder/resolution-limited | **Sharp — but only via the 2K hybrid** | N/A at 720p (§3) |
| Compute / memory | ViT-L single pass | -L: 30 FPS@518, 6 FPS@2K; temporal adds ~1 % | Temporal path is cheap; hybrid is 2-model |
| Training cost | 55 k iters, single-frame | Temporal: 5-frame clips, ½ day 8×A100 | Temporal module is **cheap** to add |
| Long sequences | Independent per frame | Designed for streaming ≤1000 frames | FlashDepth's strength |
| Compat. with Phase 2/3 | Native | Only the temporal *idea* is compatible | §5 |

---

## 3. Dataset compatibility (720×1280, ~55 k frames, 297 sequences, ZED metric GT)

- **Resolution kills the hybrid model's value.** FlashDepth's headline
  contribution (sharp 2K boundaries at 24 FPS) exists *because* DAv2-L is slow at
  2K. Your data is 720×1280 — DINOv2/DAv2-L already runs comfortably and
  accurately at this resolution (the paper itself shows 518 px ≈ 2K accuracy;
  boundaries are only ~1 % of pixels). **The entire ViT-S/ViT-L cross-attention
  hybrid is unnecessary and should not be adopted.**
- **Metric GT is the mismatch.** ZED gives absolute metric depth, and this repo
  correctly trains Phase 1 with scale-*variant* SigLog to keep that metric
  signal. FlashDepth is relative. Adopting FlashDepth *wholesale* would discard
  the metric property Phase 2's instance depth layers and Phase 3's `L_dist`
  depend on. (The temporal module *alone* does not force relative output — §4.)
- **Sequence structure is a good fit for the temporal idea.** 297 sequences of
  ~185 frames each, already tracked, is exactly the streaming regime FlashDepth's
  RNN targets, and the 5-frame-clip + stride recipe is cheap on this data volume.

---

## 4. Integration with InstanceDepth — what fits, what doesn't

**Directly applicable (adopt the concept, adapt the placement):**
- The **residual, zero-init, downsample-then-recur temporal module** (Eq. 1–2).
  It is architecture-agnostic: it takes a decoder feature map, aligns it across
  frames via a small recurrent net, and adds the correction back. In this repo it
  would attach to the **finest Depth Range Decoder feature** (`feat_final` / F_2)
  *before* the confidence/bin/depth heads — the analogue of "after the DPT
  decoder, before the final head." Zero-init means Phase 1 is unchanged at step 0
  (same safety property as this project's Phase-3 Φo and DAv2's own temporal
  init).
- The **training recipe**: short clips (4–8 frames) + stride augmentation, a
  fast temporal LR over a frozen/slow backbone. Cheap, low-risk, proven to
  generalize to long videos.

**Incompatible / reject:**
- **The hybrid 2K stream** — irrelevant at 720p (§3); doubles model residency.
- **Replacing Phase 1 with DAv2+DPT** — would abandon the paper-faithful Depth
  Range Decoder + Eq. 1–4 bin refinement that *is* the reproduction's core.
- **FlashDepth's relative output / L1-on-×100-GT recipe** — the temporal module
  must be trained with this repo's existing **metric SigLog** loss instead, so
  the depth stays metric. The temporal module aligns *frames to each other*
  (consistency), which is orthogonal to metric-vs-relative — so keeping SigLog is
  both possible and necessary.
- **Reusing FlashDepth's pretrained temporal weights** — they were trained on
  relative synthetic data; train the module from scratch on your metric data
  (it's tiny and fast).

**Does it replace or improve the holistic stage?** *Improve, optionally* — it
adds temporal consistency the current per-frame Phase 1 lacks. It does not
replace the stage.

**Additional outputs required?** No new *shapes*. `HolisticDepthOutput`,
`Phase2Output`, `RefinedDepthOutput` stay as-is. The only interface change is that
Phase 1 becomes **stateful**: inference must call a `reset()`/`start_sequence()`
at each video boundary and carry the hidden state frame-to-frame
(`HDIInferencer` gains streaming methods; `Phase3Inferencer`/video tools reset
per sequence). Training gains a clip-based dataloader path.

**Phase 2 / Phase 3 interface changes?** None structural. Phase 2 is an
independent Swin-L branch — untouched. Phase 3 consumes Phase-1 `feat_final` /
`depth_final`; a temporally-consistent Phase 1 simply makes that base steadier
across frames (which should *help* occlusion reasoning frame-to-frame). This is
exactly the "deferred temporal module" the Phase-3 design doc already anticipated
(`docs/PHASE3_DESIGN.md §9`), now with a concrete, cheap implementation.

---

## 5. Critical evaluation (not "newer = better")

**Where FlashDepth's idea genuinely helps here:**
- Temporal flicker reduction — the current Phase 1 is per-frame and *will*
  flicker; the paper shows temporal alignment gives up to ~20 % δ1 improvement
  over per-frame DAv2 and is the single biggest lever for video quality.
- Cost — directly answers your reason for rejecting Video Depth Anything: VDA
  jointly optimizes 32-frame batches (≈40 GB VRAM at 518 px per the paper);
  FlashDepth's module is ~1 % params, trains on 5-frame clips, streams one frame
  at a time. Strictly cheaper in memory, training, and inference.

**Where it does NOT help / could mislead:**
- Boundary sharpness (its marquee result) is **resolution-driven** and vanishes
  at 720p — do not expect it.
- Per-frame *accuracy* is not FlashDepth's advantage (VDA beats it; your metric
  SigLog Phase 1 is already fine). The temporal module trades a little per-frame
  accuracy for consistency.
- Residual flickering remains (the paper's own stated limitation, §5) — it
  reduces, not eliminates, inconsistency.

**Costs / risks of adopting even just the module:**
- **`mamba-ssm` dependency** (CUDA kernels + `causal-conv1d` + Triton) is
  install-fragile on a proxy-restricted Backend.AI box. *Mitigation:* the config's
  `use_transformer_rnn` / `use_xlstm` show the recurrent backend is swappable — a
  dependency-free GRU/transformer-RNN aligner captures ~the same idea without
  mamba-ssm. Recommend starting there.
- Statefulness complicates inference and eval (per-sequence reset, ordered
  frames, careful batching). Modest but real engineering.
- Measuring the benefit needs a **temporal metric** (e.g. temporal-alignment
  error / optical-flow-warped consistency) — the current dense metrics won't show
  it, so success is otherwise invisible.

---

## 6. Recommendation

**1. Should FlashDepth be incorporated?** **Partially, and opt-in.** Adopt the
*temporal-alignment module concept*; reject the hybrid 2K model and any
wholesale architecture swap. This is a *justified optional enhancement*, not a
faithfulness requirement — the InstanceDepth method is per-frame, so keep the
faithful baseline untouched and gate the temporal module behind a config flag
(the same faithful/enhanced discipline used elsewhere in this repo).

**2. How to integrate faithfully.** Add a lightweight, **zero-init, residual**
streaming temporal aligner on Phase 1's finest decoder feature (`feat_final`),
downsample-recur-upsample (Eq. 1–2), placed before the confidence/bin/depth
heads. Train it with the **existing metric SigLog loss** on short clips (4–8
frames) + stride augmentation, fast temporal LR (~1e-4) over a slow/frozen
backbone (~1e-6) — FlashDepth's Stage-1 recipe, re-pointed at metric GT. Make
Phase 1 inference streaming (per-sequence `reset()` + hidden-state carry). Keep
it OFF in `hdi*.yaml`, ON in a new `hdi_temporal.yaml`.

**3. Adopt:** (a) residual zero-init temporal module; (b) downsample-before-recur
efficiency trick (`downsample ≈ 0.1`); (c) short-clip + stride training recipe;
(d) two-tier LR. **Prefer a GRU/transformer-RNN backend over Mamba** to avoid the
`mamba-ssm` dependency, unless Mamba is later shown necessary.

**4. Leave unchanged:** DINOv2 backbone, Depth Range Feature Decoder, Eq. 1–4 bin
refinement, SigLog metric loss, Phase 2 (Mask2Former, depth-layer head, matcher),
Phase 3 (candidate/pair, Φo, compositing, losses), and every output contract's
shape.

**5. Per-phase modifications.** *Phase 1:* add optional temporal module + streaming
inference + a temporal training config/entry; no output-shape change. *Phase 2:*
none. *Phase 3:* none structural — it benefits from a steadier base depth; a
temporal `L_dist`-style consistency term across tracked frames is a possible
future add, not required.

**6. Expected benefits & trade-offs.** *Benefit:* markedly better temporal
consistency (less flicker) on your 297 sequences at ~1 % parameter and low
training cost — the cheap temporal path you wanted instead of VDA, low-risk by
zero-init construction. *Trade-offs:* a recurrent dependency (mitigated by the
swappable backend), stateful inference, training the module from scratch
(cheap), and **no boundary or metric-accuracy gain at 720p** — the value is
consistency, nothing else. Adopt it only if temporal flicker is a real problem
for your downstream use; otherwise the faithful per-frame Phase 1 remains the
correct default.
