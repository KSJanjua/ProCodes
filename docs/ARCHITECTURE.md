# InstanceDepth — Architecture

Visual reference for the whole project: how data, models, contracts and
training stages fit together. Rendered by GitHub natively (Mermaid).

Split into focused diagrams rather than one wall of boxes:

| # | Diagram | Answers |
|---|---------|---------|
| [1](#1-system-overview) | System overview | How do the three phases connect end to end? |
| [2](#2-data-engine-raw-capture--annotations) | Data engine | Where does the training data come from? |
| [3](#3-phase-1--holistic-depth-initialization) | Phase 1 internals | How is dense metric depth produced? (Eq. 1–4) |
| [4](#4-temporal-module-flashdepth-improvement) | Temporal module | The FlashDepth improvement over the baseline |
| [5](#5-phase-2--instance-depth-layer-prediction) | Phase 2 internals | How are instances + `Dep_i` produced? (Eq. 5–7) |
| [6](#6-phase-3--occlusion-aware-depth-refinement) | Phase 3 internals | How is occluded depth corrected? (Eq. 8–12) |
| [7](#7-training-stages--freezing-strategy) | Training stages | What is frozen/trained when, and at which LR? |
| [8](#8-loss-landscape) | Losses | Every loss term, where it applies, and its provenance |
| [9](#9-inference--tooling) | Inference & tooling | How do I get predictions, videos, metrics? |
| [10](#10-repository-map) | Repository map | Which file does what? |

Legend used throughout:

- **Blue** = Phase 1 (depth branch) · **Green** = Phase 2 (instance branch) ·
  **Orange** = Phase 3 (refinement) · **Grey** = data/IO · **Purple** = losses/metrics
- Solid arrows = tensor flow · dashed arrows = gradients / control / weights

---

## 1. System overview

The paper's two-stage design: Phase 1 and Phase 2 are **independent parallel
branches** that only meet in Phase 3. The three dataclass contracts
(`HolisticDepthOutput`, `Phase2Output`, `RefinedDepthOutput`) are the only
coupling points — each is versioned so a stale consumer fails loudly.

```mermaid
flowchart LR
    RGB([RGB frame<br/>B×3×H×W]):::io

    subgraph P1 ["PHASE 1 · Holistic Depth Initialization"]
        direction TB
        P1M["DINOv2 ViT-L/14<br/>+ Depth Range Decoder<br/>+ Eq.1–4 bin refinement"]:::p1
        P1M --> P1OUT[["HolisticDepthOutput v1.2<br/>depth_final · feat_final<br/>seg_final · feat_levels"]]:::contract
    end

    subgraph P2 ["PHASE 2 · Instance Depth Layer Prediction"]
        direction TB
        P2M["Swin-L Mask2Former<br/>+ DepthLayerHead<br/>(own backbone — independent)"]:::p2
        P2M --> P2OUT[["Phase2Output v1.0<br/>mask_logits · class_logits<br/>depth_layers Dep_i"]]:::contract
    end

    subgraph P3 ["PHASE 3 · Occlusion-Aware Depth Refinement"]
        direction TB
        P3M["Candidate filter → occlusion pairs<br/>→ ROIAlign → Φo (Eq.8–9)<br/>→ composite"]:::p3
        P3M --> P3OUT[["RefinedDepthOutput v1.0<br/>refined_depth · base_depth<br/>refined_layers · pairs"]]:::contract
    end

    RGB --> P1M
    RGB --> P2M
    P1OUT -->|"feat_final → F_obj<br/>depth_final → D_obj + base map"| P3M
    P2OUT -->|"masks, class conf,<br/>Dep_i → candidates & pairs"| P3M
    P3OUT --> FINAL([Occlusion-aware<br/>instance-level metric depth]):::io

    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef p3 fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef contract fill:#fef9c3,stroke:#ca8a04,color:#713f12
```

> **Why Phase 2 is independent:** it carries its own Swin-L backbone
> (COCO-pretrained Mask2Former). This deviates from the paper's
> shared-encoder prose but is what makes the Sec. 4.3 freeze strategy
> coherent — Phase 3 fine-tunes Phase 1's encoder while Phase 2 stays
> frozen, which would otherwise drift the "frozen" decoder's features.
> Consequence: **Phase 2's output is invariant to the Phase-1 checkpoint.**

---

## 2. Data engine (raw capture → annotations)

Run once, offline. Produces the GID-style annotations every phase consumes.

```mermaid
flowchart TB
    CAM([ZED stereo camera<br/>306 sequences · 50,323 frames<br/>720×1280 · mean 164 frames/seq]):::io
    CAM --> RGBF[/"RGB frames"/]:::io
    CAM --> DEPTHF[/"Depth maps<br/>metric metres, 0 = invalid"/]:::io

    RGBF --> SAM["SAM3 video tracking<br/>concept prompts: 'person'"]:::data
    SAM --> IDS["Identity merging + IoU repair<br/>(consistent track_ids)"]:::data
    IDS --> FLAT["Flatten to single-label id-map<br/>nearest-depth-wins per pixel"]:::data
    DEPTHF --> LAYER["Depth layer per instance<br/>Dep_i = mean of VALID depth in mask"]:::data
    IDS --> LAYER

    FLAT --> ANN[("gid_custom/<br/>object_masks · ground_masks<br/>annotations.json<br/>train.txt / test.txt")]:::io
    LAYER --> ANN
    DEPTHF --> ANN

    ANN --> DS1["GIDInstanceDepthDataset<br/>(shuffled frames)"]:::data
    ANN --> DS2["GIDClipDataset<br/>(ordered clips, stride-augmented)"]:::data
    ANN --> DS3["occlusion_index<br/>(frames w/ overlapping boxes)"]:::data

    classDef data fill:#e2e8f0,stroke:#475569,color:#0f172a
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
```

> **Two consequences of this pipeline that shape the whole project:**
> 1. **GT masks are modal and disjoint** — the id-map gives each pixel to
>    exactly one instance, so two occluding people have ~0 *mask* IoU.
>    Occlusion must be detected by **bounding boxes**.
> 2. **No GT behind occluders** — a single sensor only sees front surfaces,
>    so Phase 3's `L_obj` can only be supervised on *visible* pixels.

---

## 3. Phase 1 — Holistic Depth Initialization

Paper Sec. 4.1, Eq. 1–4. Depth is built **iteratively**: a seed `D_0`, then
three coarse-to-fine correction rounds.

```mermaid
flowchart TB
    IMG([RGB 728×1288]):::io --> BB["DINOv2 ViT-L/14<br/>hook layers 5 / 14 / 23"]:::p1
    BB -->|"3 × (B,1024,52,92)"| DEC

    subgraph DEC ["Depth Range Feature Decoder (Fig. 5)"]
        direction LR
        L0["level 0 · 1/8<br/>patch 4 → attn"]:::p1
        L1["level 1 · 1/4<br/>patch 8 → attn"]:::p1
        L2["level 2 · 1/2<br/>patch 16 → attn"]:::p1
        L0 -->|"top-down<br/>additive fusion"| L1 --> L2
    end

    L0 --> F0(["F_0"]):::feat
    L1 --> F1(["F_1"]):::feat
    L2 --> F2(["F_2"]):::feat

    F0 --> SEED["InitialDepthHead → D_0"]:::p1

    subgraph REF ["Iterative Bin Refinement (Eq. 1–4) — ×3 levels"]
        direction TB
        C["C_i = Sigmoid(Φd(F_i, D_i))<br/><i>Eq.1 · per-bin confidence</i>"]:::p1
        S["S_i = OrdinalBinHead(F_i)<br/><i>independent per-bin sigmoid, not softmax</i>"]:::p1
        R["R_i = Σ_bins (C_i · S_i)<br/><i>Eq.2</i>"]:::p1
        E["E_i = 2(R_i − 1)·(MAX_d / r_d)<br/><i>Eq.3</i>"]:::p1
        D["D_i+1 = up(D_i) + E_i<br/><i>Eq.4</i>"]:::p1
        C --> R
        S --> R --> E --> D
    end

    SEED --> REF
    F0 --> REF
    F1 --> REF
    F2 --> REF
    REF -->|"D_3"| UP["bilinear ↑ to full res"]:::p1
    UP --> OUT[["HolisticDepthOutput<br/>depth_final · seg_final<br/>feat_final (= F_2) · feat_levels"]]:::contract

    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef feat fill:#eff6ff,stroke:#60a5fa,color:#1e3a8a
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef contract fill:#fef9c3,stroke:#ca8a04,color:#713f12
```

> `r_d = 5` bins over `MAX_d = 10 m` = the paper's best "2-metre partitioning"
> ablation (Table 5). `S_i` is an **ordinal** (independent-sigmoid) encoding,
> not a softmax — a softmax would force Eq. 2–4's correction to be
> one-directional, which the equations don't support.

---

## 4. Temporal module (FlashDepth improvement)

The first research improvement over the reproduced baseline. **Off by default**
(`temporal.enabled: false`) so the faithful baseline stays bit-identical.

```mermaid
flowchart LR
    subgraph FRAME ["per frame t"]
        F2in(["F_2 (B,256,h,w)"]):::feat
        F2in --> DOWN["↓ Down ×0.1/side"]:::temporal
        DOWN --> PIN["1×1 proj → d_model 128"]:::temporal
        PIN --> GRU["ConvGRU ×2 blocks"]:::temporal
        GRU --> POUT["1×1 proj<br/><b>ZERO-INIT</b>"]:::temporal
        POUT --> UPS["↑ Up to h×w"]:::temporal
        UPS --> ADD(("+")):::temporal
        F2in --> ADD
        ADD --> F2out(["F_2 aligned → heads"]):::feat
    end

    H0[("H_t−1<br/>hidden state")]:::state -.->|"carried within<br/>a sequence"| GRU
    GRU -.->|"H_t"| H1[("H_t")]:::state
    RESET{{"reset_temporal_state()<br/>at every sequence start"}}:::state -.->|"H ← 0"| H0

    classDef temporal fill:#f3e8ff,stroke:#9333ea,color:#4c1d95
    classDef feat fill:#eff6ff,stroke:#60a5fa,color:#1e3a8a
    classDef state fill:#fae8ff,stroke:#c026d3,color:#701a75
```

**Why zero-init is the crux:** at step 0 the module outputs exactly 0, so
`F_2_aligned == F_2` — the model *is* the pretrained per-frame model, and an
empty-memory first frame can never be worse than the baseline. Zero *hidden
state* at sequence starts is FlashDepth's actual behaviour (there is no
learned initial state); the zero-init *output projection* is what makes that
safe.

| | Training | Inference |
|---|---|---|
| Input | 5-frame clips, strides {1,2,4,8} | full sequence, in order |
| State | reset per clip; full BPTT within it | carried across the sequence |
| Reset | every clip | every sequence boundary |
| Batch | independent clips | **1** (state is per-stream) |

---

## 5. Phase 2 — Instance Depth Layer Prediction

Paper Sec. 4.2.1, Eq. 5–7. Official COCO-pretrained Mask2Former + one new head.

```mermaid
flowchart TB
    IMG([RGB 736×1280]):::io --> SWIN["Swin-L backbone"]:::p2
    SWIN --> PIXDEC["MSDeformAttn pixel decoder"]:::p2
    PIXDEC --> TDEC["Transformer decoder<br/>200 queries · 9 layers<br/>masked attention"]:::p2

    TDEC -->|"query embeddings<br/>(B,200,256)"| CLS["class head"]:::p2
    TDEC -->|"query embeddings"| MSK["mask head"]:::p2
    TDEC -->|"query embeddings"| DEP["<b>DepthLayerHead</b> (new)<br/>MLP + Softplus<br/><i>Eq.5–7 · Dep_i</i>"]:::new

    CLS --> OUT[["Phase2Output<br/>mask_logits · class_logits<br/>depth_layers · query_embeddings"]]:::contract
    MSK --> OUT
    DEP --> OUT

    OUT -.->|"training only"| MATCH["Hungarian matcher<br/>cost = 2·class + 5·mask<br/>+ 5·dice + 1·depth"]:::loss
    MATCH -.-> CRIT["Phase2Criterion<br/>CE + point-sampled BCE/Dice<br/>+ smooth-L1 on Dep_i"]:::loss

    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef new fill:#bbf7d0,stroke:#15803d,color:#14532d,stroke-width:3px
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef contract fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef loss fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
```

> **`Dep_i` never reads Phase 1.** It is an MLP on Mask2Former's own query
> embeddings, trained directly against GT depth layers ("Reading A" of the
> paper's ambiguous "query fusion"). This is why Phase-2 output cannot show
> any effect of the temporal module.

---

## 6. Phase 3 — Occlusion-Aware Depth Refinement

Paper Sec. 4.2.2, Eq. 8–12. The join point.

```mermaid
flowchart TB
    P2IN[["Phase2Output<br/>(frozen)"]]:::contract --> FILT
    P1IN[["HolisticDepthOutput<br/>(trainable)"]]:::contract --> ROI

    subgraph FRONT ["Candidate & pair construction (non-differentiable)"]
        direction TB
        FILT["filter queries<br/>cat conf > 0.9 ∧ mask conf > 0.8"]:::p3
        FILT --> OVER["overlap: <b>box</b> IoU > 0.1<br/><i>(mask IoU is ~0 — modal GT)</i>"]:::p3
        OVER --> GUEST["keep depth-nearest partner<br/>→ dedup to unordered pairs<br/>order: nearer (occluder) first"]:::p3
    end

    GUEST -->|"P pairs<br/>+ normalized boxes"| ROI

    subgraph ROI ["ROIAlign (normalized boxes → shared Hp×Wp)"]
        direction TB
        FOBJ["F_obj = ROIAlign(feat_final)<br/>(P,2,C,28,28)"]:::p3
        DOBJ["D_obj = ROIAlign(depth_final)<br/>(P,2,1,28,28)"]:::p3
        GOBJ["G_obj = mask logits ⊕ norm coords<br/>⊕ global depth"]:::p3
    end

    FOBJ --> PHI
    GOBJ --> PHI
    PHI["<b>Φo</b> · 1×1 conv MLP<br/>pair-coupled (main ‖ guest)<br/>E_obj = Sigmoid(Φo([F_obj,G_obj]))<br/><i>Eq.8 · ZERO-INIT</i>"]:::p3
    PHI --> EQ9["D̂ = (2E−1)·D_obj + D_obj = <b>2E·D_obj</b><br/><i>Eq.9 · multiplicative ratio</i>"]:::p3
    DOBJ --> EQ9

    EQ9 --> COMP["composite: refined = up(2E) × base<br/>inside mask · feathered edges<br/>nearest-<b>layer</b>-wins across instances"]:::p3
    EQ9 -.-> LOSS["L_obj = SigLog(D̂, DT) <i>Eq.10</i><br/>L_dist = ‖ΔD̂² − ΔDT²‖ <i>Eq.11</i><br/>L_ref = λ₁L_obj + λ₂L_dist <i>Eq.12</i>"]:::loss
    COMP --> OUT[["RefinedDepthOutput<br/>refined_depth · base_depth<br/>refined_layers · pairs"]]:::contract

    classDef p3 fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef contract fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef loss fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
```

> **Eq. 9 is a ratio, not a paste.** `D̂ = 2E·D_obj` means the refinement is a
> *multiplicative correction field*. Compositing therefore applies the
> upsampled **ratio** to full-resolution base depth — so `E = 0.5` is an exact
> no-op and the base map's fine geometry is preserved, merely modulated.
> (Pasting the 28×28 ROI depth instead was the primary defect in the first
> implementation.)

---

## 7. Training stages & freezing strategy

Paper Sec. 4.3 prescribes the three stages; the temporal stage is the
project's addition.

```mermaid
flowchart TB
    S1["<b>Stage 1 · Phase 1</b><br/>train_hdi<br/>55k iters · LR 1e-5<br/>everything trainable"]:::p1
    S1 --> CK1[("runs/hdi_*/best.pth")]:::io

    CK1 --> S1B["<b>Stage 1b · Temporal</b> (new)<br/>train_hdi_temporal<br/>8k iters · LR 1e-4<br/>❄ spatial frozen · 🔥 Φtemporal only<br/>1.84M / 357.7M trainable"]:::temporal
    S1B --> CK1B[("runs/hdi_temporal/best.pth")]:::io

    S2["<b>Stage 2 · Phase 2</b><br/>train_phase2<br/>20k iters · LR 1e-4<br/>backbone ×0.1<br/><i>independent of Phase 1</i>"]:::p2
    S2 --> CK2[("runs/phase2_*/best.pth")]:::io

    CK1 --> S3
    CK2 --> S3
    S3["<b>Stage 3 · Phase 3</b><br/>train_phase3<br/>25k iters · LR 1e-6<br/>❄ Phase 2 frozen<br/>🔥 Phase 1 fine-tuned + Φo"]:::p3
    S3 --> CK3[("runs/phase3_*/best.pth")]:::io

    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef p3 fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef temporal fill:#f3e8ff,stroke:#9333ea,color:#4c1d95
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
```

> ⚠️ **Stage 3 hazard (observed, then fixed):** Phase 3 fine-tunes the *whole*
> depth branch but its losses only touch paired-instance ROIs — the first run
> drifted `overall_base` abs_rel **0.078 → 0.139** while supervised instance
> regions stayed at 0.078. Classic catastrophic forgetting. The run profiles
> now enable `holistic_weight: 0.3` (a dense anti-forgetting term, a flagged
> deviation from Eq. 12).

---

## 8. Loss landscape

```mermaid
flowchart LR
    subgraph L1 ["Phase 1 — paper specifies NO loss ⇒ free choice"]
        direction TB
        A1["SigLog(D_final, GT)<br/><i>scale-VARIANT at λ=0.5</i>"]:::pspec
        A2["deep supervision<br/>SigLog on D_1, D_2 · w = .5/.25"]:::infer
        A3["ordinal bin BCE on S_0..S_2"]:::infer
        A4["gradient matching ×0.5<br/><i>sharpens edges — SigLog is pointwise</i>"]:::optin
        A5["canonical disparity L1 ×0.5<br/><i>near-field emphasis</i>"]:::optin
    end

    subgraph L2 ["Phase 2 — Eq. 5–7 (paper specified)"]
        direction TB
        B1["CE class (eos 0.1) ×2"]:::pspec
        B2["point-sampled BCE ×5 + Dice ×5"]:::pspec
        B3["smooth-L1 on Dep_i ×1"]:::pspec
    end

    subgraph L3 ["Phase 3 — Eq. 10–12 (paper specified)"]
        direction TB
        C1["L_obj = SigLog(D̂, DT) ×1<br/><i>visible GT pixels only</i>"]:::pspec
        C2["L_dist = ‖ΔD̂² − ΔDT²‖ ×0.5"]:::pspec
        C3["holistic anti-forgetting ×0.3"]:::optin
    end

    classDef pspec fill:#ddd6fe,stroke:#6d28d9,color:#3b0764
    classDef infer fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95
    classDef optin fill:#fce7f3,stroke:#db2777,color:#831843
```

**Legend:** ▉ paper-specified · ▉ inferred · ▉ opt-in deviation (off in faithful profiles)

> **On SigLog vs SSI** — a recurring question. *SSI* (MiDaS/DPT) least-squares
> aligns scale **and shift** before comparing, discarding metric scale: that is
> the *multi-source* loss, and this project never uses it. *SILog* (Eigen 2014)
> at λ<1 **penalizes** absolute scale — substituting `pred → α·pred` leaves a
> residual `(1−λ)[(log α)² − 2 log α·mean(d)]` that vanishes only at λ=1. It is
> the standard single-sensor metric-depth loss (BTS, AdaBins, NeWCRFs,
> ZoeDepth). Pinned by `test_siglog_penalizes_absolute_scale`.

---

## 9. Inference & tooling

```mermaid
flowchart TB
    CFG[/"YAML profile<br/>+ --override dotlist"/]:::io --> FACT
    CKPT[("checkpoint")]:::io --> FACT

    FACT["predict.py<br/><b>build_depth_predictor</b> (phase 1|3)<br/><b>build_scene_predictor</b> (phase 1|2|3)"]:::tool

    FACT --> I1["HDIInferencer<br/>(Phase 1)"]:::p1
    FACT --> I2["Phase2Model<br/>(instances only)"]:::p2
    FACT --> I3["Phase3Inferencer<br/>(full pipeline)"]:::p3

    I1 --> TOOLS
    I2 --> TOOLS
    I3 --> TOOLS

    subgraph TOOLS ["Tools"]
        direction TB
        V1["make_sequence_videos<br/>RGB | GT | pred | instances"]:::tool
        V2["infer_video<br/>arbitrary real video"]:::tool
        V3["visualize_phase3<br/>12-panel debug grid + ROI strips"]:::tool
        V4["visualize_hdi / visualize_phase2<br/>TensorBoard strips"]:::tool
    end

    subgraph EVAL ["Evaluation"]
        direction TB
        E1["evaluate_hdi<br/>REL/RMS/σ + --streaming → <b>TAE</b>"]:::metric
        E2["evaluate_phase2<br/>P/R/F1/IoU + Dep MAE<br/>+ <b>COCO AP/AP50/AP75/APs/m/l</b>"]:::metric
        E3["evaluate_phase3<br/>refined vs base × overall vs occlusion"]:::metric
    end

    PIPE{{"run_full_pipeline.sh<br/>P1 → P1b → P2 → P3<br/>train + eval + viz + video<br/><i>fault-tolerant</i>"}}:::tool
    PIPE -.-> TOOLS
    PIPE -.-> EVAL
    EVAL --> RES[("results/*.json")]:::io

    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef p3 fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef tool fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef metric fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
```

**Fault tolerance in the pipeline:** training failures gate only their true
dependents (Phase 2 runs even if Phase 1 fails; Phase 3 needs both
checkpoints). Eval / visualization / video failures are recorded and skipped
past, and a pass/fail/skip summary sets the exit code.

---

## 10. Repository map

```mermaid
flowchart LR
    subgraph CFGD ["configs/"]
        direction TB
        CF1["config.py · HDIConfig<br/>(+ TemporalConfig)"]:::p1
        CF2["phase2_config.py"]:::p2
        CF3["phase3_config.py"]:::p3
        CF4["hdi{,_enhanced,_dav2,_temporal}.yaml<br/>phase2{,_dav2}.yaml<br/>phase3{,_enhanced,_dav2,_current}.yaml"]:::io
    end

    subgraph MODELS ["models/"]
        direction TB
        M0["backbone/dinov2_wrapper.py<br/><i>vanilla DINOv2 | DAv2 encoder</i>"]:::p1
        M1["hdi/ · depth_range_decoder · bin_heads<br/>iterative_refinement · temporal · model"]:::p1
        M2["phase2/ · mask2former_wrapper · depth_head<br/>matcher · criterion · model"]:::p2
        M3["phase3/ · candidates · roi_extract<br/>relation_head · targets · model"]:::p3
    end

    subgraph REST ["engine/ · losses/ · data/ · utils/"]
        direction TB
        E["engine/ · trainer (phase-agnostic)<br/>train_/evaluate_ per phase"]:::tool
        LO["losses/ · hdi_losses · phase3_losses"]:::metric
        DA["data/ · gid_dataset · clip_dataset<br/>occlusion_index"]:::io
        UT["utils/ · metrics · phase2_metrics (COCO AP)<br/>viz · camera · checkpoint"]:::tool
    end

    subgraph DOCS ["docs/"]
        direction TB
        D6["ARCHITECTURE.md · this file"]:::doc
        D7["videodepth/docs/DESIGN.md · extension package"]:::doc
    end

    classDef p1 fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef p2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef p3 fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef tool fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef metric fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef doc fill:#fef3c7,stroke:#d97706,color:#78350f
```

---

## Key invariants (the things that must stay true)

| Invariant | Why it matters | Pinned by |
|---|---|---|
| Contracts are versioned dataclasses | a stale consumer fails loudly, never silently misreads a tensor | `test_output_contract.py` |
| `temporal.enabled: false` everywhere by default | the paper-faithful baseline stays bit-identical and comparable | `test_temporal_config_wiring` |
| Zero-init ⇒ new modules start as exact no-ops (Φo, temporal) | training can only *earn* a change; never a cold-start regression | `test_aligner_identity_at_init`, `test_composite_identity_at_e_half` |
| Occlusion detected by **box** IoU, never mask IoU | GT masks are modal/disjoint ⇒ mask IoU is structurally ~0 | `test_box_iou_catches_modal_disjoint_masks...` |
| Phase-3 supervision masked to **visible** GT | one sensor has no ground truth behind occluders | `targets.build_dense_gt_rois` |
| Phase 2 never reads Phase 1 | its output is invariant to the Phase-1 checkpoint (incl. temporal) | `predict.py` docstring |
