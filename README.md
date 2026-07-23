# InstanceDepth

A reproduction of **"Instance-Level Video Depth in Groups Beyond Occlusions"**
(Liang et al., ICCV 2025), adapted to a **single-sensor RGB-D** dataset of
human-activity scenes, plus a **streaming temporal-consistency extension**.

The method predicts **instance-level, occlusion-aware metric depth**: not just a
per-pixel depth map, but per-person depth that stays geometrically consistent
where bodies overlap. It follows the paper's three-stage pipeline, and adds an
optional temporal stabilizer on top:

| Stage | Package / module | What it produces |
|-------|------------------|------------------|
| **Phase 1** — Holistic Depth Initialization | `instancedepth/models/hdi` | Dense metric depth + multi-scale depth features (Eq. 1–4) |
| **Phase 2** — Instance Depth-Layer Prediction | `instancedepth/models/phase2` | Per-instance mask, class, and depth layer `Dep_i` (Eq. 5–7) |
| **Phase 3** — Occlusion-Aware Refinement | `instancedepth/models/phase3`, `videodepth/` | Occlusion-corrected per-instance depth (Eq. 8–12) |
| **Temporal** *(extension)* | `videodepth/` | Streaming ConvGRU depth stabilizer trained with a TGM loss |

> 📐 **New here? Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first** — the
> whole project in ten diagrams. The temporal/video package has its own design
> note: [videodepth/docs/DESIGN.md](videodepth/docs/DESIGN.md).

---

## Results at a glance

These are the recorded reference numbers (held-out **test** split: 61 sequences
/ 10,400 frames; occlusion slice: 3,083 frames). Each row lists the config and
run that produce it and the reference JSON in [`results/`](results/). See
[§4 Reproduce the results](#4-reproduce-the-results) for the exact commands.

| What | Headline metric | Config → run | Reference JSON |
|------|-----------------|--------------|----------------|
| **Phase 1** dense depth | AbsRel **0.078**, δ₁ 0.938 | `hdi_dav2.yaml` → `hdi_dav2` | `results/hdi_dav2_eval.json` |
| **Phase 2** instances | F1 **0.943**, mIoU 0.979, depth-MAE 0.060 m | `phase2_mask2former.yaml` → `phase2_run` | `results/phase2_run_eval.json` |
| **Phase 3** refinement | occ AbsRel **0.0592 → 0.0554** (−6.5%), frame unchanged | `phase3_dav2_p2run.yaml` → `phase3_video_dav2` | `results/phase3_video_dav2_eval.json` |
| **Temporal** extension | AbsRel **0.0778 → 0.0724** (−6.9%), TAE 0.0558 → 0.0547 | `video_temporal_dav2.yaml` → `video_temporal_dav2` | `results/video_temporal_dav2_eval_streaming.json` |

Improvements are **modest and consistent**; per-sequence significance testing is
available (see [§5](#5-ablations--baselines)) but not yet run on these numbers.

---

## Repository layout

```
instancedepth/            # the 3-stage pipeline (per-frame)
  configs/                # dataclass config trees + YAML profiles (one per phase/variant)
  data/                   # dataset loader + occlusion-frame sampler
  data_engine/            # automatic annotation (SAM-family masks, track IDs, depth layers)
  models/{backbone,hdi,phase2,phase3}
  losses/  engine/  utils/
  meta.json               # dataset statistics (306 seq / 50,323 frames)
videodepth/               # temporal + video-aware extension (the headline P3 + temporal runs)
  configs/                # video_temporal_dav2.yaml, phase3_dav2_p2run.yaml, ...
  engine/                 # train_video / evaluate_video / train_phase3_video / ...
  run_pipeline.sh         # one-command temporal + Phase-3 reproduction
docs/ARCHITECTURE.md      # start here
scripts/                  # baselines, significance, visualization, video tools
results/                  # recorded eval JSONs (reference numbers).  runs/ is gitignored
tests/                    # unit/regression tests (synthetic tensors; no weights needed)
```

---

## 1. Install

```bash
git clone <this-repo> && cd <repo-dir>
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .            # torch>=2.2, transformers>=4.44, opencv, numpy, pyyaml, tensorboard
pip install -e ".[dev]"     # + pytest (to run the tests)
```

Requires Python ≥ 3.10 and a CUDA GPU for training/eval (the tests run on CPU).

**Verify the install end-to-end without data or long training** — a 5–10 min
smoke run that exercises every real code path (checkpoint load, data load,
forward/backward, checkpoint + eval-JSON writing) with ~3 iterations, in
throwaway `runs/smoke_*` dirs:

```bash
SMOKE=1 bash videodepth/run_pipeline.sh
```

## 2. Pretrained weights

The loaders are **fail-loud and never silently download**. Fetch these public
checkpoints once, then point the configs at your local copies:

| Weight | Used by | Source |
|--------|---------|--------|
| DINOv2 ViT-L/14 (`dinov2_vitl14.safetensors`) | Phase 1 (vanilla) | `facebook/dinov2-large` (Hugging Face) |
| Depth-Anything-V2 ViT-L (`depth_anything_v2_vitl.pth`) | Phase 1 (headline) | [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) releases |
| Mask2Former Swin-L COCO-instance | Phase 2 | `facebook/mask2former-swin-large-coco-instance` (Hugging Face) |

Set the paths either by editing the YAML, or per-run with `--override`:

- Phase 1: `backbone.checkpoint_path=/path/to/depth_anything_v2_vitl.pth`
- Phase 2: `model.checkpoint_dir=/path/to/mask2former-swin-large-coco-instance`

> The committed configs point at the authors' server paths
> (`/home/work/intern_storage/...`). Replace them with your own.

## 3. Data

The corpus is a **custom single-sensor ZED RGB-D capture** of human-activity
scenes (306 sequences, 50,323 frames, one category — *person*; metric range
0.01–10 m; 245 train / 61 test, video-level split, seed 2026). It is **not
redistributed** here, so exact-number reproduction is possible for the authors
(on the server) or for anyone with the raw captures. The pipeline itself runs on
**any** RGB-D video organized in the expected layout.

**Expected raw layout** (per [`configs/gid_custom.yaml`](instancedepth/configs/gid_custom.yaml)):

```
InstanceDepth/Dataset/
  Batch 1/ .. Batch 10/
    <sequence>/
      left_rgb/       # RGB frames
      left_filled/    # depth PNGs   (and/or)
      left_filled_np/ # depth .npy   (preferred; metric metres or mm, auto-detected)
```

**Generate the instance annotations** (masks, track IDs, per-instance depth
layers → written to `gid_custom/`). This needs a SAM-family video segmenter
installed separately (`sam3.backend` in the config; `mock` for a dry run):

```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml     # add --limit N to annotate the first N sequences
```

All training/eval configs then read `data.annotations_root: gid_custom`.

---

## 4. Reproduce the results

Eval scripts write `runs/<run_name>/eval_<split>[_streaming].json`. The files in
[`results/`](results/) are recorded copies of those (renamed
`<run_name>_eval[...].json`) — compare your run against them.

### One command (temporal + Phase-3)

With trained Phase-1 (`runs/hdi_dav2/best.pth`) and Phase-2
(`runs/phase2_run/best.pth`) checkpoints present, this reproduces the temporal
and Phase-3 rows end to end (baseline eval → temporal train → temporal eval →
Phase-3 train → Phase-3 eval):

```bash
bash videodepth/run_pipeline.sh
# override checkpoints/configs via env vars, e.g.:
#   P1_CKPT=runs/hdi_dav2/best.pth P2_CKPT=runs/phase2_run/best.pth bash videodepth/run_pipeline.sh
```

### Step by step

Each phase has a `train_*` / `evaluate_*` entry point. Configs take
`--override key.subkey=value` dotlists and an optional `--run-name`.

```bash
# ── Phase 1: holistic depth (DAv2 encoder) ──────────────────────────────────
python -m instancedepth.engine.train_hdi     --config instancedepth/configs/hdi_dav2.yaml
python -m instancedepth.engine.evaluate_hdi  --config instancedepth/configs/hdi_dav2.yaml \
    --checkpoint runs/hdi_dav2/best.pth --split test                    # -> results/hdi_dav2_eval.json
python -m instancedepth.engine.evaluate_hdi  --config instancedepth/configs/hdi_dav2.yaml \
    --checkpoint runs/hdi_dav2/best.pth --split test --streaming        # -> results/hdi_dav2_eval_streaming.json (temporal baseline)

# ── Phase 2: instances + depth layers (run named phase2_run) ────────────────
python -m instancedepth.engine.train_phase2    --config instancedepth/configs/phase2_mask2former.yaml --run-name phase2_run
python -m instancedepth.engine.evaluate_phase2 --config instancedepth/configs/phase2_mask2former.yaml \
    --checkpoint runs/phase2_run/best.pth --split test                  # -> results/phase2_run_eval.json

# ── Phase 3: occlusion refinement, bounded pair-attention head (videodepth) ─
python -m videodepth.engine.train_phase3_video    --config videodepth/configs/phase3_dav2_p2run.yaml \
    --override phase1_checkpoint=runs/hdi_dav2/best.pth phase2_checkpoint=runs/phase2_run/best.pth
python -m videodepth.engine.evaluate_phase3_video --config videodepth/configs/phase3_dav2_p2run.yaml \
    --checkpoint runs/phase3_video_dav2/best.pth --split test           # -> results/phase3_video_dav2_eval.json

# ── Temporal stabilizer, TGM loss (videodepth) ──────────────────────────────
python -m videodepth.engine.train_video    --config videodepth/configs/video_temporal_dav2.yaml \
    --override init_checkpoint=runs/hdi_dav2/best.pth
python -m videodepth.engine.evaluate_video --config videodepth/configs/video_temporal_dav2.yaml \
    --checkpoint runs/video_temporal_dav2/best.pth --split test         # -> results/video_temporal_dav2_eval_streaming.json
```

Phases run **in order** (Phase 3 needs Phase 1 + Phase 2 checkpoints; the
temporal stage needs the Phase-1 checkpoint). Training honours `seed: 2026` in
every config.

---

## 5. Ablations & baselines

```bash
# Encoder / loss ablations (Phase 1)
python -m instancedepth.engine.train_hdi --config instancedepth/configs/hdi.yaml            # vanilla DINOv2  -> runs/hdi_faithful
python -m instancedepth.engine.train_hdi --config instancedepth/configs/hdi_enhanced.yaml   # vanilla + disparity/grad-matching aux
python -m instancedepth.engine.train_hdi --config instancedepth/configs/hdi_dav2.yaml \
    --override loss.regression=berhu --run-name hdi_dav2_berhu                               # berHu instead of SigLog

# External baseline: zero-shot metric Video-Depth-Anything, scored with THIS repo's metrics.
# Needs a local clone of Video-Depth-Anything (not a dependency). Use --median-align for a fair,
# scale-aligned comparison (raw metric VDA is off-domain on this data).
python -m scripts.eval_vda_baseline --vda-repo /path/to/Video-Depth-Anything \
    --vda-ckpt /path/to/metric_video_depth_anything_vitl.pth \
    --hdi-config instancedepth/configs/hdi_dav2.yaml --split test --median-align

# Per-sequence significance (the 61 test sequences are the independent unit).
# Step 1 needs the checkpoint (GPU); step 2 is pure numpy.
python -m scripts.eval_phase3_per_sequence --config videodepth/configs/phase3_dav2_p2run.yaml \
    --checkpoint runs/phase3_video_dav2/best.pth --run-name phase3_video_dav2
python -m scripts.paired_significance within --file results/phase3_video_dav2_per_sequence.json
```

## 6. Qualitative tools

`make_sequence_videos` and `infer_video` build the model through
`instancedepth/predict.py`, which **auto-detects** the checkpoint's head
(paper-faithful relation head, `videodepth` bounded pair-attention head, or
temporal stabilizer) — so they accept the headline `videodepth` runs directly.
`visualize_phase3` is hard-wired to the instancedepth **relation-head** Phase-3,
so give it an instancedepth Phase-3 config + a relation-head checkpoint.

```bash
# Phase-3 debug panels (RGB, GT, base vs refined, error maps, pair graph, ROI strips)
#   -- relation-head Phase-3 only (instancedepth config + checkpoint)
python -m scripts.visualize_phase3 --config instancedepth/configs/phase3_dav2.yaml \
    --checkpoint runs/phase3_dav2/best.pth --out-dir viz/phase3

# Side-by-side comparison videos from test sequences (auto-detects the head)
python -m scripts.make_sequence_videos --phase 3 --config videodepth/configs/phase3_dav2_p2run.yaml \
    --checkpoint runs/phase3_video_dav2/best.pth --out-dir videos/phase3 --include-gt

# Inference on an arbitrary real-world video (auto-detects the head)
python -m scripts.infer_video --phase 3 --config videodepth/configs/phase3_dav2_p2run.yaml \
    --checkpoint runs/phase3_video_dav2/best.pth --video input.mp4 --out videos/input_p3
```

## 7. Tests

```bash
pytest tests/          # synthetic tensors; no pretrained weights or data needed
```

---

## Config profiles

`*.yaml` = paper-faithful baseline · `*_dav2.yaml` = Depth-Anything-V2 encoder ·
`*_enhanced.yaml` = opt-in, flagged extra losses · `video_*` / `phase3_*_p2run` =
the temporal/Phase-3 runs in `videodepth/`.

| Config | Package | Run name | Role |
|--------|---------|----------|------|
| `hdi.yaml` | instancedepth | `hdi_faithful` | Phase 1, vanilla DINOv2 |
| `hdi_dav2.yaml` | instancedepth | `hdi_dav2` | **Phase 1, headline (DAv2 encoder)** |
| `hdi_enhanced.yaml` | instancedepth | `hdi_enhanced` | Phase 1 ablation (aux losses) |
| `phase2_mask2former.yaml` | instancedepth | `phase2_mask2former` / `phase2_run` | **Phase 2, headline** |
| `phase2_dav2.yaml` | instancedepth | `phase2_dav2` | Phase 2, isolated re-run |
| `phase3_dav2_p2run.yaml` | videodepth | `phase3_video_dav2` | **Phase 3, bounded head** |
| `video_temporal_dav2.yaml` | videodepth | `video_temporal_dav2` | **Temporal stabilizer** |

## Key adaptations to the single-sensor setting

- **Metric, scale-variant SigLog loss** (not scale-shift-invariant): the
  calibrated RGB-D sensor provides true metric ground truth.
- **Box-IoU occlusion pairing**: GT masks are modal and disjoint, so mask IoU is
  ~0 even under occlusion — overlap is detected from bounding boxes.
- **Visible-only, non-degrading refinement**: a single sensor has no ground
  truth behind an occluder, so Phase 3's `L_obj` is masked to each instance's
  visible pixels, and the Phase-1 branch is frozen so refinement cannot degrade
  the dense map.
- **Temporal-gradient-matching (TGM) stabilizer**: matches frame-to-frame
  log-depth changes to the ground truth, penalizing flicker without punishing
  genuine motion.
