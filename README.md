# InstanceDepth

A faithful reproduction of **"Instance-Level Video Depth in Groups Beyond
Occlusions"** (Liang et al., ICCV 2025), adapted to a custom single-sensor
RGB-D dataset of human-activity scenes.

The method predicts **instance-level, occlusion-aware metric depth**: not just
a per-pixel depth map, but per-person depth that stays geometrically consistent
where bodies overlap. It follows the paper's three-stage pipeline:

| Stage | Module | What it produces |
|-------|--------|------------------|
| **Phase 1** — Holistic Depth Initialization | `models/hdi` | Dense metric depth + multi-scale depth features (Eq. 1–4) |
| **Phase 2** — Instance Depth Layer Prediction | `models/phase2` | Per-instance mask, class, and depth layer `Dep_i` (Eq. 5–7) |
| **Phase 3** — Occlusion-Aware Depth Refinement | `models/phase3` | Occlusion-corrected per-instance + dense depth (Eq. 8–12) |

Every non-trivial design decision is tagged in-code as `[Paper Specified]`,
`[Strongly Inferred]`, or `[Reasonable Assumption]`.

📐 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — start here.** Ten diagrams
covering the full data flow: system overview, data engine, each phase's
internals, the temporal module, training stages, losses, tooling, and a
repository map.

| Doc | What it covers |
|---|---|
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | **The whole project in ten diagrams — start here** |
| **[videodepth/docs/DESIGN.md](videodepth/docs/DESIGN.md)** | **`videodepth/`: temporal consistency (TGM) + video-aware occlusion handling — the beyond-the-paper extension package** |

## Repository layout

```
instancedepth/
  configs/          Dataclass config trees + YAML profiles (one per phase/variant)
  data/             GID-style dataset + occlusion-frame sampler
  data_engine/      Annotation generation (SAM3 masks, tracking IDs, depth layers)
  models/
    backbone/       DINOv2 wrapper (vanilla or Depth-Anything-V2 encoder)
    hdi/            Phase 1: depth-range decoder, iterative bin refinement
    phase2/         Phase 2: Mask2Former wrapper, depth-layer head, matcher/criterion
    phase3/         Phase 3: candidate/pair construction, ROI extraction, Phi_o, compositing
  losses/           Phase 1 and Phase 3 loss functions
  engine/           Phase-agnostic Trainer + per-phase train_/evaluate_ entry points
  utils/            Metrics, camera/disparity, checkpointing, seeding, visualization
  predict.py        Unified depth-predictor factory for the video/viz tools
docs/               Architecture diagrams (ARCHITECTURE.md)
scripts/            Runnable tools (visualization, video generation, pipelines)
tests/              Unit / regression tests (synthetic tensors; no weights needed)
results/            Recorded eval JSONs (runs/ is gitignored)
```

## Setup

```bash
pip install -e .            # torch, transformers, opencv, numpy, pyyaml, tensorboard
pip install -e ".[dev]"     # + pytest
```

Pretrained weights (DINOv2 / Depth-Anything-V2 encoder, COCO Mask2Former) are
loaded from **local paths** set in the config files — the loaders are
fail-loud and never silently download. Point `backbone.checkpoint_path` /
`model.checkpoint_dir` at your local snapshots.

## Training

Each phase has a `train_<phase>` / `evaluate_<phase>` entry point sharing the
generic `engine/trainer.py`. Configs use a `--override key.subkey=value`
dotlist for ad-hoc changes.

```bash
# Phase 1 (vanilla DINOv2, or hdi_dav2.yaml for the DAv2 encoder)
python -m instancedepth.engine.train_hdi     --config instancedepth/configs/hdi.yaml
python -m instancedepth.engine.evaluate_hdi  --config instancedepth/configs/hdi.yaml     --checkpoint runs/hdi_faithful/best.pth

# Phase 2
python -m instancedepth.engine.train_phase2    --config instancedepth/configs/phase2_mask2former.yaml
python -m instancedepth.engine.evaluate_phase2 --config instancedepth/configs/phase2_mask2former.yaml --checkpoint runs/phase2_mask2former/best.pth

# Phase 3 (composes trained Phase 1 + Phase 2 checkpoints)
python -m instancedepth.engine.train_phase3    --config instancedepth/configs/phase3.yaml \
    --override phase1_checkpoint=runs/hdi_faithful/best.pth phase2_checkpoint=runs/phase2_mask2former/best.pth
python -m instancedepth.engine.evaluate_phase3 --config instancedepth/configs/phase3.yaml --checkpoint runs/phase3_refine/best.pth
```

Config profiles: `*.yaml` is the paper-faithful baseline; `*_enhanced.yaml`
enables opt-in, flagged deviations; `*_dav2.yaml` uses the Depth-Anything-V2
encoder; `phase3_current.yaml` runs Phase 3 on pre-existing checkpoints.
`scripts/run_full_pipeline.sh` (baseline) and `scripts/run_dav2_pipeline.sh`
(DAv2 ablation) chain full train+eval sequences for unattended runs.

## External baseline

Zero-shot Video-Depth-Anything (metric, ViT-L) on the test split, scored with
this repo's metrics for a directly comparable row. Requires a local clone of
[Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
(not a dependency of this repo); pass the same `--hdi-config` your compared
numbers use so GT and masks are identical.

```bash
python -m scripts.eval_vda_baseline \
    --vda-repo /path/to/Video-Depth-Anything \
    --vda-ckpt /path/to/metric_video_depth_anything_vitl.pth \
    --hdi-config instancedepth/configs/hdi_dav2.yaml --split test
```

## Qualitative evaluation

```bash
# Phase 3 debugging panels (RGB, GT, base vs refined, error maps, pair graph, ROI strips)
python -m scripts.visualize_phase3 --config instancedepth/configs/phase3.yaml \
    --checkpoint runs/phase3_refine/best.pth --out-dir viz/phase3

# Side-by-side comparison videos from test sequences
# (RGB | GT depth | prediction | instance masks with depth-layer labels)
python -m scripts.make_sequence_videos --phase 3 --config instancedepth/configs/phase3.yaml \
    --checkpoint runs/phase3_refine/best.pth --out-dir videos/phase3 --include-gt

# Inference on an arbitrary real-world video (generalization check)
python -m scripts.infer_video --phase 3 --config instancedepth/configs/phase3.yaml \
    --checkpoint runs/phase3_refine/best.pth --video input.mp4 --out videos/input_p3
```

## Tests

```bash
pytest tests/          # or: python -m pytest tests/test_phase3.py -v
```

Tests run on synthetic tensors and require no pretrained weights.

## Key adaptations to the custom dataset

- **Metric depth, scale-variant SigLog loss** (not scale-shift-invariant): the
  single calibrated RGB-D sensor provides true metric ground truth.
- **Box-IoU occlusion pairing**: GT instance masks are modal and disjoint, so
  mask IoU is structurally ~0 even under occlusion — overlap is detected via
  bounding boxes.
- **Visible-only supervision**: a single sensor has no ground truth for hidden
  pixels, so Phase 3's `L_obj` is masked to each instance's visible surface.
