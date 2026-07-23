# InstanceDepth — Technical Walkthrough

**Dataset Creation · Phase 1 (Holistic Depth) · Phase 2 (Instance Depth) · Phase 3 (Occlusion) · Temporal · Training**

This document explains, in plain language, how the whole InstanceDepth system works: how we
manufacture the training labels from raw video, how each of the three phases produces its part of
the answer, how the video (temporal) extension keeps depth steady over time, and how everything is
trained. Every Computer-Vision term is defined the first time it appears, and each Part opens with
a short vocabulary table you can look back at.

The system reproduces *"Instance-Level Video Depth in Groups Beyond Occlusions"*
(Liang et al., ICCV 2025), adapted to a custom single-sensor RGB-D dataset of
human-activity scenes. It is a three-stage pipeline:

| Stage | What it produces |
|-------|------------------|
| **Phase 1** — Holistic Depth Initialization | Dense metric depth map + multi-scale depth features |
| **Phase 2** — Instance Depth Layer Prediction | Per-person mask, class, and depth layer `Dep_i` |
| **Phase 3** — Occlusion-Aware Depth Refinement | Occlusion-corrected per-instance + dense depth |

## New to all of this? Read this first (60 seconds)

Imagine a video of several people in a room. We want the computer to answer, for **every pixel**,
"how many metres away is this?" — and to do it *per person*, staying correct even where people
**overlap** (one standing in front of another). That is the whole goal.

The system reaches it in stages, and this document follows them in order:

- **Part A — Dataset creation.** The camera gives us the colour picture and a rough distance
  reading, but *not* the answers the model must learn from (which pixels are which person, and how
  far each person is). So we first build those answer-labels automatically.
- **Part B — Phase 1.** Predict a plain distance-for-every-pixel map from one picture.
- **Part C — Phase 2.** Find each separate person — their outline, their label ("person"), and
  their rough distance.
- **Part D — Phase 3.** Fix the distances exactly where people overlap (the hard part the paper
  is named after).
- **Part E — Temporal.** Stop the distances from flickering frame-to-frame in a video.
- **Part F — Training.** How each of the above is actually *taught* to the computer.

You don't need any prior computer-vision knowledge — every term ("mask", "depth", "backbone",
"loss", …) is explained in plain words as it comes up. All parts (A–F) are covered below.

---

# PART A — GROUND-TRUTH DATASET CREATION

## A.0 The big picture

We have a camera (a ZED stereo camera) that recorded many videos of people. For each
video we already have two things per frame: the **color image** (RGB) and the **depth**
(how far every pixel is from the camera). But the model needs *more* than that to learn.
It needs to know **which pixels belong to which person**, **the same person's identity
across all frames**, and **how far each person is on average**. Nobody labeled that by
hand. This module is an automatic annotator: it reads the raw videos and writes out all
those extra labels — the **Ground Truth** (the "correct answers" the model trains against).

**Why it must exist:** with only raw RGB + depth, the model has no idea who is who.
Phase 1 (dense depth) could still train, but Phase 2 (per-instance) and Phase 3
(occlusion) would have no targets at all. This module makes the instance side of the
project possible.

The pipeline lives in `instancedepth/data_engine/`.

## A.1 Shared vocabulary

| Term | Meaning |
|---|---|
| **RGB image** | A normal color photo: a grid of pixels, each with Red/Green/Blue values. e.g. `720 × 1280 × 3`. |
| **Depth map** | Same width/height as the photo, but each pixel holds one number = meters from the camera. `0` means "sensor could not measure here." |
| **Segmentation / mask** | A black-and-white stencil the size of the image: `True` where an object's pixels are, `False` elsewhere. |
| **Instance** | One specific object. Two people = two instances. |
| **Tracking / track ID** | Following the *same* instance across frames, giving it one persistent number. |
| **Occlusion** | One object partly hiding another (a person walking in front of another). |
| **IoU (Intersection over Union)** | Overlap area ÷ combined area of two masks; 1 = identical, 0 = no overlap. Used to decide if two masks are the same object. |
| **Depth layer (`Dep_i`)** | One number per person = the average depth of their visible pixels. |
| **bbox (bounding box)** | Smallest rectangle around a mask: `[x_min, y_min, x_max, y_max]`. |

## A.2 Folder and files

`instancedepth/data_engine/` — seven files:

| File | Job |
|---|---|
| `config.py` | The settings sheet: all knobs (folder names, depth range, thresholds); loads from YAML. |
| `discover.py` | Walks the dataset folders, finds every video, pairs each RGB frame with its depth file. |
| `depth_io.py` | Reads raw depth files and converts them into clean meters, marking bad pixels as 0. |
| `sam3_engine.py` | Runs the SAM3 model to find & track "person"/"floor" across a video → raw masks. |
| `identity.py` | Fixes broken/duplicate track IDs so each person has one consistent identity. |
| `annotate.py` | Combines masks + depth into the final files: `object_masks`, `ground_masks`, `annotations.json`. |
| `run_generate.py` | Runs the whole thing on all videos, then splits into train/test and writes stats. |

## A.3 End-to-end pipeline

```
        RAW DATASET (from the ZED camera)
        Dataset/Batch 1..10/<timestamp>/
            left_rgb/       frame_*.jpg     ← color photos
            left_filled/    frame_*.png     ← depth (16-bit)
            left_filled_np/ frame_*.npy     ← depth (float)
                     │
                     ▼
   run_generate.py  ── one command starts everything
                     │
        ┌────────────┼─────────────────────────────┐
        ▼            ▼                               ▼
   discover.py   sam3_engine.py                depth_io.py
   find videos,  run SAM3 → raw masks          read raw depth
   pair rgb+dep  + track IDs (per concept)     → clean meters
        │            │                               │
        │            ▼                               │
        │        identity.py                         │
        │        merge concepts, remove duplicates,  │
        │        repair broken IDs, drop junk        │
        │            │                               │
        └────────────┴───────────────┬───────────────┘
                                      ▼
                                 annotate.py
                    per frame: bbox + depth layer per person,
                    flatten to one id-map (nearest-depth wins),
                    write files
                                      │
                                      ▼
              OUTPUT:  gid_custom/
                         Batch x/<timestamp>/
                            object_masks/frame_*.png   (pixel value = track ID)
                            ground_masks/frame_*.png   (floor = white)
                            annotations.json           (per-person bbox, depth, etc.)
                         train.txt  test.txt  meta.json
                                      │
                                      ▼
        CONSUMED BY: data/gid_dataset.py (the PyTorch Dataset)
                     → feeds Phase 1 / Phase 2 / Phase 3 training
```

---

## A.4 `config.py` — the control panel

**Why it exists.** The pipeline has ~30 tunable numbers (dataset location, depth range,
model confidence thresholds, IoU thresholds, test fraction, …). Collecting them into one
place with sensible defaults, loadable from a YAML file, keeps the other six files clean
and the whole run reproducible.

**Where it is used.** Every other file imports from here. `run_generate.py` builds a
`DataEngineConfig` from the YAML and passes it around.

**Input / Output.** In: a `.yaml` path. Out: a `DataEngineConfig` object (settings only,
nothing written to disk).

**Structure.** It is mostly **dataclasses** (a dataclass is a class that neatly holds
named fields):

- `SequenceLayout` — folder names inside each video (`rgb_dir="left_rgb"`,
  `depth_png_dir="left_filled"`, `depth_npy_dir="left_filled_np"`) and file extensions.
- `DepthConfig` — how to turn raw depth into meters: `unit="auto"`, `min_depth_m=0.01`,
  `max_depth_m=10.0`, `prefer_npy=True`.
- `SAM3Config` — segmentation settings: `object_prompts=("person",)`,
  `ground_prompts=("floor","ground")`, `min_object_score=0.5` (ignore detections below
  50% confidence), `min_mask_area_frac=1e-4` (drop speck masks).
- `IdentityConfig` — ID-cleanup thresholds: `cross_concept_dedup_iou=0.75`,
  `reid_iou=0.5`, `max_gap=10`, `min_track_length=5`.
- `OutputConfig` — where/how to write results (`out_root="gid_custom"`, mask folder names,
  `annotation_file="annotations.json"`).
- `SplitConfig` — the split: `test_fraction=0.20`, `seed=2026`, `stratify_by_batch=True`.
- `DataEngineConfig` — the master box holding all of the above.

**Key methods.**

```python
def __post_init__(self):
    if not self.category_ids:                                  # auto-number the prompts
        self.category_ids = {p: i for i, p in enumerate(self.sam3.object_prompts)}
```
`enumerate(("person",))` → `{"person": 0}`. This is the integer class id the model predicts.

```python
@classmethod
def from_dict(cls, d):
    def sub(klass, key):
        raw = dict(d.get(key, {}))               # use YAML values, else {} → all defaults
        for k, v in raw.items():
            if isinstance(v, list): raw[k] = tuple(v)   # YAML gives lists; dataclasses want tuples
        return klass(**raw)
    ...
```
`d.get(key, {})` is why a minimal YAML still works — missing keys fall back to defaults.

---

## A.5 `discover.py` — the scout

**Why it exists.** Before annotating, we must find the raw data and pair each depth file
with its photo. Filenames are not always perfectly aligned, so this needs care.

**Where it is used.** Called by `run_generate.py` via `discover_dataset(cfg)`. Its records
are consumed by `annotate.py` and `depth_io.py`.

**Input / Output.** In: dataset root + folder layout. Out: a list of `SequenceRecord`
objects, each holding an ordered list of `FrameRecord`s (RGB path + matched depth paths).

**Functions.**

- `natural_key(path)` — human-like sorting so `frame_2` comes before `frame_10` (plain
  text sort would put `frame_10` first). Correct video order is essential for tracking.
- `FrameRecord` — one frame: name, rgb path, depth paths, and `has_depth` (does it have at
  least one depth file?).
- `SequenceRecord` — one video: batch, name, root, frames; `seq_id` = `"Batch 1/20260105_012545"`.
- `_list_files(dir, exts)` — lists files with the wanted extensions, naturally sorted.
- `_pair_by_stem_or_index(rgb, dep, ...)` — the matchmaker. For each RGB file it tries to
  find the depth file with a matching name (tries `frame_0001`, `frame_0001_depth`, and
  the stem with `_rgb` stripped). If name-matching mostly fails, it falls back to
  **positional pairing** (1st rgb ↔ 1st depth) with a warning.
- `discover_sequence(...)` — processes one video folder, builds `FrameRecord`s, and **drops
  any frame with no depth** (it cannot be a training target).
- `discover_dataset(cfg)` — walks all `Batch*` folders, calls `discover_sequence` on each
  timestamp folder. Raises `FileNotFoundError` if the root is missing.

**Example.** A folder with 100 jpgs + 100 npys + 100 pngs (names matching) → one
`SequenceRecord` with `len == 100`, each frame carrying its `.jpg`, `.npy`, `.png` paths.

---

## A.6 `depth_io.py` — the depth translator

**Why it exists.** The camera stores depth raw, and **not always in meters**. Some sensors
store millimeters as 16-bit integers (so `3500` means 3.5 m). If we misread the unit, a
person 3.5 m away would look 3500 m away and every label would be wrong.

**Where it is used.** `annotate.py` calls `detect_unit_scale(...)` once per sequence and
`load_depth_meters(...)` per frame.

**Input / Output.** In: a `FrameRecord` + `DepthConfig`. Out: a float32 `(H, W)` depth map
in meters, with invalid pixels set to exactly `0`.

**Functions.**

- `_read_raw(...)` — opens the depth file. For `.npy/.npz` uses `np.load`; for `.png` uses
  `cv2.imread(..., IMREAD_UNCHANGED)` which preserves the 16-bit values.
- `detect_unit_scale(...)` — samples ~5 frames, takes the median positive depth, and
  reasons: real indoor scenes are 0.01–10 m, so a median of ~4500 must be millimeters →
  scale `1e-3`. The unit can also be pinned in the config.
- `load_depth_meters(...)` — reads raw, multiplies by the scale, and sets **invalid**
  pixels to `0` (non-finite, below `min_depth_m`, or above `max_depth_m`).

```python
med = np.median(vals)
if   med > 800.0: scale = 1e-3     # clearly millimeters
elif med >  80.0: scale = 1e-3     # ambiguous mm/cm zone; mm sensors are common
else:             scale = 1.0      # already meters
```

**Example.** Raw 16-bit pixel `3500` → scale `1e-3` → `3.5 m`. A raw `0` (sensor gap) →
stays `0.0`. A raw `50000` (→ 50 m, beyond the 10 m max) → `0.0`. The `0` value is the
universal "no ground truth here" flag, and the model's losses ignore it — the convention
matches end-to-end.

---

## A.7 `sam3_engine.py` — the eyes (segmentation + tracking)

**Why it exists.** Something must look at a video and say "here are the person-shaped
pixels in every frame, and here is the same person tracked over time." That is what
**SAM3** does: give it a text prompt like `"person"`, and it finds and follows that
concept through the whole video, returning a mask per object per frame plus a temporary ID.
The original paper used two tools (SAM for masks + DEVA for tracking); we replace both with
SAM3, which does masks and tracking in one pass.

**Where it is used.** `run_generate.py` builds one segmenter; `annotate.py` calls
`segmenter.track_concept(rgb_paths, hw, prompt)` per prompt.

**Input / Output.** In: the RGB frame paths, image size, one text prompt. Out: a normalized
structure `FramesOut = {frame_index: {local_obj_id: MaskObs(mask, score)}}`.

**Components.**

- `MaskObs` — one detected object on one frame: boolean mask + confidence score.
- `_to_bool_mask(...)` — normalizes any backend format (torch/numpy, extra dimensions,
  logits/probabilities/booleans, low resolution) into a clean boolean mask at full image
  size.
- `_area_ok(...)` — "is this mask big enough to be real, not a speck?"
- `VideoSegmenter` — the base interface (`track_concept`, `close`); the rest of the
  pipeline codes against this, not a specific backend.
- `NativeSAM3Segmenter` — the official SAM3 (GitHub) session API: stage frames into a JPEG
  folder if needed, start a session, inject the text prompt at frame 0, propagate through
  the video, and keep objects that pass the score floor and area check.
- `HFSAM3Segmenter` — the same behavior via the HuggingFace `transformers` SAM3 API.
- `MockSegmenter` — a synthetic segmenter (two moving discs, one occluding the other, plus
  a floor) so the whole pipeline — including occlusion logic — can run in CI with no GPU
  and no weights.
- `build_segmenter(cfg)` — the factory: native / hf / mock.

**A crucial design fact.** One SAM3 session tracks exactly one concept. So `"person"` is
one session and `"floor"` is another. IDs from different concepts are only locally unique —
which is why `identity.py` must merge and renumber them.

**Example.** `track_concept(paths, (720,1280), "person")` on a 100-frame clip with 3 people
returns, per frame, `{1: mask_A, 2: mask_B, 3: mask_C}`. Those `1,2,3` are local IDs for
this "person" session only.

---

## A.8 `identity.py` — the ID cleaner

**Why it exists.** SAM3's raw tracking is imperfect and has a structural quirk: each
concept runs in its own session (IDs only locally unique), and a single person's track can
**break** — SAM3 loses them behind an occlusion, then re-detects them as a "new" person
with a new ID. If we shipped those raw IDs, "person 3" in frame 1 might become "person 7"
in frame 40, and the model would think they are two different people.

**Where it is used.** `annotate.py` calls `build_tracks(per_concept_obj, category_ids, cfg)`.

**Input / Output.** In: the raw per-concept SAM3 output. Out: a clean dict
`{global_track_id: Track}` where each `Track` holds that person's mask on every frame.

**Functions.**

- `Track` — one person's whole life in the video: which frames they appear in and their
  mask each frame.
- `mask_iou(a, b)` — the overlap score (intersection ÷ union).
- `assign_global_ids(...)` — flattens every concept's local IDs into fresh global IDs so
  IDs from different sessions cannot collide.
- `dedup_cross_concept(...)` — if two *different* concepts produced masks overlapping above
  IoU 0.75 on the same frame, they are the same object seen twice; the lower-score one
  loses that frame.
- `repair_identities(...)` — the key step. It finds a track that **ends** and another track
  of the same category that **starts** within `max_gap=10` frames, and checks whether the
  dying track's last mask overlaps the newborn track's first mask (IoU > `reid_iou=0.5`).
  If yes, they are the same person — it merges them. It loops until no more merges happen.
  This is the occlusion-recovery step that preserves temporal identity.
- `filter_short_tracks(...)` — drops tracks shorter than `min_track_length=5` frames (noise).
- `renumber(...)` — compacts surviving IDs to `1..K` (masks are saved as uint16 PNGs where
  the pixel value *is* the track ID, so IDs must stay small and gap-free).
- `build_tracks(...)` — runs all five in order: assign → dedup → repair → filter → renumber.

**The repair loop.**

```python
gap = b.first_frame - a.last_frame
if not (0 < gap <= cfg.max_gap): continue          # b starts after a ends, within 10 frames
iou = mask_iou(a.masks[a.last_frame], b.masks[b.first_frame])
if iou > best_iou: best_g, best_iou = gb, iou       # best_iou starts at reid_iou = 0.5
...
if best_g is not None:
    b = tracks.pop(best_g)
    a.masks.update(b.masks); a.scores.update(b.scores)   # absorb the newborn into the older track
```
In words: *"Person A vanished at frame 30. Did a 'new' person B appear at frame 33 in
almost the same place? If so, B is really A — glue them together."*

**Example.** Track A = frames 0–30, track B = frames 33–90 (B born where A died because
someone walked in front at frame 31). Gap = 3 (≤10), boundary IoU = 0.7 (>0.5) → merge into
one track spanning 0–90. A stray 2-frame blip is filtered out (< 5 frames). Survivors are
renumbered 1, 2, 3, …

---

## A.9 `annotate.py` — the writer (produces the ground truth)

**Why it exists.** This is where all ingredients converge into the final per-frame ground
truth: the per-pixel identity image, the ground mask, and the JSON listing every person's
box, area, and depth.

**Where it is used.** `run_generate.py` calls `annotate_sequence(seq, cfg, segmenter,
segmenter)` per video. Its outputs are read by `data/gid_dataset.py` during training.

**Output (per video, under `gid_custom/Batch x/<timestamp>/`).**

- `object_masks/frame_XXXX.png` — a **uint16** image where each pixel's value is the
  **track ID** of the person there (`0` = background). Instance masks + identities in one
  file.
- `ground_masks/frame_XXXX.png` — a **uint8** black/white image, white = floor.
- `annotations.json` — per frame, a list of `instances`, each with `track_id`, `category`,
  `bbox_xyxy`, `area`, `depth_layer_m`, `depth_valid_px`, `score`.

**Functions.**

- `_bbox_from_mask(mask)` — the tight rectangle `[x_min, y_min, x_max+1, y_max+1]` around
  the True pixels.
- `_depth_layer(mask, depth_m)` — the person's depth layer: takes all depth values inside
  their mask, keeps only the valid ones (`> 0`), returns their mean plus the valid count.
- `_flatten_id_map(...)` — the occlusion-resolution step. The saved mask gives each pixel to
  exactly one person, but masks can overlap. Where two people overlap a pixel, it assigns
  the pixel to the one with the **smaller depth** (nearer to the camera = the occluder,
  which is what you actually see) using a running "closest depth so far" buffer (a z-buffer).
- `_ground_union(...)` — merges all floor/ground masks on a frame into one boolean mask.
- `annotate_sequence(...)` — the orchestrator: (1) read image size from the first frame;
  (2) run SAM3 for each object and ground prompt; (3) `build_tracks(...)` for clean
  identities; (4) detect the depth scale; (5) per frame — load depth in meters, compute
  per-track bbox + depth layer + area, build the flattened id-map, build the ground union
  and let objects win over ground, write the two PNGs and record the manifest; (6) write
  `annotations.json`.

**Two subtle lines.**

```python
layer_by_gid[gid] = layer if layer > 0 else np.inf     # a person with no valid depth never
                                                        # wrongly "wins" an overlapping pixel
...
ground &= id_map == 0                                    # a pixel that belongs to a person is NOT floor
```

Inside `_flatten_id_map`:
```python
win = m & (d < depth_buf)     # pixels in this mask nearer than anything seen so far
id_map[win] = gid             # claim them for this person
depth_buf[win] = d            # remember the new nearest depth
```

A careful detail: each person's depth layer is computed from their **full mask before
flattening**, so a partly-hidden person's depth is not biased by losing pixels to the
occluder. Only the saved id-map is flattened.

**Example.** Two people overlap; person 1 at 5.0 m, person 2 at 3.5 m. In the overlap
region the `annotations.json` keeps both honest depth layers (5.0 and 3.5), while the saved
`object_masks` image writes the overlap pixels as **2** (the nearer person, whom the camera
actually sees).

---

## A.10 `run_generate.py` — the conductor (the command you run)

**Why it exists.** One entry point that runs the whole pipeline: discover all videos, build
SAM3 once, annotate every sequence, split into train/test, compute dataset statistics, and
write the top-level index files.

**How it is run.**
```bash
python -m instancedepth.data_engine.run_generate --config instancedepth/configs/gid_custom.yaml
```

**Functions.**

- `split_sequences(...)` — decides which whole **videos** go to test (20%). Done at the
  **video level** (never split frames of one video across train/test — that would leak).
  With `stratify_by_batch=True`, it takes 20% from each batch, so every recording condition
  appears in both splits. Uses a fixed `seed`, so the split is reproducible.
- `compute_statistics(...)` — object counts per 2 m depth bucket, per-video stats (frames,
  tracks, avg objects/frame), and totals.
- `generate(...)` — the orchestration: discover → build segmenter once → loop
  `annotate_sequence` over every video (in a `try/finally` that always closes the segmenter)
  → split → write `train.txt`/`test.txt` → compute stats → write `meta.json`.
- `main()` — CLI: `--config`, `--limit N` (annotate first N sequences for a dry run), `-v`.

```python
segmenter = build_segmenter(cfg.sam3)                 # built ONCE (expensive; loads onto GPU)
try:
    for seq in sequences:
        manifests.append(annotate_sequence(seq, cfg, segmenter, segmenter))
finally:
    segmenter.close()                                 # always freed, even on error
```

**Example.** 306 sequences across the batches → 245 train / 61 test (≈20%, drawn evenly
from every batch). `meta.json` reports object counts per depth bucket, total frames, average
objects per frame, etc.

## A.11 Two consequences of this pipeline that shape the whole project

1. **GT masks are modal and disjoint.** The id-map gives each pixel to exactly one instance,
   so two occluding people have ~0 *mask* IoU. Occlusion is therefore detected by **bounding
   boxes**, not mask overlap.
2. **No GT behind occluders.** A single sensor only sees front surfaces, so Phase 3's object
   loss can only be supervised on **visible** pixels.

---

# PART B — PHASE 1: HOLISTIC DEPTH INITIALIZATION

## B.0 The big picture

Phase 1 takes **one color photo** and predicts, **for every pixel, how far it is from the
camera in real meters** — a full depth map. It also produces rich internal **feature maps**
that Phase 2 and Phase 3 reuse. "Holistic" = it looks at the whole scene at once (not
per-person yet). "Initialization" = it produces the first, scene-wide depth that later
phases refine.

**Why it exists:** Phase 2 and Phase 3 do not work on raw pixels — they work on Phase 1's
feature maps (a compressed, meaning-rich version of the image that already understands 3D
structure) and its initial depth. Phase 1 is the foundation the other two are built on.

Files: `instancedepth/models/backbone/dinov2_wrapper.py`, `instancedepth/models/hdi/*`,
`instancedepth/losses/hdi_losses.py`, config in `instancedepth/configs/config.py`.

## B.1 Shared vocabulary

| Term | Meaning |
|---|---|
| **Monocular depth** | Estimating distance from a single image. Hard, because a photo is flat — the model must learn cues (size, perspective, blur). |
| **Metric vs relative depth** | Metric = real meters. Relative = only ordering with unknown scale. We do metric, because the calibrated sensor gives true meters. |
| **Feature / feature map** | Not the raw image, but learned "meaning maps," shape `(Channels, height, width)`. Each channel highlights some pattern (edges, textures, receding surfaces). |
| **Backbone** | The big pretrained network that turns the image into feature maps. Ours is **DINOv2**. |
| **DINOv2** | A powerful image-understanding network (Meta), pretrained on huge unlabeled data; it already "gets" objects and 3D structure. |
| **ViT / patch / token** | DINOv2 is a Vision Transformer: it chops the image into 14×14 **patches**, turns each into a vector (**token**), and lets tokens share information via **attention**. |
| **Attention** | Each patch looks at all other patches and decides which are relevant — how transformers share information across the image. |
| **Decoder** | Takes the backbone's compact features and gradually enlarges them toward image size while specializing them for depth. |
| **Upsample / interpolate** | Resize a small map to a bigger one, filling in between pixels smoothly (bilinear). |
| **Depth bins + ordinal regression** | Split the 0–10 m range into bins (5 bins, 2 m each) and answer ordinal yes/no questions ("farther than 2 m? than 4 m? …"). Easier to learn than a raw regression. |
| **Iterative refinement** | Predict a rough seed depth, then apply three coarse-to-fine corrections. |
| **Deep supervision** | During training, grade the intermediate rounds too, not only the final answer. |

## B.2 The forward pass

```
RGB image (B, 3, 728, 1288)          ← 728,1288 divisible by 14 (DINOv2 patch)
        │
        ▼   dinov2_wrapper.py
DINOv2 ViT-L/14, tap 3 blocks (23, 14, 5)     ← deep, middle, shallow
        │   → 3 feature maps, each (B, 1024, 52, 92)   [52=728/14, 92=1288/14]
        ▼   depth_range_decoder.py
Depth Range Feature Decoder (3 coarse-to-fine levels)
        │   → F_0 (1/8), F_1 (1/4), F_2 (1/2) feature maps
        ▼   iterative_refinement.py  (+ bin_heads.py)
Eq. 1–4 iterative bin refinement
        │   seed D_0 → +correction → D_1 → +correction → D_2 → +correction → D_3
        ▼   model.py (final bilinear upsample)
depth_final (B, 1, 728, 1288)  metric meters
seg_final   (per-bin logits)
feat_final  (F_2 features)     ← handed to Phase 2/3
        │
        ▼   hdi_losses.py (training only)
compare depth_final (+ D_1,D_2 + bins) against GT depth → loss → backprop
```

---

## B.3 `dinov2_wrapper.py` — the backbone (the model's eyes)

**Why it exists.** To read a photo, the model first converts it into features. Training such
a converter from scratch needs millions of images; instead we reuse **DINOv2**, already
trained on huge data. This file wraps DINOv2 so Phase 1 gets exactly the three feature maps
it needs, and loads DINOv2's weights safely (prefer local files, never silently download,
never proceed with half-loaded weights).

**Input / Output.** In: a normalized RGB tensor `(B, 3, H, W)`, H and W divisible by 14.
Out: a list of 3 feature maps, each `(B, 1024, H/14, W/14)`.

**Which blocks we tap, and why.** DINOv2-Large has 24 transformer blocks. We tap three
depths — blocks **23 (deep), 14 (middle), 5 (shallow)** — because information becomes more
abstract with depth:

- **Shallow blocks (block 5)** carry fine, low-level detail: edges, textures, local
  patterns. "How the image is drawn."
- **Deep blocks (block 23)** carry coarse, high-level meaning: object identity, overall
  scene structure. "What the image means."

Note: DINOv2 is a ViT, so **all three feature maps are the same spatial size (52×92)**; they
differ in *semantic depth*, not resolution. The actual 1/8–1/4–1/2 multi-scale pyramid is
created afterward, by the decoder resizing each map.

**Weight-loading machinery.**

- `_SIZE_CONFIGS` / `build_dinov2_config(name)` — hardcodes each DINOv2 size's architecture,
  so building the model needs no internet. ViT-L = 1024 wide, 24 layers.
- `_rename_original_dinov2_state_dict(...)` — renames the original Meta checkpoint keys into
  HuggingFace format so the weights fit.
- `_convert_dav2_encoder_state_dict(...)` — extracts just the DINOv2 encoder from a
  Depth-Anything-V2 checkpoint (DINOv2 + a depth head bundled together) and drops the head.
  This is how the DAv2 variant loads a depth-fine-tuned encoder.
- `load_dinov2_weights(...)` — the fail-loud loader: prefers the local checkpoint, only
  touches the network if explicitly allowed, and raises if the weights do not cleanly match
  (a few harmless keys, used only in a pretraining mode we never run, are allowed to be
  missing).

**Forward.**
```python
h, w = H // 14, W // 14                          # 728//14=52, 1288//14=92
out = self.model(pixel_values, output_hidden_states=True)
for layer_idx in self.cfg.hook_layers:           # (23, 14, 5)
    hs = out.hidden_states[layer_idx + 1]         # block i's OUTPUT is hidden_states[i+1]
    patch_tokens = hs[:, n_extra:, :]             # drop the CLS (+register) tokens
    spatial = patch_tokens.transpose(1, 2).reshape(B, 1024, h, w)   # flat tokens → 2-D map
    features.append(spatial)
```

**Example.** Input `(4, 3, 728, 1288)` → DINOv2 chops into `52×92 = 4784` patches → outputs
3 maps `(4, 1024, 52, 92)` from blocks 23, 14, 5.

---

## B.4 `depth_range_decoder.py` — the Depth Range Feature Decoder

**Why it exists.** DINOv2's features are small (52×92) and generic. Before we can predict a
sharp, full-size depth map, we need to enlarge them toward image size and specialize them
for depth at several scales. This file (the paper's Fig. 5) builds three coarse-to-fine
feature maps `F_0, F_1, F_2` at 1/8, 1/4, 1/2 of the image.

**The design: three "workers," coarse to fine.** Think of three artists at three zoom
levels. Each finer worker builds on the previous (coarser) worker's draft:

| Worker | Resolution | Patch size | Backbone map it receives |
|---|---|---|---|
| **Worker 0** | 1/8 (coarsest) | **4×4** | **block 23** (deep, semantic) |
| **Worker 1** | 1/4 | **8×8** | block 14 (middle) |
| **Worker 2** | 1/2 (finest) | **16×16** | **block 5** (shallow, detail) |

Two design points here:

- **Which backbone map feeds which level.** The deep, semantic features (block 23) drive the
  **coarse** level, where the overall scene depth structure is set; the shallow, detailed
  features (block 5) drive the **fine** level, where sharp edges are recovered. (This is the
  DPT convention: deep→coarse, shallow→fine.)
- **Patch size grows with resolution.** A big map (1/2) is cut into big 16×16 patches; a
  small map (1/8) into small 4×4 patches — so **every level ends up with the same number of
  tokens** for the attention step:
  ```
  1/8 map  (H/8 × W/8),  patch 4×4   → (H/32 × W/32) tokens
  1/4 map  (H/4 × W/4),  patch 8×8   → (H/32 × W/32) tokens
  1/2 map  (H/2 × W/2),  patch 16×16 → (H/32 × W/32) tokens
  ```
  Equal token counts keep the attention cost balanced across levels.

**What one worker (`DecoderLevel`) does.**
1. `project_in` — a `Conv2d` that squeezes 1024 channels down to 256.
2. `F.interpolate` — resize the map to this level's target resolution.
3. `_pad_to_multiple` — pad so the next step tiles evenly (explained below).
4. `patchify` — a strided conv that cuts the map into patch-sized tiles (tokens).
5. transformer blocks — **patch attention**: each tile looks at the others and shares
   information, so distant parts of the same object stay consistent in depth.
6. `unpatchify` — reassemble the tiles into a per-pixel map.
7. crop off the padding from step 3.

**`_pad_to_multiple(x, k)` — why padding is needed.** The patchify step is a conv with
stride `k` (4/8/16); for it to tile cleanly, the map's height/width must be exact multiples
of `k`. A level resolution like `91 × 161` is not a multiple of 4/16, so we pad just enough
zero rows/columns on the bottom and right to reach the next multiple, remember how much we
added, and crop it back afterward.
```python
pad_h = (k - h % k) % k        # rows to add; the outer % k gives 0 when already a multiple
pad_w = (k - w % k) % k
x = F.pad(x, (0, pad_w, 0, pad_h))     # pad right and bottom with zeros
```

**Coarse-to-fine fusion (the heart of the decoder).**
```python
for level_module, feat, target_hw in zip(self.levels, backbone_features, self.target_hws):
    level_out = level_module(feat)                          # this worker's own drawing
    if prev is not None:
        prev_up = F.interpolate(prev, size=target_hw, ...)  # enlarge the coarser level's output
        level_out = level_out + prev_up                     # ADD it in (top-down fusion)
    prev = level_out
```
Because the three backbone maps are zipped one-to-one with the three levels, **three maps in
gives three maps out** — never nine. Each finer level also pulls in the previous level's
output, but by *adding* it (merging two maps into one), so the count stays three:
`F_0, F_1, F_2`. `F_2` (the finest) is the map handed to Phase 2 and Phase 3.

**Example.** Image `728 × 1288` → backbone maps `(1024, 52, 92)` →
`F_0 (256, 91, 161)`, `F_1 (256, 182, 322)`, `F_2 (256, 364, 644)`.

---

## B.5 `bin_heads.py` — the three prediction heads

**Why it exists.** The refinement step needs three small "output networks" (heads) to turn
feature maps into numbers: a seed depth, a per-bin confidence, and per-bin ordinal scores.

- `_small_conv_head(in, out)` — a reusable little network: `Conv3×3 → GroupNorm → ReLU →
  Conv1×1`. Every head is built from this.
- `InitialDepthHead` — makes the seed depth `D_0` from the coarsest features, then applies
  **Softplus** to force it positive (you cannot be −2 m away).
- `ConfidenceHead` (the paper's `Φ_d`) — takes features **and the current depth estimate**,
  outputs a confidence per bin via sigmoid ("how sure am I that this pixel belongs beyond
  each depth threshold?"). Feeding in the current depth is what makes each round a
  *correction*.
- `OrdinalBinHead` — outputs 5 **ordinal** scores per pixel using **independent per-bin
  sigmoids, not a softmax**. It answers 5 independent yes/no questions ("farther than 2 m?
  than 4 m? …"), which is what the equations need. It returns raw logits; the sigmoid is
  applied later where a probability is needed (more numerically stable for training).

---

## B.6 `iterative_refinement.py` — the Eq. 1–4 core

**Why it exists.** This is the mathematical heart of Phase 1. Instead of predicting depth in
one shot, it predicts a rough seed and corrects it three times, each at a higher resolution.

**Input / Output.** In: the three decoder feature maps. Out: a `RefinementTrace` holding
every depth `D_0..D_3` and every bin output `S_0, S_1, S_2`. `D_3` is the final depth
(before the last upsample); the intermediate `D_1, D_2` and all `S_i` are kept for deep
supervision during training.

**The recurrence.**
```python
d_cur = self.initial_depth_head(features.levels[0])   # D_0: rough seed
for i in range(3):                                     # 3 correction rounds
    f_i = features.levels[i]
    d_upsampled = F.interpolate(d_cur, size=f_i.shape[-2:], ...)  # bring depth up to this level's size
    c_i = self.confidence_heads[i](f_i, d_upsampled)   # Eq.1: per-bin confidence (5 numbers/pixel)
    s_i = self.bin_heads[i](f_i).sigmoid()             # per-bin ordinal scores (5 numbers/pixel)
    r_i = (c_i * s_i).sum(dim=1, keepdim=True)         # Eq.2: sum over the 5 bins → 1 number/pixel
    e_i = 2.0 * (r_i - 1.0) * (max_depth / rd)         # Eq.3: turn r_i into a metric correction
    d_cur = d_upsampled + e_i                          # Eq.4: corrected depth D_{i+1}
```

**The equations behind the code (paper Eq. 1–4).** The range 0–`MAX_d` is split into `rd`
bins of width `MAX_d/rd` (here 10 m ÷ 5 = 2 m). At level `i`, for every pixel:

- **Eq. 1 — per-bin confidence:** `C_i = σ(Φ_d(F_i, D_i))`, where `F_i` is the level's
  feature map, `D_i` the current depth, `Φ_d` a small conv network, and `σ` the sigmoid that
  squashes each of the `rd` outputs into [0, 1]. Each entry `C_i,b` reads as "how much do I
  trust bin `b` at this pixel."
- **Eq. 2 — relative error score:** `R_i = Σ_{b=0}^{rd-1} C_i,b · S_i,b`, the
  confidence-weighted sum over bins of the ordinal bin scores `S_i,b ∈ [0, 1]`. Collapsing the
  `rd` bins into one scalar produces a single "how far off, and which way" signal.
- **Eq. 3 — metric correction:** `E_i = 2·(R_i − 1)·(MAX_d/rd)`, which turns the unitless
  `R_i` into a signed correction in **meters**. It is centered so that `R_i = 1 ⇒ E_i = 0`
  (leave the depth alone); the `MAX_d/rd` factor scales one unit of `R_i` to one bin-width, so
  `R_i` above 1 pushes the pixel farther and below 1 pulls it nearer.
- **Eq. 4 — refined depth:** `D_{i+1} = D_i + E_i`, which becomes the input to the next, finer
  level.

The seed is `D_0 = softplus(head(F_0))`, where `softplus(x) = log(1 + e^x)` is a smooth,
always-positive function guaranteeing a physically valid starting depth.

**What each variable is (per pixel).**

| Name | Meaning | Count |
|---|---|---|
| `c_i` (confidence) | For each of the 5 bins: "how sure am I about this bin here?" (0–1). Uses features **+ current depth**. | 5 numbers |
| `s_i` (bin scores) | For each of the 5 bins: the ordinal "beyond this threshold?" score (0–1). | 5 numbers |
| `r_i` (error score) | `c_i × s_i` summed over the 5 bins. A dial centered at 1. | 1 number |
| `e_i` (correction, meters) | `e_i = 2 × (r_i − 1) × bin_width`, with `bin_width = 10/5 = 2 m`. | 1 number |
| `d_cur` (current depth) | `d_cur = (upsampled old depth) + e_i`. | 1 number |

The dial intuition: `r_i = 1` → no change; `r_i > 1` → push farther; `r_i < 1` → pull nearer.

**Worked example — one pixel whose true depth is 5.0 m; seed `D_0 = 6.0 m`.**

Round 0 (`d_upsampled = 6.0`): the heads output
```
c_0 = [0.8, 0.6, 0.4, 0.2, 0.1]
s_0 = [0.5, 0.5, 0.2, 0.1, 0.0]
c_0 × s_0 = [0.40, 0.30, 0.08, 0.02, 0.00]  →  r_0 = 0.80
e_0 = 2 × (0.80 − 1) × 2 = −0.80 m
d_cur = 6.0 + (−0.80) = 5.20 m           (this is D_1)
```
Round 1 — upsample `D_1` to the next-finer grid, then correct:
```
r_1 = 0.95  →  e_1 = 2 × (0.95 − 1) × 2 = −0.20  →  d_cur = 5.20 − 0.20 = 5.00 m   (D_2)
```
Round 2 — upsample `D_2` to the finest grid, then a tiny final correction:
```
r_2 = 1.00  →  e_2 = 0  →  d_cur = 5.00 m   (D_3, final)
```
Journey: `6.0 → 5.2 → 5.0 → 5.0` (matches the true 5.0 m).

**The upsampling step.** Each round works at a finer, bigger grid than the last. Before a
round can add its correction, the depth from the previous round must be stretched up to the
finer grid (bilinear = smooth averaging of neighbors):
```
before (small grid):   [5.2,     5.6,     6.0]
after  (bigger grid):  [5.2, 5.4, 5.6, 5.8, 6.0]      ← in-between values filled by averaging
```

---

## B.7 `model.py` and `output.py`

**`model.py` (`HolisticDepthModel`)** composes backbone → decoder → refinement, does the
**final bilinear upsample** of `D_3` to full input resolution, and packages everything into
`HolisticDepthOutput`. It exports `F_2` as `feat_final` (for Phase 2/3) and keeps
`D_1, D_2` and the bin outputs for the loss. (An optional temporal aligner exists but is
disabled in the per-frame configuration.)

```python
depth_final = F.interpolate(trace.final_depth, size=(H, W), mode="bilinear", ...)
return HolisticDepthOutput(depth_final=..., seg_final=..., feat_final=decoder_feats.finest,
                           depth_levels=trace.depths[1:3],   # D_1, D_2 for deep supervision
                           seg_levels=trace.bins, ...)
```

**`output.py` (`HolisticDepthOutput`)** is the versioned contract Phase 2/3 consume. Fields:
`depth_final`, `seg_final` (per-bin logits), `feat_final` (F_2), `depth_levels`,
`seg_levels`, plus helpers `feat_stride()` (how much smaller the features are than the
image, for aligning boxes to features) and `seg_confidence()` (sigmoid of the logits). A
`contract_version` is bumped whenever a field's shape or meaning changes, so a downstream
phase reading a stale output fails loudly.

---

## B.8 `hdi_losses.py` — how Phase 1 learns

A model only learns if we can measure how wrong it is. This file computes Phase 1's loss by
comparing the predicted depth (and bins) against the sensor's ground-truth depth.

### The main depth term — `SigLogLoss`

A **scale-invariant log** error (Eigen, Puhrsch & Fergus, NeurIPS 2014; the exact form
Depth-Anything-V2 uses). Let `d = log(D) − log(D*)` be the per-pixel error in **log-space**,
where `D` is the prediction and `D*` the ground truth, taken only over valid pixels
(`D* > 0`). The loss is

```
SigLog = sqrt(  mean(d²)  −  λ · mean(d)²  ),   λ = 0.5
```

each part:
- `d = log(D) − log(D*)` — working in log-space makes the error *relative*: being off by
  0.5 m matters far more at 2 m than at 9 m, and taking logs turns that ratio into a plain
  difference, so near and far pixels are weighted comparably.
- `mean(d²)` — the average squared log-error; penalizes the overall magnitude of the errors.
- `λ · mean(d)²` — `mean(d)` is the *average* error, which captures a **global** over- or
  under-estimate (a constant scale offset multiplies every depth and shifts every `d` by the
  same amount). Subtracting a fraction of its square makes the loss care less about a uniform
  scale bias and more about the relative *structure* of the depth.
- `λ = 0.5` — a compromise: `λ = 1` makes the loss fully scale-invariant (ignoring global
  scale entirely, as MiDaS/DAv2 do); `λ = 0` is plain mean-squared log-error. We keep 0.5
  because the sensor is calibrated (true metric depth), so we deliberately retain *some* scale
  sensitivity rather than discard it.
- `sqrt(·)` returns the value to depth-like units; the `clamp_min(0)` inside it only guards
  against a tiny negative from floating-point rounding (the quantity is non-negative in
  theory).

```python
diff_log = log(target[mask]) - log(pred[mask])                    # d
variance  = diff_log.pow(2).mean() - lam * diff_log.mean().pow(2)  # mean(d²) − λ·mean(d)²
loss      = sqrt(variance.clamp_min(0))
```
The `[mask]` (`gt_depth > 0`) restricts every term to pixels the sensor actually measured —
the same "0 = no ground truth" convention the data engine writes.

### Deep supervision

The refinement produces intermediate depths `D_1, D_2` on the way to `D_final`. Rather than
grade only the final answer, we also grade the intermediate ones:

```
L_depth = SigLog(D_final, D*)  +  Σ_{i∈{1,2}} w_i · SigLog(D_i, D*),   w = (0.5, 0.25)
```

where each `D_i` is compared against the ground truth resized (nearest-neighbour) to that
level's smaller resolution. The weights `w_i < 1` let the early rounds *guide* training —
richer gradients, faster and more stable convergence — without overriding the final output.

### `GradientMatchingLoss` — sharpen the edges

SigLog is a **pointwise** loss: it looks at each pixel on its own and does not care *where*
the error sits. A blurry prediction that smears a depth edge (a person's outline) can score
almost as well as a sharp one. `GradientMatchingLoss` fixes this — it penalizes the
prediction whenever its depth edges do not line up with the ground-truth's edges.

It builds an error map (residual) in log-space, `R = log(D*) − log(D)` (set to 0 where
invalid), and penalizes its **gradient** — how fast the error changes between neighbouring
pixels — averaged over four scales:

```
L_gm = ( Σ_{k=0}^{3}  Σ_{valid pairs} ( |R[i,j] − R[i,j−1]| + |R[i,j] − R[i−1,j]| ) )  /  (number of valid pairs)
```

each part:
- `R = log(D*) − log(D)` — the per-pixel error map, in the same log-space as SigLog's `d`.
- `|R[i,j] − R[i,j−1]|`, `|R[i,j] − R[i−1,j]|` — the horizontal and vertical **gradients** of
  the error: large exactly where the error changes abruptly, i.e. where the prediction has an
  edge the ground truth doesn't (or misses one it does).
- `Σ_{k=0}^{3}` at stride `2^k` (1, 2, 4, 8) — the residual is subsampled at four scales, so a
  stride-1 term catches the sharpest one-pixel edges while coarser strides catch broad
  structure; together they align depth boundaries at every scale.
- "valid pairs only" — a pair straddling a sensor hole (an invalid pixel) is skipped, so a
  missing measurement can never masquerade as a huge fake edge.

Logic: if the prediction's edges match the ground truth's, the errors cancel in the same
places so **R is smooth** (low gradient, low loss); if the prediction smears or misplaces an
edge, **R spikes** there (high gradient, high loss).

Worked example (one row, a person-to-wall edge `gt = [5,5,5,8,8,8]`):
```
Sharp pred [5,5,5,8,8,8]:   R = [0,0,0,0,0,0]         → gradient 0        → loss 0
Blurry pred [5,5,6.5,6.5,8,8]: R = [0,0,-0.26,+0.21,0,0] → gradient sum ≈ 0.94 → penalized
```
The blurry prediction's pixels are individually close to GT, but its edge is the wrong
shape, so the gradient term punishes it — forcing sharp, correctly-placed boundaries.

### Ordinal bin supervision

**`ordinal_bin_targets`** turns each pixel's true depth into the training label for the bin
head. Instead of one-hot (marking the single bin the depth falls into), it uses an ordinal
**thermometer** encoding — bin `b`'s target is 1 if the depth lies beyond that bin's
threshold:

```
y_b = 1[ D* > b · (MAX_d / rd) ]      for b = 0 … rd−1      (thresholds 0, 2, 4, 6, 8 m)
```

So 5.0 m → `[1, 1, 1, 0, 0]` (beyond 0, 2, 4 — yes; beyond 6, 8 — no). The number of 1s is a
coarse depth reading (1 m → one `1`; 9 m → five `1`s). This keeps depth *ordered* — unlike
one-hot, where the bins are unrelated labels — and matches the bin head's independent per-bin
sigmoids.

**`OrdinalBinBCE`** measures how far the bin head's predictions `z_b` (raw logits) are from
that thermometer target, treating each bin at each pixel as its own yes/no question:

```
L_bin = −(1/|M|) · Σ_{pixels} Σ_{b} [ y_b · log σ(z_b)  +  (1 − y_b) · log(1 − σ(z_b)) ]
```

where `σ` is the sigmoid, `σ(z_b)` the predicted probability that the depth is beyond
threshold `b`, and `|M|` the number of valid (pixel, bin) entries. Each term is near 0 when
the prediction is confident and correct and grows large when it is confidently wrong.
`binary_cross_entropy_with_logits` applies the sigmoid internally in a numerically stable,
autocast-safe way (instead of sigmoid-then-log, which can overflow on a saturated output).

Worked example — pixel at 5 m, target `[1,1,1,0,0]`, predictions (after sigmoid)
`p = [0.9, 0.8, 0.6, 0.3, 0.1]`:
```
bin0: −log(0.9)   = 0.105
bin1: −log(0.8)   = 0.223
bin2: −log(0.6)   = 0.511
bin3: −log(1−0.3) = 0.357
bin4: −log(1−0.1) = 0.105
average = 1.301 / 5 ≈ 0.26     ← the loss at this pixel
```

### A pluggable regression registry

The regression term is a registry — `silog | l1 | l2 | berhu` — so an alternative can be
swapped in for an ablation without touching the trainer. `BerHuLoss` (reverse Huber) is L1
below a per-batch adaptive threshold and L2 above it.

### The combined loss

`HDILoss` adds the terms above:

```
L_hdi = SigLog(D_final, D*)  +  Σ_{i∈{1,2}} w_i·SigLog(D_i, D*)          (depth + deep supervision)
        +  λ_ce · Σ_{i=0}^{2} L_bin(S_i, D*)                             (ordinal bin BCE, all 3 levels)
        [ +  λ_gm · L_gm  +  λ_disp · L_disp ]                          (optional; off in the faithful profile)
```

with `w = (0.5, 0.25)` and `λ_ce = 1.0` by default. Every term is masked to valid depth
(`D* > 0`). The bracketed edge-sharpening (`L_gm`) and disparity terms carry weight 0 in the
paper-faithful configuration and are enabled only in the enhanced profile.

## B.9 Inference and training entry points

- **`inference.py` (`HDIInferencer`)** loads a checkpoint once, then `predict(rgb)` per
  frame: it normalizes exactly like the training dataset, runs the model at
  `cfg.data.image_size`, then resizes depth back to the caller's original resolution. Phase
  2/3 consume Phase 1 features in-process this way (caching features for ~50k frames would
  be hundreds of GB).
- **`train_hdi.py`** wires the `GIDInstanceDepthDataset` (the data engine's output) into a
  DataLoader, defines one training step (move batch to GPU → `model(image)` → `HDILoss`),
  and hands everything to the generic Trainer. This is the concrete link between the data
  engine and Phase 1: the annotations we generated are exactly what the dataloader reads.

## B.10 Phase 1 summary

Phase 1 predicts a full metric depth map from a single image and produces the features the
later phases reuse. A DINOv2 backbone gives features from three tapped blocks; the Depth
Range Feature Decoder turns them into three coarse-to-fine feature maps with patch attention
and top-down fusion (deep semantic features drive the coarse level, shallow detail features
drive the fine level, patch sizes scaled to keep token counts balanced); then the Eq. 1–4
iterative bin refinement builds depth as a seed plus three corrections. Training uses a
scale-invariant log loss on true metric ground truth, with deep supervision and an ordinal
bin BCE, optionally sharpened by gradient matching.

---

# PART C — PHASE 2: INSTANCE DEPTH LAYER PREDICTION

## C.0 The big picture

Phase 1 told us how far every pixel is, but not **who is who**. Phase 2 fills that gap.
Given one image, it produces, for each person: a **mask** (which pixels are this person), a
**class** ("person"), and a **depth layer `Dep_i`** (roughly how far the whole person is).
This is the paper's Eq. 5–7.

**Why it exists:** Phase 3 (occlusion refinement) reasons about pairs of people who overlap
("A is in front of B, so A's depth should be smaller"). To do that it needs to know where
each person is and their rough depth — which is exactly what Phase 2 provides.

Files: `instancedepth/models/phase2/*`, config in `instancedepth/configs/phase2_config.py`.

## C.1 Shared vocabulary

| Term | Meaning |
|---|---|
| **Instance segmentation** | Not just "person-ish pixels" but "person #1 here, person #2 there" — a separate mask per object. |
| **Mask2Former** | A well-known instance-segmentation model. We reuse the official COCO-pretrained one. |
| **Swin-L** | The image backbone inside Mask2Former ("Swin Transformer, Large") — Phase 2's own eyes, separate from Phase 1's DINOv2. |
| **Object query** | Think of **100 detectives**. Each is a slot sent into the image to find one object, returning a mask + a class guess + a depth. Most find nothing ("no object"); a few find the real people. |
| **Query embedding** | The 256-number vector each detective carries, summarizing what it found. Mask, class, and depth are all read off this vector. |
| **Bipartite / Hungarian matching** | The model outputs 100 predictions but there are only a few real people, in no fixed order. Matching decides, one-to-one, which prediction corresponds to which real person. |
| **No-object class** | An extra label meaning "this query found nothing." Most queries get it. |
| **Point sampling (PointRend)** | Computing mask error on all pixels × all queries is too expensive; instead sample ~12,544 points, focusing on uncertain (boundary) ones. |
| **Dice loss** | A mask-overlap score (cousin of IoU); good when the object is small vs the whole image. |
| **Smooth-L1** | An error measure like absolute error but gently rounded near zero — robust and stable. Used for the depth-layer regression. |

## C.2 The forward pass

```
RGB image (B, 3, 736, 1280)          ← 736,1280 divisible by 32 (Swin's stages)
        │
        ▼   mask2former_wrapper.py
Mask2Former (Swin-L)  →  for 100 queries:
        │   • class_logits     (B, 100, num_classes+1)
        │   • mask_logits      (B, 100, H, W)
        │   • query_embeddings (B, 100, 256)
        ▼   depth_head.py
DepthLayerHead: small MLP on each query embedding
        │   → depth_layers (B, 100)            ← Dep_i per query
        ▼   model.py  (assemble + upsample masks to full res)
Phase2Output { mask_logits, class_logits, depth_layers, query_embeddings }
        │
        ▼   TRAINING ONLY:
matcher.py  → pair the 100 predictions to the few real people (Hungarian)
criterion.py → loss on matched pairs (class + mask + dice + depth) → backprop
```

We use **100 object queries**. The COCO-pretrained Mask2Former ships with 200; the busiest
scenes in this dataset have far fewer people, so 100 slots are more than enough while
halving the transformer-decoder and matcher query budget. Only the two query-embedding
tables depend on the count; every other weight loads from the checkpoint unchanged.

---

## C.3 `mask2former_wrapper.py` — the segmentation engine

**Why it exists.** We need per-instance masks + classes. Rather than build that from
scratch, we reuse the official COCO-pretrained Mask2Former (Swin-L). This file wraps it and
exposes a clean, stable output: class logits, mask logits, and the **query embeddings**
(which the depth head needs). It loads the checkpoint fail-loud (prefer a local snapshot,
never silently hang on a network download).

**Input / Output.** In: the image. Out:
`RawMask2FormerPrediction(class_logits, mask_logits, query_embeddings)`.

**Key parts.**

- `_resolve_checkpoint_source(...)` — prefer a local directory; if it is missing, raise
  rather than hang; only download if explicitly opted in.
- `Mask2FormerWrapper.__init__` — loads `Mask2FormerForUniversalSegmentation`. Sets
  `num_labels = 1` (person) and `num_queries = 100`, then loads the checkpoint with
  `ignore_mismatched_sizes=True` so the resized class head and the two query-embedding
  tables are reinitialized while everything else (backbone, pixel decoder, transformer
  decoder, mask head) loads from COCO.
- `forward(...)` — runs the model and pulls out the three fields, searching known field
  names for the per-query embedding and **verifying the shape** is `(B, num_queries,
  hidden_dim)` before accepting it (refusing to silently guess if the library changed).

```python
if num_classes is not None: config.num_labels  = num_classes
if num_queries is not None: config.num_queries = num_queries
self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
    source, config=config,
    ignore_mismatched_sizes=(num_classes is not None or num_queries is not None),
    local_files_only=local_files_only)
```

---

## C.4 `depth_head.py` — the depth-layer head (Eq. 5–7's `Dep_i`)

**Why it exists.** Mask2Former gives masks and classes but not depth. The paper's
contribution in this stage is adding a **depth layer** per instance. This small file reads
each query's embedding and outputs one depth number for that instance.

**Input / Output.** In: query embeddings `(B, 100, 256)`. Out: depth per query `(B, 100)`.

```python
class DepthLayerHead(nn.Module):        # Linear(256→512) → ReLU → Linear(512→1) → Softplus
    def forward(self, query_embeddings):
        return self.softplus(self.mlp(query_embeddings)).squeeze(-1)
```
The MLP maps each 256-vector to one number; Softplus forces it positive (same convention as
Phase 1's seed head). This mirrors how Mask2Former reads the class off the same embedding.

**Example.** Query #57 found the left person → its 256-vector → `Dep_57 = 5.0 m`. Query #12
found nothing → its depth is meaningless (that query will be matched to "no object" and its
depth ignored by the loss).

### Companion — `diagnostics.py` (Reading B, a non-learned cross-check)

There are two ways to obtain a per-instance depth layer, and the plan calls them **Reading A**
and **Reading B**. The head above is Reading A — a *learned* MLP that reads `Dep_i` straight
from the query vector. [`diagnostics.py`](../instancedepth/models/phase2/diagnostics.py) is
**Reading B**: a *non-learned* alternative that takes each predicted mask, looks up **Phase 1's
depth map** inside it, and averages — which is "the average depth of the instance" by the
paper's own definition, obtained by pooling rather than learning.

```python
@torch.no_grad()
def mask_pooled_depth(mask_logits, holistic_depth, threshold=0.5):
    masks       = (mask_logits.sigmoid() > threshold).float()       # (B,N,H,W) hard masks
    depth       = holistic_depth.expand(-1, masks.shape[1], -1, -1) # (B,1,H,W) → (B,N,H,W)
    pixel_count = masks.sum(dim=(-2, -1)).clamp_min(1.0)            # (B,N)
    return (masks * depth).sum(dim=(-2, -1)) / pixel_count          # (B,N) mean depth per mask
```

- `masks` — the soft mask logits are turned into hard 0/1 masks by `sigmoid > 0.5`.
- `depth.expand(...)` — the single Phase-1 depth map is broadcast (a view, not a copy) so all
  `N` masks can pool against it.
- `(masks * depth).sum / pixel_count` — a **masked mean**: sum the depth inside each mask,
  divide by its pixel count (`clamp_min(1)` guards empty masks). Unlike the data engine's GT
  depth layer, no zero-filtering is needed here, because Phase 1's `depth_final` is a *dense*
  prediction with no sensor holes.

This is the exact analog of how the ground-truth depth layer was built (mean depth inside the
*GT* mask, in the data engine) — only now using the *predicted* mask and *predicted* depth,
which is what makes it a meaningful check. It runs under `@torch.no_grad()` and produces no
gradients; if the learned head is healthy its predictions should track these pooled values, and
a large persistent gap would flag that the head is not converging to something physically
sensible.

The learned head (Reading A) is preferred over simply pooling because pooling compounds Phase
1's and Phase 2's errors and, more importantly, would couple Phase 2's output to the Phase-1
checkpoint — breaking the independence described in §C.10. `mask_pooled_depth` is a supplied
diagnostic **utility** rather than an active one: it is defined here and referenced by the depth
head's documentation, but is not currently wired into the training or evaluation loop.

---

## C.5 `model.py` and `output.py`

**`model.py` (`Phase2Model`)** composes the wrapper + depth head: run Mask2Former, run the
depth head on the query embeddings, upsample mask logits to full input resolution, and
package everything into `Phase2Output`.

**What resolution the work actually happens at.** Mask2Former does *not* segment at full
resolution. The Swin backbone produces a feature pyramid (1/4, 1/8, 1/16, 1/32 of the input);
the transformer decoder's queries attend to those coarse features, so `class_logits` and the
`query_embeddings` — and therefore the depth layers read off them — carry **no spatial grid at
all** (they are per-query vectors). Each query's mask is a dot product between its 256-vector
and a per-pixel mask-feature map the pixel decoder emits at **1/4 resolution**, so the raw
`mask_logits` are `(B, 100, H/4, W/4)` = `(B, 100, 184, 320)` for a 736×1280 input. Full
resolution appears only at the very end, in a single bilinear upsample:
```python
H, W = pixel_values.shape[-2:]                      # 736, 1280
mask_logits = raw.mask_logits                       # (B, 100, 184, 320)  ← quarter resolution
if mask_logits.shape[-2:] != (H, W):
    mask_logits = F.interpolate(mask_logits, size=(H, W), ...)   # → (B, 100, 736, 1280)
```

**`output.py` (`Phase2Output`)** is the versioned contract Phase 3 reads: `mask_logits`,
`class_logits`, `depth_layers`, `query_embeddings`. Two helpers that Phase 3 uses for
candidate filtering:
- `scores()` → per-query foreground confidence (softmax over classes, drop the no-object
  slot, take the max) — for Phase 3's "category confidence > 0.9" filter.
- `mask_confidence()` → sigmoid of the mask logits — for Phase 3's "mask confidence > 0.8"
  filter.

---

## C.6 `matcher.py` — Hungarian matching (Eq. 5–7)

**Why it exists.** The model outputs 100 predictions, but a frame has only a few real
people, in no fixed order. Before computing a loss we must decide **which prediction is
trying to be which real person** — one-to-one. This file does that with **Hungarian
(bipartite) matching**: it builds a cost table of "how badly does prediction *n* match real
person *g*?" and finds the assignment with the smallest total cost.

**Input / Output.** In: class logits, mask logits, depth preds, and the ground-truth
targets. Out: for each image, a `(row, col)` pairing = (which query ↔ which GT person).

**The cost (paper Eq. 5–7).** For every (query `n`, ground-truth person `g`) pair we build a
cost that is small when the query matches that person well:

```
C[n, g] = λ_cls·cost_cls[n,g] + λ_mask·cost_mask[n,g] + λ_dice·cost_dice[n,g] + λ_dep·cost_dep[n,g]
          (λ_cls = 2)          (λ_mask = 5)             (λ_dice = 5)            (λ_dep = 1)
```

each term:
- **class** `cost_cls[n,g] = −P_n(class of g)` — the negative probability the query assigns to
  person `g`'s true class (`softmax` over classes). Using `−P` rather than a full
  cross-entropy keeps the matching cost cheap and bounded; smaller means the query is more
  confident in the right class.
- **mask (BCE)** `cost_mask` — the binary cross-entropy between the query's mask and person
  `g`'s mask, evaluated at sampled points (see §C.8): pixel-by-pixel agreement.
- **mask (Dice)** `cost_dice = 1 − (2·Σ p·t + 1)/(Σ p + Σ t + 1)`, where `p = σ(mask logit)`
  and `t` is the target mask. The numerator `2·Σ p·t` is twice the overlap; the denominator is
  the two areas — so this is `1 − (soft overlap ratio)`, a smooth cousin of `1 − IoU`. The
  `+1` on top and bottom is Laplace smoothing: it avoids division by zero and keeps the cost
  well-defined for empty masks. Dice complements BCE because it is insensitive to the large
  background-vs-object area imbalance.
- **depth** `cost_dep = smoothL1(Dep_n, Dep*_g)` — the paper's Eq. 5–7 addition: how close the
  query's predicted depth layer is to person `g`'s true depth, making the matching
  *depth-aware* so a query that also gets depth right is preferred.

`smoothL1(x)` (Huber, `β = 1`) is `0.5·x²` for `|x| < 1` and `|x| − 0.5` otherwise —
quadratic near 0 (stable, sensitive) and linear far away (robust to outlier depths).

**The assignment.** `C` is a `100 × G` table, and
`scipy.optimize.linear_sum_assignment(C)` solves the **assignment problem**: pick exactly one
query per person so the *total* cost is minimum (the Hungarian algorithm). The mask terms are
evaluated on `num_points = 12544` uniformly random points, not all pixels — a uniform sample
is an unbiased, cheap estimate of the full mask cost, which is all the matcher needs to *rank*
candidates.

The whole `forward` runs under `@torch.no_grad()`: matching is a discrete decision that
selects *which* prediction supervises *which* target — it produces no gradients itself.

**Example.** 3 real people, 100 queries → a `100 × 3` cost table. Hungarian picks, e.g.,
query 57 ↔ person 0, query 12 ↔ person 1, query 43 ↔ person 2 — the trio that jointly
minimizes total (class + mask + dice + depth) cost. The other 97 queries are unmatched and
will be trained toward "no object" by the criterion.

---

## C.7 `criterion.py` — the training loss

**Why it exists.** Once matching says "query 57 = person 0," this file computes how wrong
each matched prediction is (mask, class, depth) and turns it into a loss. It also teaches all
the unmatched queries to say "no object."

**Matcher vs. criterion.** The matcher only *decides* the pairing (no gradients); the
criterion *uses* that pairing to compute the actual loss the model learns from. Their terms
look alike but differ where it matters: the criterion's class term is a full cross-entropy
(with a no-object class), and its mask points are chosen by *uncertainty*, not uniformly.

**The three terms**, on matched pairs (except the class term, which also supervises the
unmatched queries):

- **class** — cross-entropy over `num_classes + 1` labels (the `+1` is the **no-object**
  class): matched queries are supervised toward their true class, every other query toward
  no-object.
  ```
  L_cls = CE( class_logits, target ),   with class weight w_eos = 0.1 on the no-object label
  ```
  The `eos_coef = 0.1` down-weight is a **class-imbalance** fix: with 100 queries and only a
  few people, ~97 targets are "no-object"; left equal, that one class would dominate the loss
  and the model would learn to predict nothing. Weighting it ×0.1 rebalances the few real
  instances against the many empties.
- **mask** — point-sampled binary cross-entropy plus Dice on each matched (predicted, true)
  mask pair:
  ```
  L_mask = mean over sampled points of  BCE( mask_logit, target )
  L_dice = 1 − (2·Σ p·t + 1) / (Σ p + Σ t + 1),     p = σ(mask_logit),  t = target mask
  ```
  BCE gets each pixel right on average; Dice enforces overall overlap and is robust to the
  object-vs-background imbalance. Points here are chosen by uncertainty sampling (§C.8), so the
  gradient concentrates on the hard boundary.
- **depth** — smooth-L1 between each matched query's predicted depth layer and its GT:
  ```
  L_depth = ( Σ_matched  smoothL1(Dep_n, Dep*_g) ) / num_masks
  ```

Each term is divided by `num_masks` (the number of matched instances in the batch, floored at
1), so the loss magnitude — and thus the gradient scale — does not depend on how many people
happen to be in the batch. The four terms are then combined:

```
L_phase2 = 2·L_cls + 5·L_mask + 5·L_dice + 1·L_depth
```

**Example.** Query 57 (matched to person 0 at true 5.0 m) predicts a mask, class "person",
and `Dep = 5.3 m`: `L_cls` pushes it to confidently say "person," `L_mask`/`L_dice` sharpen
its mask to person 0's shape, and `L_depth = smoothL1(5.3, 5.0)` nudges its depth toward 5.0.
Query 12 (unmatched) is pushed to say "no-object"; its mask and depth are ignored.

---

## C.8 `point_sample.py` — cheap, focused mask loss (PointRend)

**Why not just use every pixel?** A dense mask loss would evaluate BCE + Dice over
`(number of matched masks) × H × W` values, and that cost grows with every instance in the
batch — most of it spent on easy, obviously-inside or obviously-outside pixels. The idea, from
**PointRend** (Kirillov et al., CVPR 2020) and adopted by Mask2Former's official
matcher/criterion, is to supervise the mask on a small fixed set of `K = 12544` (= 112²)
sampled points instead, with almost no loss of accuracy — because a mask is easy in its
interior and only genuinely hard along its **boundary**. We reimplement it against
`torch.nn.functional.grid_sample` so we do not depend on Detectron2.

- `point_sample(input, coords)` — reads the mask value at arbitrary continuous coordinates
  (normalized to [0, 1]) via `grid_sample`, which **bilinearly interpolates** between the four
  surrounding pixels. This is what lets us evaluate a mask at any point, not only on the pixel
  grid.
- `calculate_uncertainty(logits) = −|logit|` — a point is "uncertain" when its logit is near
  0, i.e. the predicted probability `σ(logit)` is near 0.5: the model is undecided
  inside-vs-outside there, which is exactly the mask **boundary**.
- `get_uncertain_point_coords_with_randomness(...)` — the training-time recipe (used by the
  criterion): draw `oversample_ratio × K` (3×) uniform candidate points, score each by
  uncertainty, keep the `importance_sample_ratio` (75%) most uncertain, and fill the remaining
  25% with fresh uniform points. So three-quarters of the supervision lands on the hard
  boundary while one-quarter stays spread out, keeping the interior honest.

**Matcher vs. criterion sampling.** The matcher uses plain *uniform random* points — it only
needs an unbiased, cheap estimate of the mask cost to *rank* candidates. The criterion uses
the *uncertainty-based* points above — that is the actual gradient, so it is worth
concentrating on the boundary where the mask is decided.

---

## C.9 `phase2_config.py` — settings

Notable values: `num_classes = 1` (person), `num_queries = 100`, `image_size = (736, 1280)`
(divisible by 32 for Swin's four patch-merging stages — different from Phase 1's 728×1288/14),
matcher and loss weights `2/5/5/1` (class/mask/dice/depth), point budget `12544`. The
optimizer uses Mask2Former's own fine-tuning recipe (`lr=1e-4`, `backbone_lr_mult=0.1`).

## C.10 A key design decision

Phase 2 carries its **own Swin-L backbone**, independent of Phase 1's DINOv2. This makes the
training freeze strategy coherent: Phase 3 fine-tunes Phase 1's encoder while Phase 2 stays
frozen, so Phase 2's outputs do not drift. A direct consequence is that **Phase 2's output
is invariant to the Phase-1 checkpoint**.

## C.11 Phase 2 summary

Phase 2 predicts per-instance masks, classes, and a depth layer. It uses the official
COCO-pretrained Swin-L Mask2Former as a query-based instance segmenter — 100 object queries,
each producing a mask and a class — plus a small MLP depth head that reads each query's
embedding to predict that instance's average depth `Dep_i` (the paper's Eq. 5–7). Training
does Hungarian bipartite matching between the 100 predictions and the few real people, with a
cost combining class, mask BCE, Dice, and the new depth term; the criterion then applies
those losses on matched pairs and trains the rest toward "no object." Mask supervision uses
PointRend point sampling for efficiency.

---

# PART D — PHASE 3: OCCLUSION-AWARE DEPTH REFINEMENT

## D.0 The big picture

Phase 1 gives dense depth; Phase 2 gives instances and their rough per-person depth. But the
depth is least reliable exactly where two people **overlap** — the occlusion boundary, where
it is ambiguous which surface a pixel belongs to. Phase 3 is the fix: it finds **occluder–
occludee pairs**, reasons about the two people **jointly**, and corrects each instance's depth
so the front person ends up nearer than the back person and the metric depths line up with the
sensor. This is the paper's Eq. 8–12.

**Why it exists:** it is the whole point of the paper — "instance-level video depth in groups
*beyond occlusions*." Phases 1 and 2 set it up; Phase 3 delivers the occlusion-aware result.
It composes the two earlier phases rather than replacing them.

Files: `instancedepth/models/phase3/*`, `instancedepth/losses/phase3_losses.py`, config in
`instancedepth/configs/phase3_config.py`.

### The simplest way to picture it (a worked story)

Two people stand facing the camera: **Alice** is 3 m away and partly in front of **Bob**, who
is 5 m away. Phase 1 already produced a depth map, but right at the edge where Alice overlaps
Bob, the depths are shaky (it's unclear which pixel is whose surface). Phase 3 cleans up exactly
that overlap. In plain steps:

1. **Notice the overlap.** It looks at the people Phase 2 found and sees that Alice's box and
   Bob's box overlap — so they are an *occlusion pair*. It labels Alice the **front** one
   (smaller depth) and Bob the **back** one.
2. **Cut a small stamp of each.** It crops a small fixed-size (28×28) patch around Alice and
   around Bob — from the depth map, from Phase 1's depth features, and from the mask — so it can
   study just those two people up close.
3. **Judge both together.** It hands both stamps to a tiny network (Φ_o) that looks at Alice and
   Bob **at the same time** and outputs a small *nudge* for each — "make Alice a touch nearer,
   Bob a touch farther" — expressed as a **percentage** of their current depth.
4. **Apply the nudge.** It multiplies each person's depth by that nudge (say ×0.98 or ×1.03), so
   it gently *corrects* rather than repaints. Before any training the nudge is exactly ×1.0, so
   nothing changes — Phase 3 starts as a harmless no-op and only learns to nudge from there.
5. **Paste back.** It writes the nudged depths into the full depth map, but **only inside each
   person's silhouette** — every other pixel stays exactly as Phase 1 left it.

It learns from two lessons at once: **(a)** each nudged depth should match the sensor wherever
the person is actually visible, and **(b)** the **gap** between the two people should match the
true gap — so Alice really ends up about 2 m in front of Bob. Everything below is just these
five steps, made precise.

## D.1 Shared vocabulary

| Term | Meaning |
|---|---|
| **Occlusion pair** | Two overlapping instances processed together. Ordered **nearer-first**: member 0 = the **occluder** (smaller predicted depth, "main"), member 1 = the occludee ("guest"). |
| **Candidate filtering** | Keeping only confident instances (high class + mask confidence) worth refining. |
| **ROIAlign** | Crop-and-resample a fixed `Hp×Wp` patch of a feature/depth map inside a box, with sub-pixel bilinear sampling. Lets two maps at different resolutions be compared in one shared box frame. |
| **Relation reasoning (Φ_o)** | A small network that takes *both* instances of a pair at once, so each instance's correction depends on the other — genuine pairwise reasoning, not per-instance. |
| **Residual / multiplicative correction** | The head predicts a *ratio*, not an absolute depth: it nudges the base depth up or down, starting from "no change." |
| **Compositing** | Pasting the per-pair ROI corrections back into the full-resolution dense depth map. |
| **Non-degrading** | A design property: on non-occluded regions the output equals Phase-1 depth exactly, so Phase 3 can only help, never hurt. |

## D.2 The forward pass (`Phase3Model`)

```
RGB at the Phase-2 frame (B, 3, 736, 1280)
        │
        ├── Phase 1 (trainable OR frozen) at 728×1288 → dense depth + F_2 features
        ├── Phase 2 (frozen)              at 736×1280 → masks, class conf, Dep_i
        ▼
candidates.build_pairs        → filter confident instances, find overlapping pairs,
                                keep depth-nearest partner, order nearer-first  → PairSet
        ▼
roi_extract.extract_pair_roi_inputs   → ROIAlign each pair member into a shared 28×28 frame:
                                F_obj (depth features), D_obj (base depth), G_obj (geom priors)
        ▼
relation_head.OcclusionRelationHead Φ_o   → E_obj = σ(Φ_o([F_obj,G_obj]))     (Eq. 8)
                                            D_hat = (2·E_obj − 1)·D_obj + D_obj (Eq. 9)
        ▼
roi_masked_mean               → scalar refined depth per instance (for L_dist / reporting)
composite_refined_depth       → paste corrections back into the dense map (nearest-layer-wins)
        ▼
RefinedDepthOutput { refined_depth, base_depth, pairs, refined_layers, ROI tensors }
```

Two branches, two resolutions. Because DINOv2/14 wants multiples of 14 and Swin/32 wants
multiples of 32 (their only small shared multiple, 224, is too restrictive), Phase 3 keeps
**each branch at its own native resolution** and reconciles them through **normalized [0, 1]
box coordinates** in ROIAlign. The dataset is served in the **Phase-2 frame** (736×1280), where
the GT masks, the frozen Phase-2 masks, and the boxes all align; the model internally resizes
the RGB to 728×1288 for the depth branch.

---

## D.3 `candidates.py` — pick the pairs worth refining

**Why it exists.** Not every instance needs refinement, and refinement is about *pairs*. This
file reads the frozen Phase-2 predictions and produces the set of occlusion pairs (paper
Sec. 4.2.2). It is entirely non-differentiable (thresholds + argmin over frozen predictions),
so it runs under `@torch.no_grad()`.

**Steps.**
1. **Filter candidates** — keep queries with **category confidence > 0.9 AND mask confidence
   > 0.8** (both paper-specified), and a non-empty box. "Mask confidence" is reduced from the
   dense map to a scalar via the standard Mask2Former mask-quality score (mean sigmoid over the
   binarized foreground).
2. **Find overlaps** — among survivors, compute pairwise **bounding-box IoU** and keep pairs
   with **IoU > 0.1**. Box IoU, not mask IoU, is used deliberately: the dataset's GT masks are
   modal and disjoint (the data engine assigns every contested pixel to the nearer instance),
   so Phase-2's predicted masks inherit near-zero overlap for occluded pairs — box IoU still
   detects them. For each instance, its **depth-nearest** overlapping partner is chosen as the
   guest.
3. **Deduplicate + order** — a mutually-overlapping pair would otherwise appear twice, `(A,B)`
   and `(B,A)`, refining the same person twice with disagreeing fields. Each unordered pair is
   kept once, with members ordered **nearer-first** (occluder = channel 0) so Φ_o sees a
   consistent "front vs back" channel semantic.

The result is a `PairSet`: for `P` pairs across the batch, `batch_index (P,)`, `query_idx
(P,2)` (the two query indices), `boxes_norm (P,2,4)` (normalized boxes), and `iou (P,)`.

**Example.** In one frame, filtering keeps 4 confident people; box-IoU finds that persons A&B
overlap (IoU 0.4) and C&D overlap (IoU 0.2). B is nearer than A, C nearer than D → two ordered
pairs `[B, A]` and `[C, D]`. If nobody overlaps, `PairSet` is empty and the refined depth is
just Phase-1 depth (the non-degrading fallback).

---

## D.4 `roi_extract.py` — build Φ_o's inputs with ROIAlign

**Why it exists.** Φ_o needs, for each pair member, a fixed-size patch of depth evidence and
geometry cropped to that instance's box. This file uses `torchvision.ops.roi_align` to cut a
**28×28** patch from each source map inside the (normalized) box, so maps living at *different*
resolutions all land in one shared `Hp×Wp` frame.

**What it produces per pair (member order [main, guest]):**
- **`F_obj` `(P,2,C,28,28)`** — ROIAligned Phase-1 depth **features** (F_2, or F_0/F_1/F_2
  concatenated in the multi-scale variant): the learned depth evidence.
- **`D_obj` `(P,2,1,28,28)`** — ROIAligned Phase-1 **base depth** — this is Eq. 9's `D_obj`,
  the depth being corrected.
- **`G_obj` `(P,2,Gc,28,28)`** — **geometric priors**: the ROIAligned mask logits, per-cell
  **normalized coordinates**, and the global (ROIAligned) depth. The normalized-coordinate
  channels are the crucial glue — they tell Φ_o *where in the full frame* each independently-
  cropped ROI sits, so it can relate two boxes that were cropped separately.
- **`roi_mask_logit` `(P,2,1,28,28)`** — the per-member mask, always returned because the
  scalar-depth reduction downstream needs it as a weight.

**How the resolutions are reconciled.** A normalized box `[x1,y1,x2,y2] ∈ [0,1]` is scaled by
each map's own `(W,H)` into that map's pixel grid, then `roi_align(spatial_scale=1.0)` samples
the patch. So the *same* box tensor aligns F_obj (Phase-1 feature res), D_obj (Phase-1 depth
res), the mask logits (Phase-2 res), and the GT depth (Phase-2 res) into one `28×28` frame.

---

## D.5 `relation_head.py` — Φ_o (Eq. 8–9) and the composite

**Why it exists.** This is the heart of Phase 3: the network that looks at both instances of a
pair and predicts how to correct each one's depth.

**The head (Eq. 8–9).**

```
Eq. 8:  E_obj = σ( Φ_o( [F_obj, G_obj]_main ⊕ [F_obj, G_obj]_guest ) )
Eq. 9:  D_hat = (2·E_obj − 1)·D_obj + D_obj
```

each part:
- **`⊕` (concatenate main + guest)** — Φ_o's input stacks *both* members' channels, so the two
  instances jointly determine each other's correction. That is what makes it "relation"
  reasoning rather than independent per-instance regression.
- **`Φ_o`** — a small stack of `1×1` convolutions (`hidden_dim = 256`, `num_conv = 3`) emitting
  a **2-channel** error field `[E_main, E_guest]`.
- **`σ` (sigmoid)** — squashes each error into `[0, 1]`, so `E_obj ∈ (0, 1)`.
- **Eq. 9** — a **residual multiplicative** correction. It simplifies to `D_hat = 2·E_obj·D_obj`,
  i.e. the head predicts a *ratio* `2·E_obj` on the base depth. At `E_obj = 0.5` the ratio is 1,
  so `D_hat = D_obj` exactly (no change); `E > 0.5` pushes the surface farther, `E < 0.5`
  nearer, within `(0, 2·D_obj)`.
- **Zero-initialization** — the output conv is initialized to 0, so at the start `E_obj = σ(0) =
  0.5` everywhere and Eq. 9 is the identity. Phase 3 therefore **begins as a no-op** on Phase-1
  depth and *learns* corrections away from there — it can't make things worse at initialization.

**`dense` vs `scalar` granularity** — in the default `dense` mode, `E_obj` is a full `28×28`
field, so the correction varies across the instance; in `scalar` mode it is pooled to one value
per instance and broadcast, rescaling the whole ROI uniformly.

**`roi_masked_mean(D_hat, weight)`** — reduces the dense refined ROI depth to one scalar per
instance (a mask-weighted average), used for `L_dist` and for the reported refined depth layer.

**`composite_refined_depth`** — the inference/eval step (under `@torch.no_grad()`) that writes
the corrections back into the full-resolution dense map. Because Eq. 9 is a ratio, it applies
the upsampled ratio `2·E_obj` to the **full-resolution** base depth *inside each instance's
mask* — so the base map's fine within-person geometry is preserved and merely modulated, never
replaced by a blurry 28×28 patch. Where instances contend for a pixel, **nearest-layer-wins**
(smaller scalar depth), mirroring the data engine's `_flatten_id_map`; the soft mask
probability is used as a blend alpha so the correction fades in over the silhouette instead of
leaving a hard ring. Regions with no confident pair keep Phase-1 depth verbatim.

---

## D.6 `model.py` and `output.py`

**`model.py` (`Phase3Model`)** composes everything above. In `__init__` it builds the Phase-1
depth branch, the frozen Phase-2 instance decoder, and Φ_o, and loads the trained Phase-1 /
Phase-2 checkpoints. The **freeze/train split** (paper Sec. 4.3) is:

| Component | State |
|---|---|
| Phase 1 (DINOv2 + decoder + Eq. 1–4) | **frozen by default** (`freeze_phase1=True`); paper fine-tunes at LR 1e-6 |
| Phase 2 (Swin-L Mask2Former + depth head) | **frozen** |
| Φ_o (relation head) | **trainable** (fresh) |

Its `forward` runs Phase 1, runs frozen Phase 2 under `no_grad`, builds pairs, extracts ROI
inputs, runs Φ_o, reduces to scalar layers, composites the dense map, and returns a
`RefinedDepthOutput` plus an `aux` dict (the pair set + Phase-2 output) that the training
criterion needs. If there are no pairs, refined depth = base depth (non-degrading).

**`output.py` (`RefinedDepthOutput`)** is the final contract: `refined_depth` (composited),
`base_depth` (Phase-1 depth, for measuring the delta), the per-pair bookkeeping
(`pair_query_idx`, `pair_iou`, `refined_layers`, `base_layers`), and the training-only ROI
tensors (`d_hat_roi`, `e_obj_roi`, `d_obj_roi`). Same versioned-dataclass pattern as the other
phases.

---

## D.7 `targets.py` — the ground truth for the refinement loss

**Why it exists.** The loss needs, per pair member, both a **scalar** GT depth layer (for
`L_dist`) and a **dense** GT depth patch (for `L_obj`). This file builds them, keyed by the
matcher's query↔GT assignment (the same Hungarian matcher from Phase 2, reused here to know
which GT instance each refined query corresponds to).

- **`dt_scalar (P,2)`** — each member's GT depth layer, via `build_refine_targets` (reused from
  the dataset code).
- **`dt_dense (P,2,1,28,28)`** — the GT depth, masked to the member's matched GT instance, then
  ROIAligned into the same 28×28 frame as `D_hat`.
- **`dt_valid (P,2,1,28,28)`** — **visible GT only**: `(GT depth > 0) AND (inside the matched
  instance mask)`. A single RGB-D sensor has no ground truth for hidden pixels, so `L_obj` is
  supervised only on the instance's visible surface.

**A subtle correctness fix — normalized convolution.** The GT depth is kept only on the
person's measured pixels (`depth × valid`) and zeroed elsewhere. Shrinking that map to 28×28 by
ROIAlign *averages* pixels, so a boundary cell that straddles the silhouette mixes real depth
with those zeros: a cell that is 60% person (3 m) and 40% zeroed background averages to
`0.6·3 + 0.4·0 = 1.8 m` — a target far too shallow. Left uncorrected, Φ_o would be *taught* to
push depth down at instance boundaries — exactly where occlusions occur. The fix ROIAligns a
second map (`1` on valid pixels, `0` elsewhere) to get the **valid fraction** per cell, then
divides: `dt_dense = dt_num / dt_valid_soft`. This cancels the dilution — `1.8 / 0.6 = 3 m`, the
true edge depth — by averaging over only the real pixels. Cells that end up less than half valid
are dropped, since dividing by a tiny fraction would amplify a few noisy pixels.

---

## D.8 `phase3_losses.py` — how Φ_o learns (Eq. 10–12)

```
L_obj  (Eq. 10) = SigLog( D_hat, DT )                                   over VISIBLE GT ROI pixels
L_dist (Eq. 11) = mean_p | (D_hat_i − D_hat_j)² − (DT_i − DT_j)² |      over valid pairs
L_ref  (Eq. 12) = λ_obj · L_obj + λ_dist · L_dist                       (λ_obj = 1, λ_dist = 0.5)
```

each term:
- **`L_obj` (Eq. 10)** — the same scale-invariant log loss from Phase 1 (§B.8), now comparing
  the **refined ROI depth `D_hat`** against the GT depth `DT`, masked to visible pixels
  (`dt_valid`). This pins the refined depth to the true metric surface where the sensor can see
  it. ROIs with fewer than `min_valid_roi_px = 16` valid GT pixels are skipped (1–2 stray
  sensor returns give maximally noisy supervision).
- **`L_dist` (Eq. 11)** — the **relative-depth-consistency** term, and the key occlusion signal.
  `D_hat_i, D_hat_j` are the two members' refined scalar depths and `DT_i, DT_j` their GT depth
  layers; the loss drives the **squared depth gap between the two people** to match the true
  gap. Using the squared difference makes it a symmetric, sign-robust measure of "how far apart
  in depth are the occluder and occludee" — which is exactly the quantity occlusion gets wrong.
- **`L_ref` (Eq. 12)** — the weighted sum. `λ_obj = 1`, `λ_dist = 0.5` (the paper only notes
  `L_obj` dominates and `L_dist` is secondary; the exact weights are engineering choices).

An optional anti-forgetting term (`holistic_weight`, a `SigLog` on the base depth) can keep the
whole-frame depth from drifting; it is 0 in the faithful profile.

**Example.** Occluder at true 3.0 m, occludee at 5.0 m (true gap² = 4.0). If the refined depths
come out 3.1 and 4.6 (gap² = 2.25), `L_dist = |2.25 − 4.0| = 1.75` pushes them apart toward the
correct 2.0 m separation, while `L_obj` simultaneously pins each toward its own visible GT.

---

## D.9 `phase3_config.py` and the key design decisions

Notable values: candidate thresholds `cat_conf > 0.9`, `mask_conf > 0.8`, `overlap_iou > 0.1`
(all paper-specified); `overlap_metric = "box_iou"`; ROI size `28×28`; `hidden_dim = 256`,
`num_conv = 3`; loss weights `λ_obj = 1`, `λ_dist = 0.5`; fine-tune LR `1e-6` with `head_lr_mult
= 10`; `total_iters = 25000`; `batch_size = 1` (two backbones resident). Phase 1 and Phase 2 are
loaded from their own YAMLs so their exact architectures are reproduced, not re-specified.

Design decisions worth stating:
- **Phase 1 is frozen by default** (`freeze_phase1 = True`), deviating from the paper's 1e-6
  joint fine-tune. On this single-sensor dataset, Phase 3 only supervises ROI (instance) pixels,
  so fine-tuning the whole-frame depth branch drifts the dense depth badly (measured abs_rel
  0.078 → 0.139). Frozen, the dense base stays at Phase-1 quality and the refinement head can
  only help — Phase 3 becomes **non-degrading by construction**. Set `False` to reproduce the
  paper verbatim.
- **Box IoU for overlap** — because modal, disjoint masks make mask IoU ≈ 0 for occluded pairs.
- **Zero-init Φ_o** — Phase 3 starts as an identity no-op and learns corrections from there.
- **Normalized-box ROIAlign** — reconciles the two branches' native resolutions without forcing
  a common input size.
- **Visible-only supervision** — a single sensor has no GT behind occluders.

## D.10 Phase 3 summary

Phase 3 refines depth where instances overlap. It runs the (frozen) Phase-1 depth branch and
(frozen) Phase-2 instance decoder, filters confident instances, and forms nearer-first
occlusion pairs by bounding-box IoU. For each pair it ROIAligns the depth features, base depth,
and geometric priors into a shared 28×28 frame, and a small relation head Φ_o — seeing both
members at once — predicts a residual multiplicative correction (Eq. 8–9), zero-initialized so
it starts as a no-op. Training uses a scale-invariant log loss on the refined depth over visible
GT (`L_obj`) plus a relative-depth-consistency term that matches the occluder–occludee depth gap
to ground truth (`L_dist`). The corrections are composited back into the full-resolution depth
map as a ratio field, nearest-layer-wins, leaving non-crowded regions at Phase-1 quality.

---

# PART E — TEMPORAL CONSISTENCY (VIDEO EXTENSION)

## E.0 The big picture (in plain words)

Everything so far works on **one picture at a time** — the paper's method never looks at more
than a single frame. That causes a small but annoying problem on video: the depth of something
that isn't moving — a wall, a parked chair — can **wobble slightly from frame to frame**
(3.00 m, then 3.04, then 2.98), simply because each frame is predicted from scratch. On screen
this looks like a faint shimmer, called **flicker**.

The temporal extension adds a **memory across frames** so the depth of still things stays steady,
while genuinely moving things are still allowed to change — and it must do this **without making
the single-frame depth any worse**. It is a "beyond-the-paper" add-on: the original
`instancedepth/` code is untouched; the extension lives in `videodepth/`.

## E.1 Vocabulary (plain words)

| Term | Meaning |
|---|---|
| **Flicker** | Depth of a still object wobbling frame-to-frame though nothing moved. The thing we remove. |
| **Streaming** | Processing a video one frame at a time, in order, remembering a little from the previous frame. |
| **Hidden state / memory** | A small bundle of numbers carried from one frame to the next — the model's "what did I just see" note. |
| **Recurrent / GRU** | A network that updates a memory each step. A GRU (Gated Recurrent Unit) has little "gates" deciding how much old memory to keep vs. overwrite. |
| **Residual add** | Instead of replacing the features, it computes a small *change* and adds it on top — it can only gently adjust, never destroy. |
| **Zero-init (no-op start)** | The output layer starts at all-zeros, so at the very beginning it adds nothing — the model behaves exactly like the per-frame version, and can only improve from there. |
| **Clip** | A short run of consecutive frames from one video, used for training. |
| **BPTT** | "Backprop through time" — training the memory by unrolling it over a clip's frames and learning across all of them. |

## E.2 Where it plugs in

Recall Phase 1: `image → backbone → decoder (F_0,F_1,F_2) → refinement → depth`. The temporal
module is a small block inserted **between the decoder and the refinement**, on the finest
feature map (`F_2`). For each new frame it takes `F_2`, mixes in its memory of previous frames,
and passes a **time-smoothed `F_2`** onward. Everything else in Phase 1 is unchanged.

```
frame t:  image_t → backbone → decoder → F_2_t
                                          │
                                   TemporalStabilizer  ←── memory H_{t-1}
                                          │   (updates memory → H_t)
                                     F_2_smoothed_t
                                          │
                                    refinement → depth_t
```

## E.3 The stabilizer, step by step (easy words)

Each frame, the module ([temporal_head.py](../videodepth/models/temporal_head.py)) does five
simple things:

```
F_smoothed = F  +  Up( GRU( Down(F), memory ) )
```

1. **Shrink it (`Down`).** Shrink the feature map to a **quarter of its size** on each side.
   Fixing flicker doesn't need every pixel — a small grid is enough and far cheaper.
2. **Thin the channels (`proj_in`).** A 1×1 conv reduces the channels to a small working width (128).
3. **Update the memory (`GRU`).** Two GRU cells blend the current (shrunk) features with the
   memory carried from previous frames, and produce an updated memory — where "what was here a
   moment ago" meets "what's here now."
4. **Grow it back (`proj_out` + `Up`).** A 1×1 conv — **started at all zeros** — turns the memory
   into a feature-shaped correction, upsampled back to full size.
5. **Add it on (`residual`).** The correction is **added** to the original features. Because
   step 4 starts at zero, the correction is exactly 0 at first — the module changes nothing — and
   it learns small corrections from there.

Two properties that matter: it **starts harmless** (zero-init → identical to the per-frame model
at first, so it can never start worse), and it **streams forever** (a fixed-size memory updated
once per frame → any video length at constant cost; the memory is wiped at each new video via
`reset_state()`).

## E.4 The memory update — the GRU gates (easy words)

Inside, each GRU cell ([ConvGRUCell](../instancedepth/models/hdi/temporal.py)) decides, **per
pixel**, how much to trust the old memory vs. the new frame, using two little "dials" (gates),
each a number between 0 and 1:

```
z, r = sigmoid(...)               # update dial z, reset dial r
c     = tanh(...)                 # a fresh candidate memory, using r to filter the old memory
memory_new = (1 − z)·memory_old  +  z·c
```

- The **reset dial `r`** decides how much of the old memory to ignore when forming the candidate.
- The **update dial `z`** decides how much to keep the old memory vs. write the new candidate. On
  a **still** region the model learns `z ≈ 0` → "hold steady" (this is what kills flicker); on a
  **moving** region `z ≈ 1` → "follow the new frame" (so real motion still comes through).

## E.5 The training signal — Temporal Gradient Matching (TGM)

A memory does nothing unless training **rewards steadiness**. The obvious idea — "force this
frame's depth to equal last frame's" — is **wrong**, because a person genuinely walking closer
*should* change depth. So instead of matching depths, TGM ([temporal_losses.py](../videodepth/losses/temporal_losses.py))
matches **changes**: the prediction's frame-to-frame change should equal the ground truth's
change.

```
L_tgm = average | (log d_t − log d_{t-1}) − (log dt_t − log dt_{t-1}) |
```

- `d` = prediction, `dt` = ground truth; `log` keeps it scale-consistent (same space as SigLog).
- `(log d_t − log d_{t-1})` is **how much the prediction's depth changed since last frame**; the
  `dt` term is how much it *truly* changed.
- If the prediction changes **exactly as the ground truth does**, the two cancel and the loss is
  **0** — a moving person is never punished for legitimately changing depth.
- Only **flicker** — a change the ground truth does *not* have — makes the two differ, and that
  is what gets penalized.
- It is measured only where **both** frames have a valid sensor reading, so a sensor hole can't
  invent a fake change.

This is the flicker metric (TAE) turned into a trainable objective. The full temporal loss is
`per-frame SigLog + weight · TGM`: single-frame accuracy stays pinned while flicker is squeezed.

## E.6 How the temporal stage is trained (execution)

The temporal module is trained as a **second stage** on top of a finished Phase-1 model
([train_video.py](../videodepth/engine/train_video.py)):

1. **Load the trained Phase-1 weights and freeze them** — only the small stabilizer learns (a
   few hundred K parameters). The single-frame depth stays fixed, so the temporal effect is clean
   to measure.
2. **Feed short clips, not shuffled frames.** Training uses ordered clips of consecutive frames
   ([clip_dataset.py](../instancedepth/data/clip_dataset.py) / [motion_clips.py](../videodepth/data/motion_clips.py)).
   They are **stride-augmented** so a 5-frame clip can span up to ~100 frames — because motion
   here is tiny between neighbors and only becomes visible over long spans. Any random flip is
   decided **once per clip** (a per-frame flip would look like impossible motion to the memory).
3. **Unroll over the clip (BPTT).** For each clip: wipe the memory, run the frames in order,
   carrying the memory (and its gradient) across them, and sum the per-frame loss plus TGM.
   Backprop then flows through the whole clip.
4. **Pick the best checkpoint on a *streaming* score.** The saved "best" model is chosen by
   running a real, in-order, memory-on evaluation and scoring `abs_rel + weight·TAE` — so the
   choice reflects how steady the video looks, not just single-frame accuracy.

## E.7 The video model + per-person steadiness (brief)

- **[video_model.py](../videodepth/models/video_model.py)** wraps a trained Phase-1 checkpoint
  and re-runs `backbone → decoder → [stabilizer] → refinement`, exposing a clip forward pass — no
  edits to `instancedepth`.
- The package also keeps **per-person** depth steady through time with **no extra training**:
  `TrackDepthMemory` smooths each tracked person's depth layer and, when a person is hidden by an
  occluder, lets their depth **coast on its recent velocity** (a person walking away keeps
  receding while unseen) — supplying occluded depth from *time* instead of a second sensor;
  `QueryInstanceTracker` keeps stable person IDs across frames by matching Phase-2's query
  embeddings (so two people crossing keep their identities, and someone reappearing after an
  occlusion is re-matched).

## E.8 Temporal summary

The temporal extension adds a small residual ConvGRU "stabilizer" on the finest Phase-1 feature
map. Each frame it shrinks the features, blends them with a carried-over memory through gated
updates, and adds a small correction back — starting from an exact no-op (zero-init) so it can
never start worse. It is trained as a frozen-spatial second stage on motion-spanning clips with
backprop-through-time, using Temporal Gradient Matching, which penalizes only *flicker* (change
the ground truth doesn't have) and never a moving object's legitimate depth change; the best
checkpoint is chosen on a streaming, flicker-aware score. Separately, training-free per-track
memory keeps each person's depth steady and coasts it through occlusions.

---

# PART F — HOW TRAINING WORKS (all phases)

## F.0 The big picture (in plain words)

"Training" a model means: show it an example, measure **how wrong** its answer is (a single
number called the **loss**), and then nudge its internal numbers (its **weights**) a tiny bit so
it's a little less wrong next time. Repeat that millions of times, and the weights slowly settle
into values that give good answers.

Every phase in this project trains through **one shared engine**
([trainer.py](../instancedepth/engine/trainer.py)). Each phase only supplies three things — a
**model**, a **loss function**, and a **data source** — and the engine handles all the repetitive
machinery (looping, saving, measuring). So once you understand this one engine, you understand
how *all four* trainings run.

## F.1 Vocabulary (plain words)

| Term | Meaning |
|---|---|
| **Weights / parameters** | The model's adjustable numbers — exactly what training changes. |
| **Loss** | One number saying "how wrong is this prediction." Training makes it smaller. |
| **Gradient** | For each weight, the direction (and how much) to nudge it to reduce the loss. Computed automatically by calculus. |
| **Backpropagation ("backward")** | The step that computes those gradients for every weight at once. |
| **Optimizer** | The rule that applies the nudges. We use **AdamW**. |
| **Learning rate (LR)** | How big each nudge is. Too big → unstable; too small → painfully slow. |
| **Iteration / step** | Processing one batch = one nudge. Training runs a fixed number of these. |
| **Batch** | A handful of examples processed together (e.g. 4 images at once). |
| **Checkpoint** | A saved snapshot of the model's weights (so you can stop, resume, or evaluate). |
| **Freeze** | Mark some weights "do not change" during training. |
| **Fine-tune** | Keep training an already-trained model, gently (small LR). |
| **Mixed precision** | Doing the math in a smaller number format (bf16) to save memory and time. |

## F.2 The one shared engine (`Trainer`)

You hand the `Trainer` a model, a `compute_loss` function, and a data loader. On construction it
sets up: the **optimizer** (how weights get nudged), the **LR schedule** (how the nudge size
changes over time), **mixed precision**, a **TensorBoard** logger, and an iteration counter. Then
`fit()` runs the loop. Everything below is what lives inside that.

## F.3 One training step, in plain words (`train_step`)

This is the heart of training — the same seven moves every step:

```python
model.train()                          # 1. put the model in "learning" mode
optimizer.zero_grad()                  # 2. clear last step's leftover gradients
with autocast(bf16):                   # 3. (mixed precision) run the model + compute the loss
    losses = compute_loss(model, batch, device)
    total  = losses["total"]
total.backward()                       # 4. backward: compute the gradient for every weight
clip_grad_norm_(model.parameters(), grad_clip_norm)   # 5. cap unusually huge gradients (safety)
optimizer.step()                       # 6. nudge every weight (nudge size set by the LR)
scheduler.step()                       # 7. advance the LR schedule
```

In words: **run → measure wrongness → find which way to nudge each weight → nudge them a little →
repeat.** Two safety touches: **gradient clipping** (step 5) shrinks the nudge if it's freakishly
large, so one weird batch can't blow up training; and there's a guard for a rare case — if a
batch produces a loss that isn't connected to any *trainable* weight (e.g. Phase 3 with a frozen
depth branch on a frame that happens to have **no overlapping people**), there's nothing to learn,
so it **skips the backward/step but still advances the counter** — training never crashes and
stays on its iteration budget.

## F.4 Learning rate: warm up, then slow down

The nudge size (LR) is not constant — it follows a schedule with two ideas:

```python
if step < warmup_iters:  lr = base_lr × (step / warmup_iters)     # (a) warm up
else:                    lr = base_lr × (1 − step/total_iters)^0.9 # (b) slowly decay to ~0
```

- **(a) Warmup** — start the LR small and ramp it up over the first steps, so the very first
  nudges don't jolt the freshly-loaded model.
- **(b) Polynomial decay** — gradually shrink the LR toward 0 by the end, so early training can
  explore boldly and late training settles into a precise answer.

## F.5 Different speeds for different parts (parameter groups)

Not every weight should move at the same speed. The **pretrained backbone** already knows a lot,
so it's nudged **slowly** (small LR); the **fresh heads** start from scratch, so they're nudged
**faster** (bigger LR). The optimizer splits the weights into groups with different LRs to do
this. **Frozen** weights are left out of the optimizer entirely, so they never move.

```python
groups = [ {backbone weights: lr × backbone_mult},   # slow
           {head weights:     lr × head_mult} ]        # fast
AdamW(groups, weight_decay=...)
```

## F.6 Mixed precision (speed) and gradient clipping (safety)

- **Mixed precision (bf16)** — doing the arithmetic in a compact 16-bit number format uses about
  half the memory and runs faster on the GPU, with no meaningful accuracy loss for this work.
- **Gradient clipping** — already mentioned: caps the overall size of the nudge so training stays
  stable.

## F.7 The training loop (`fit`): logging, saving, choosing the best

```python
while iteration < total_iters:                 # runs a fixed number of steps, cycling the data forever
    batch = next(data)
    loss  = train_step(batch)
    iteration += 1
    if iteration % log_every  == 0:  log the loss + LR to TensorBoard
    if iteration % ckpt_every == 0:  save "latest.pth" (+ a numbered snapshot)
    if iteration % eval_every == 0:
        metrics = eval_fn(model)               # measure accuracy on held-out data
        if metrics improved:  save "best.pth"  # keep the best model seen so far
```

Two things worth noting: training is counted in **iterations**, not epochs — the data loader is
just cycled forever until the iteration budget is spent. And **"best.pth"** is whichever
checkpoint scored best on the periodic evaluation, so you always keep the best model, not just the
last one. A small **manifest** file also records the exact config and code version, so any run can
be reproduced later.

## F.8 What each phase plugs in (`compute_loss`)

The *only* thing that really differs between phases is the `compute_loss` function — one small
recipe describing "run the model on this batch and return the loss":

- **Phase 1** ([train_hdi.py](../instancedepth/engine/train_hdi.py)): `image → model → depth`,
  then `HDILoss` against the GT depth.
- **Phase 2** ([train_phase2.py](../instancedepth/engine/train_phase2.py)): `image → predictions`,
  then **match** the 100 predictions to the real people (Hungarian matcher), then `Phase2Criterion`.
- **Phase 3** ([train_phase3.py](../instancedepth/engine/train_phase3.py)): `image → refined
  output`, build the occlusion-pair targets, then `Phase3Criterion`.
- **Temporal** ([train_video.py](../videodepth/engine/train_video.py)): run a whole **clip**, then
  per-frame `SigLog` + the `TGM` temporal loss (Part E).

## F.9 The training recipe per phase (concrete numbers)

| | Phase 1 | Phase 2 | Phase 3 | Temporal |
|---|---|---|---|---|
| **What it learns** | dense depth + features | per-person masks, class, depth | occlusion corrections (Φ_o) | frame-to-frame steadiness |
| **Learning rate** | 1e-5 | 1e-4 | 1e-6 | small |
| **Iterations** | 55,000 | 20,000 | 25,000 | (per config) |
| **Batch size** | 4 | 2 | 1 | small |
| **What's frozen** | nothing | nothing | **Phase 1 + Phase 2 frozen; only Φ_o trains** | **spatial model frozen; only the stabilizer trains** |
| **Number format** | bf16 | bf16 | bf16 | bf16 |

Smaller batches and lower LRs go with the later stages because they hold more (and larger) frozen
sub-models in memory and are *fine-tuning* delicately rather than learning from scratch.

## F.10 The order the phases are trained (how it all chains)

```
1. Generate the dataset          (data engine, Part A)          → gid_custom/ labels
2. Train Phase 1 (depth)         reads the labels               → hdi checkpoint
3. Train Phase 2 (instances)     reads the labels, independent  → phase2 checkpoint
4. Train Phase 3 (occlusion)     loads FROZEN Phase 1 + Phase 2, trains only Φ_o
5. (optional) Train temporal     loads FROZEN Phase 1, trains only the stabilizer
```

Phases 1 and 2 are independent, so they can be trained in either order (or in parallel). Phase 3
and the temporal stage both build **on top of** finished checkpoints and only train their own
small new piece — which is why they're fast and can never damage the parts they freeze.

---

# WHERE THE THREE PARTS CONNECT

```
Data engine (Part A)  ──►  gid_custom/ annotations (masks, track IDs, depth layers)
                              │
              ┌───────────────┴───────────────┐
              ▼                                 ▼
   Phase 1 (Part B)                    Phase 2 (Part C)
   dense metric depth +                per-person masks, classes,
   multi-scale features                depth layers Dep_i
              │                                 │
              └───────────────┬─────────────────┘
                              ▼
                   Phase 3 — Occlusion-Aware Depth Refinement
                   (pairs overlapping people, corrects depth where they overlap)
```

The data engine manufactures the labels that both Phase 1 and Phase 2 train against; Phase 1
provides the initial depth and features; Phase 2 provides the instances and their rough
depths; Phase 3 combines them to fix depth around occlusions.
