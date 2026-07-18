"""Unified depth-predictor factory used by the video/visualization scripts.

Returns a single ``predict(bgr_frame) -> (H,W) float32 metric depth`` closure
for either the Phase-1 holistic model or the full Phase-3 pipeline, so tools
that only need "RGB frame in, depth map out" (scripts/make_sequence_videos.py,
scripts/infer_video.py) don't each re-implement the phase dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

log = logging.getLogger("instancedepth.predict")


def _build_phase3_inferencer(cfg, checkpoint: str):
    """Dispatch to the Phase-3 inferencer matching this checkpoint's relation-
    head architecture.

    The paper-faithful MLP Φo (``models/phase3/relation_head.py``) and the
    videodepth package's bounded pair-attention head
    (``videodepth/models/occlusion.py``) have different state-dict keys and
    shapes, so loading a checkpoint trained with one into a model built with
    the other raises immediately (docs/AUDIT_2026.md). Peek the checkpoint's
    own keys and pick the matching class -- the same detect-from-file-and-log
    pattern already used for the DAv2 encoder checkpoint in
    ``models/backbone/dinov2_wrapper.py`` -- so callers (and this module)
    never need a --relation-head flag or prior knowledge of the checkpoint.

    ``instancedepth`` stays videodepth-independent for the common case: the
    import of ``videodepth`` only happens, lazily, for a checkpoint that
    actually needs it.
    """
    import torch

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    keys = ckpt.get("model", ckpt).keys()
    del ckpt   # this load was only to peek; both inferencers reload from disk themselves

    # Pure key check, inlined (not imported from videodepth): keeps the
    # vanilla path -- the overwhelming common case -- free of any
    # videodepth import. Must stay in lockstep with
    # videodepth.models.phase3_video.is_bounded_relation_head_checkpoint,
    # which the test suite checks agree.
    is_bounded = any(k.startswith("relation_head.cross.") for k in keys)

    if is_bounded:
        log.info("Checkpoint '%s' looks like a videodepth Phase-3 checkpoint "
                 "(bounded pair-attention relation head); using Phase3VideoInferencer.",
                 checkpoint)
        from videodepth.models.phase3_video import Phase3VideoInferencer
        return Phase3VideoInferencer(cfg, checkpoint)

    from instancedepth.models.phase3.inference import Phase3Inferencer
    return Phase3Inferencer(cfg, checkpoint)


def build_depth_predictor(
    phase: int, config: str, checkpoint: str, overrides: List[str] | None = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Load a trained model and return ``(predict, max_depth)``.

    ``predict`` takes a BGR uint8 frame and returns a float32 metric-depth map
    at the model's own output resolution (callers resize as needed).
    ``phase`` is 1 (holistic) or 3 (occlusion-refined). ``max_depth`` is the
    configured metric range, for colorization.
    """
    overrides = overrides or []
    if phase == 1:
        import torch

        # Route by checkpoint content, like _build_phase3_inferencer below: a
        # temporal Phase-1 checkpoint (videodepth VideoDepthModel: spatial
        # model nested under 'spatial.', stabilizer under 'temporal.') cannot
        # load into the bare HolisticDepthModel, and vice versa. Inlined key
        # check (marker 'spatial.') so the vanilla path never imports
        # videodepth; kept in lockstep with
        # videodepth.models.video_model.is_video_depth_checkpoint (test-enforced).
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        keys = ckpt.get("model", ckpt).keys()
        is_video = any(k.startswith("spatial.") for k in keys)
        del ckpt   # peek only; the inferencer reloads from disk itself

        if is_video:
            log.info("Checkpoint '%s' looks like a temporal Phase-1 (VideoDepthModel) "
                     "checkpoint; using VideoDepthInferencer. Pass the matching "
                     "videodepth config (e.g. videodepth/configs/video_temporal_dav2.yaml).",
                     checkpoint)
            from videodepth.configs.config import VideoConfig
            from videodepth.models.video_model import VideoDepthInferencer

            vcfg = VideoConfig.from_yaml_with_overrides(config, overrides)
            vinf = VideoDepthInferencer(vcfg, checkpoint)

            def predict(bgr: np.ndarray) -> np.ndarray:
                return vinf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            predict.reset = vinf.reset_temporal_state
            return predict, vinf.max_depth

        from instancedepth.configs.config import HDIConfig
        from instancedepth.models.hdi.inference import HDIInferencer

        cfg = HDIConfig.from_yaml_with_overrides(config, overrides)
        inf = HDIInferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> np.ndarray:
            out = inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return out.depth_final[0, 0].float().cpu().numpy()

        predict.reset = inf.reset_temporal_state   # sequence-boundary hook (no-op for per-frame models)
        return predict, cfg.bins.max_depth

    if phase == 3:
        from instancedepth.configs.phase3_config import Phase3Config

        cfg = Phase3Config.from_yaml_with_overrides(config, overrides)
        inf = _build_phase3_inferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> np.ndarray:
            return inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["refined"]

        predict.reset = inf.reset_temporal_state
        return predict, cfg.data.max_depth

    raise ValueError(
        f"build_depth_predictor supports phase 1 or 3, got {phase} "
        "(Phase 2 predicts instances only, no dense depth -- use build_scene_predictor)"
    )


def build_scene_predictor(
    phase: int, config: str, checkpoint: str, overrides: List[str] | None = None,
    inst_score_thresh: float = 0.5, mask_binarize_thresh: float = 0.5,
) -> Tuple[Callable[[np.ndarray], dict], float]:
    """Like :func:`build_depth_predictor`, but the returned ``predict`` also
    exposes per-frame instance predictions where the model has them.

    ``predict(bgr)`` returns a dict:
      depth        (H,W) float32 metric depth at the input frame's resolution,
                   or **None** for Phase 2, which predicts no dense depth
                   (callers must skip the depth panel)
      masks        list of (H,W) bool instance masks -- empty for Phase 1,
                   which is holistic-only (callers fall back to GT masks)
      mask_depths  list of float -- predicted depth layer Dep_i per mask
      mask_ids     list of int -- the query index behind each mask; a stable-ish
                   identifier for consistent visualization colouring (the image
                   Mask2Former has no temporal tracking, so this is only as
                   stable as query specialization -- see docs/ARCHITECTURE.md)

    ``inst_score_thresh`` is a visualization-oriented category-confidence cut
    (0.5, matching evaluate_phase2/visualize_phase2), deliberately looser than
    Phase 3's own 0.9 candidate filter so the overlay shows what the instance
    branch actually sees.
    """
    overrides = overrides or []
    if phase == 2:
        # Phase 2 is a standalone instance branch: masks + Dep_i, no dense
        # depth. Its Dep_i comes from an MLP on Mask2Former query embeddings
        # and never reads Phase 1, so these predictions are invariant to the
        # Phase-1 checkpoint (temporal or not).
        import torch

        from instancedepth.configs.phase2_config import Phase2Config
        from instancedepth.models.hdi.inference import preprocess
        from instancedepth.models.phase2.model import Phase2Model
        from instancedepth.utils.checkpoint import load_checkpoint

        cfg = Phase2Config.from_yaml_with_overrides(config, overrides)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Phase2Model(
            checkpoint=cfg.model.checkpoint, checkpoint_dir=cfg.model.checkpoint_dir,
            allow_hub_download=cfg.model.allow_hub_download, num_classes=cfg.model.num_classes,
        ).to(device)
        load_checkpoint(Path(checkpoint), model, map_location=str(device), restore_rng=False)
        model.eval()

        @torch.no_grad()
        def predict(bgr: np.ndarray) -> dict:
            H, W = bgr.shape[:2]
            x = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tuple(cfg.data.image_size), device)
            out = model(x)
            keep = torch.where(out.scores()[0] > inst_score_thresh)[0]
            masks, deps, ids, embeds = [], [], [], []
            for q in keep.tolist():
                m = (out.mask_logits[0, q].sigmoid() > mask_binarize_thresh).float().cpu().numpy()
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                m = m.astype(bool)
                if m.any():
                    masks.append(m)
                    deps.append(float(out.depth_layers[0, q]))
                    ids.append(q)
                    # final-decoder-layer query embedding: the Mask2Former-VIS
                    # identity cue, consumed by videodepth's QueryInstanceTracker
                    embeds.append(out.query_embeddings[0, q].float().cpu().numpy())
            return dict(depth=None, masks=masks, mask_depths=deps, mask_ids=ids,
                        mask_embeds=embeds)

        predict.reset = lambda: None
        return predict, cfg.data.max_depth

    if phase == 1:
        base_predict, max_depth = build_depth_predictor(1, config, checkpoint, overrides)

        def predict(bgr: np.ndarray) -> dict:
            return dict(depth=base_predict(bgr), masks=[], mask_depths=[], mask_ids=[],
                        mask_embeds=[])

        predict.reset = getattr(base_predict, "reset", lambda: None)
        return predict, max_depth

    if phase == 3:
        import torch

        from instancedepth.configs.phase3_config import Phase3Config

        cfg = Phase3Config.from_yaml_with_overrides(config, overrides)
        inf = _build_phase3_inferencer(cfg, checkpoint)

        def predict(bgr: np.ndarray) -> dict:
            H, W = bgr.shape[:2]
            out = inf.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            p2 = out["aux"]["p2"]
            keep = torch.where(p2.scores()[0] > inst_score_thresh)[0]
            masks, deps, ids, embeds = [], [], [], []
            for q in keep.tolist():
                m = (p2.mask_logits[0, q].sigmoid() > mask_binarize_thresh).float().cpu().numpy()
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                m = m.astype(bool)
                if m.any():
                    masks.append(m)
                    deps.append(float(p2.depth_layers[0, q]))
                    ids.append(q)
                    embeds.append(p2.query_embeddings[0, q].float().cpu().numpy())
            return dict(depth=out["refined"], masks=masks, mask_depths=deps, mask_ids=ids,
                        mask_embeds=embeds)

        predict.reset = inf.reset_temporal_state
        return predict, cfg.data.max_depth

    raise ValueError(f"build_scene_predictor supports phase 1, 2 or 3, got {phase}")
