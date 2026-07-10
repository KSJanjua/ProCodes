"""Occlusion-frame selection for Phase-3 training (plan SS10.2, and the
practical fix for the ~5% valid-pair rate observed on uniform sampling).

Phase 3's refinement loss only exists on frames that contain a confident
*overlapping* instance pair; uniform frame sampling wastes most steps on
pair-less frames (loss == 0, no gradient to Phi_o). This module picks the
subset of frames that have >= 2 instances whose **bounding boxes overlap** --
a cheap superset of the true mask-IoU>0.1 criterion, computed straight from
the annotation manifest (no image/mask loading), so it scales to the whole
55k-frame dataset in seconds.

It is a *superset* (bboxes can overlap while masks don't), so some selected
frames still yield no pair -- that's fine; the goal is to raise the hit rate,
not to filter exactly. Exact mask-IoU filtering happens later, per batch, in
``candidates.build_pairs``.
"""

from __future__ import annotations

import logging
from typing import List

log = logging.getLogger("instancedepth.data.occlusion_index")


def _bboxes_overlap(a, b) -> bool:
    """xyxy overlap test (touching edges don't count as overlap)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _frame_has_overlap(instances, max_depth: float) -> bool:
    boxes = [ins["bbox_xyxy"] for ins in instances
             if 0.0 < float(ins.get("depth_layer_m", 0.0)) <= max_depth
             and ins.get("bbox_xyxy") is not None]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _bboxes_overlap(boxes[i], boxes[j]):
                return True
    return False


def occlusion_frame_indices(dataset, max_depth: float = 10.0) -> List[int]:
    """Indices ``i`` into ``dataset.index`` whose frame has >= 2 valid
    instances with overlapping bounding boxes.

    ``dataset`` must expose ``index`` as a list of ``(manifest, frame_key)``
    (the ``GIDInstanceDepthDataset`` layout); each
    ``manifest['frames'][frame_key]['instances']`` carries ``bbox_xyxy`` and
    ``depth_layer_m`` (written by ``data_engine/annotate.py``).
    """
    idxs: List[int] = []
    for i, (manifest, fkey) in enumerate(dataset.index):
        instances = manifest["frames"][fkey].get("instances", [])
        if len(instances) >= 2 and _frame_has_overlap(instances, max_depth):
            idxs.append(i)
    log.info("occlusion_frame_indices: %d / %d frames have >=2 overlapping instances",
             len(idxs), len(dataset.index))
    return idxs
