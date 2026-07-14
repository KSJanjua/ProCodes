"""Shared visualization helpers (cv2-only -- no matplotlib dependency) used by
scripts/visualize_phase3.py, scripts/make_sequence_videos.py and
scripts/infer_video.py. All image outputs are uint8 BGR (cv2 convention).

Colormap conventions (documented once, used everywhere):
  * depth        : TURBO, NEAR = warm/red, FAR = cool/blue, invalid (<=0) = black.
  * abs error    : MAGMA, 0 = black -> cap = bright.
  * signed maps  : JET centered at the midpoint (negative = blue, positive = red).
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), np.float32)


# --------------------------------------------------------------------------- #
# colorization
# --------------------------------------------------------------------------- #
def colorize_depth(depth_m: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """(H,W) metric depth -> BGR. Near = warm, far = cool, invalid = black."""
    d = np.asarray(depth_m, np.float32)
    valid = d > 0
    norm = np.clip(d / max_depth, 0.0, 1.0)
    inv = ((1.0 - norm) * 255.0).astype(np.uint8)   # near -> 255 (TURBO's warm end)
    bgr = cv2.applyColorMap(inv, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


def colorize_error(err: np.ndarray, cap: float, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """(H,W) non-negative error -> BGR (MAGMA), clipped at ``cap``."""
    e = np.clip(np.asarray(err, np.float32) / max(cap, 1e-8), 0.0, 1.0)
    bgr = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    if valid is not None:
        bgr[~valid] = 0
    return bgr


def colorize_signed(x: np.ndarray, cap: float, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """(H,W) signed map -> BGR (JET), 0 at the middle, +-cap at the ends."""
    n = np.clip(np.asarray(x, np.float32) / max(cap, 1e-8), -1.0, 1.0)
    bgr = cv2.applyColorMap(((n + 1.0) * 127.5).astype(np.uint8), cv2.COLORMAP_JET)
    if valid is not None:
        bgr[~valid] = 0
    return bgr


def denormalize_image(img_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) ImageNet-normalized float array -> BGR uint8."""
    rgb = img_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# overlays
# --------------------------------------------------------------------------- #
def overlay_masks(bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.45, seed: int = 0) -> np.ndarray:
    """Overlay (N,H,W) boolean masks in distinct colors (same palette pattern
    as data_engine/annotate.py::_preview)."""
    out = bgr.copy()
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 255, size=(max(len(masks), 1), 3), dtype=np.uint8)
    for i, m in enumerate(masks):
        m = m.astype(bool)
        out[m] = ((1 - alpha) * out[m] + alpha * palette[i]).astype(np.uint8)
    return out


def draw_instances_with_depth(bgr: np.ndarray, masks: Sequence[np.ndarray],
                              depths: Sequence[float], alpha: float = 0.45) -> np.ndarray:
    """Overlay instance masks together with their depth values (metres).

    Instances are drawn far-to-near so the nearest sits on top -- the same
    occlusion-ordering convention as the dataset's id-map flattening
    (``data_engine/annotate.py::_flatten_id_map``) and
    ``scripts/visualize_phase2.py``. Each instance gets a golden-ratio-spaced
    colour, a translucent fill, a contour, and its depth printed at the mask
    centroid (dark halo + white text for legibility on any background).
    """
    out = bgr.copy()
    if len(masks) == 0:
        return out
    order = np.argsort(-np.asarray(depths, np.float32))
    for draw_i, idx in enumerate(order.tolist()):
        m = np.asarray(masks[idx], bool)
        if not m.any():
            continue
        hue = (draw_i * 0.61803398875) % 1.0          # colour by draw order for max separation
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        color = np.array([b * 255, g * 255, r * 255], np.uint8)   # BGR
        out[m] = ((1 - alpha) * out[m] + alpha * color).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, [int(c) for c in color], 2)
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = f"{float(depths[idx]):.1f}m"
        cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_pairs(bgr: np.ndarray, boxes_norm: np.ndarray, ious: Sequence[float],
               deps: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw (P,2,4) normalized pair boxes: member 0 (nearer/occluder) green,
    member 1 (farther) orange, a line between box centers, and the pair IoU."""
    out = bgr.copy()
    H, W = out.shape[:2]
    colors = [(80, 220, 80), (60, 160, 255)]   # BGR: green, orange
    for p in range(boxes_norm.shape[0]):
        centers = []
        for k in range(2):
            x1, y1, x2, y2 = boxes_norm[p, k]
            pt1 = (int(x1 * W), int(y1 * H))
            pt2 = (int(x2 * W), int(y2 * H))
            cv2.rectangle(out, pt1, pt2, colors[k], 2)
            centers.append(((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2))
            if deps is not None:
                cv2.putText(out, f"{deps[p, k]:.2f}m", (pt1[0], max(pt1[1] - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[k], 1, cv2.LINE_AA)
        cv2.line(out, centers[0], centers[1], (255, 255, 255), 1, cv2.LINE_AA)
        mid = ((centers[0][0] + centers[1][0]) // 2, (centers[0][1] + centers[1][1]) // 2)
        cv2.putText(out, f"IoU {ious[p]:.2f}", mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def put_label(bgr: np.ndarray, text: str) -> np.ndarray:
    """Stamp a label bar onto the top-left corner (in place-safe copy)."""
    out = bgr.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(out, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(out, text, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def stack_grid(panels: List[Tuple[str, np.ndarray]], cols: int, cell_h: int = 320) -> np.ndarray:
    """Tile labeled panels into a grid. Each panel is resized to ``cell_h``
    keeping the FIRST panel's aspect ratio (panels in one grid are expected to
    share aspect; differing ones are resized to fit)."""
    assert panels
    h0, w0 = panels[0][1].shape[:2]
    cell_w = int(round(cell_h * w0 / h0))
    cells = []
    for label, img in panels:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cells.append(put_label(img, label))
    rows = []
    for r in range(0, len(cells), cols):
        row = cells[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def hstack_panels(panels: List[np.ndarray], height: Optional[int] = None) -> np.ndarray:
    """Horizontally stack images, resizing each to a common height."""
    h = height or panels[0].shape[0]
    resized = []
    for p in panels:
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        scale = h / p.shape[0]
        resized.append(cv2.resize(p, (max(int(round(p.shape[1] * scale)), 1), h),
                                  interpolation=cv2.INTER_AREA))
    return np.hstack(resized)


# --------------------------------------------------------------------------- #
# video io
# --------------------------------------------------------------------------- #
def open_video_writer(path: Path, fps: float, frame_wh: Tuple[int, int]):
    """Open a cv2.VideoWriter, preferring mp4v/.mp4 and falling back to
    XVID/.avi. Returns (writer, actual_path)."""
    for fourcc, suffix in (("mp4v", ".mp4"), ("XVID", ".avi")):
        p = Path(path).with_suffix(suffix)
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, frame_wh)
        if w.isOpened():
            return w, p
        w.release()
    raise RuntimeError(f"cv2.VideoWriter failed for both mp4v and XVID at {path}")
