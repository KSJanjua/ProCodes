"""Tests for the shared visualization helpers (utils/viz.py) that carry
behavioral contracts the video/panel tools rely on."""

from __future__ import annotations

import numpy as np

from instancedepth.utils.viz import colorize_depth, draw_instances_with_depth


def test_draw_instances_touches_only_masked_pixels():
    bgr = np.full((32, 32, 3), 120, np.uint8)
    m = np.zeros((32, 32), bool)
    m[4:12, 4:12] = True
    out = draw_instances_with_depth(bgr, [m], [3.0])
    assert out.shape == bgr.shape and out.dtype == np.uint8
    assert not np.array_equal(out[4:12, 4:12], bgr[4:12, 4:12])   # fill applied
    # pixels far from the mask (and its label text) stay untouched
    assert np.array_equal(out[24:, 24:], bgr[24:, 24:])
    # input not mutated
    assert bgr[5, 5, 0] == 120


def test_draw_instances_empty_is_identity():
    bgr = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    out = draw_instances_with_depth(bgr, [], [])
    assert np.array_equal(out, bgr)


def test_draw_instances_far_to_near_order():
    """The nearer instance must be drawn LAST (on top): on a contested pixel,
    the final blend uses the nearer instance's colour over the farther's --
    so swapping the depth order must change the contested pixel."""
    bgr = np.full((32, 32, 3), 120, np.uint8)
    a = np.zeros((32, 32), bool); a[4:20, 4:20] = True
    b = np.zeros((32, 32), bool); b[12:28, 12:28] = True
    near_a = draw_instances_with_depth(bgr, [a, b], [2.0, 5.0])
    near_b = draw_instances_with_depth(bgr, [a, b], [5.0, 2.0])
    assert not np.array_equal(near_a[13:19, 13:19], near_b[13:19, 13:19])


def test_colorize_depth_invalid_black():
    d = np.zeros((8, 8), np.float32)
    d[0, 0] = 5.0
    out = colorize_depth(d, max_depth=10.0)
    assert out.shape == (8, 8, 3)
    assert (out[1:, 1:] == 0).all()          # invalid (depth==0) renders black
    assert out[0, 0].sum() > 0               # valid pixel is coloured
