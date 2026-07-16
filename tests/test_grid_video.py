"""Test for the make_grid_video layout helper (pure array assembly -- no
models or dataset files needed)."""

from __future__ import annotations

import numpy as np

from scripts.make_grid_video import compose_grid


def test_compose_grid_layout():
    H, W = 80, 100
    # five distinct solid-colour panels so we can check each lands in its cell
    rgb = np.full((H, W, 3), 10, np.uint8)
    p1 = np.full((H, W, 3), 50, np.uint8)
    gtd = np.full((H, W, 3), 90, np.uint8)
    pred = np.full((H, W, 3), 130, np.uint8)
    gti = np.full((H, W, 3), 170, np.uint8)

    grid = compose_grid(rgb, p1, gtd, pred, gti)
    assert grid.shape == (2 * H, 3 * W, 3)          # 2 rows x 3 cols

    # sample the BOTTOM-centre of each cell -- clear of the top-left label bar
    def cell(r, c):
        return grid[r * H + H - 3, c * W + W // 2, 0]
    assert cell(0, 0) == 10                          # row1: RGB
    assert cell(0, 1) == 50                          #       Phase-1 depth
    assert cell(0, 2) == 90                          #       GT depth
    assert cell(1, 0) == 10                          # row2: RGB (repeated)
    assert cell(1, 1) == 130                         #       pred instances
    assert cell(1, 2) == 170                         #       GT instances


def test_compose_grid_does_not_mutate_inputs():
    H, W = 20, 20
    rgb = np.full((H, W, 3), 10, np.uint8)
    others = [np.full((H, W, 3), v, np.uint8) for v in (50, 90, 130, 170)]
    compose_grid(rgb, *others)
    assert rgb[0, 0, 0] == 10 and all(o[0, 0, 0] == v for o, v in zip(others, (50, 90, 130, 170)))
