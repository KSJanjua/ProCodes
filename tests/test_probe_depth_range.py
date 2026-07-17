"""Unit tests for scripts/probe_depth_range.py's pure numeric core (no model,
no video I/O -- see summarize_depth_samples's own docstring for why it's
factored out separately)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.probe_depth_range import summarize_depth_samples


def test_summarize_suggests_a_tight_window_for_a_shallow_scene():
    rng = np.random.default_rng(0)
    samples = [rng.uniform(3.0, 5.0, size=2000).astype(np.float32) for _ in range(5)]
    r = summarize_depth_samples(samples, max_depth=10.0)
    assert 2.5 < r["suggested_lo"] < 3.2
    assert 4.8 < r["suggested_hi"] < 5.5
    assert r["suggested_lo"] < r["median"] < r["suggested_hi"]
    assert r["lo_saturation"] == 0.0 and r["hi_saturation"] == 0.0   # nothing pinned


def test_summarize_flags_saturation_when_pixels_pile_up_at_the_edges():
    rng = np.random.default_rng(1)
    # half the pixels pinned at (near) 0, half at (near) max_depth -- classic
    # out-of-domain saturation signature
    lo_pile = np.full(1000, 0.01, np.float32)
    hi_pile = np.full(1000, 9.99, np.float32)
    r = summarize_depth_samples([lo_pile, hi_pile], max_depth=10.0)
    assert r["lo_saturation"] > 0.4 and r["hi_saturation"] > 0.4


def test_summarize_n_pixels_and_range_sanity():
    samples = [np.array([1.0, 2.0, 3.0], np.float32), np.array([4.0, 5.0], np.float32)]
    r = summarize_depth_samples(samples, max_depth=10.0)
    assert r["n_pixels"] == 5
    assert r["min"] == pytest.approx(1.0) and r["max"] == pytest.approx(5.0)
    assert r["suggested_lo"] <= r["p2"] and r["suggested_hi"] >= r["p98"]


def test_probe_depth_range_module_imports():
    import importlib
    importlib.import_module("scripts.probe_depth_range")
