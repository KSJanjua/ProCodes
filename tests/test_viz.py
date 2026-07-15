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


def test_open_video_writer_always_produces_output():
    """Whatever encoders this OpenCV build has, open_video_writer must return
    a usable writer -- a real codec if available, else the PNG FrameDumpWriter
    fallback -- and writing frames must leave output on disk."""
    import tempfile
    from pathlib import Path
    from instancedepth.utils.viz import open_video_writer

    with tempfile.TemporaryDirectory() as td:
        writer, out_path = open_video_writer(Path(td) / "clip", fps=10.0, frame_wh=(64, 48))
        frame = np.zeros((48, 64, 3), np.uint8)
        for _ in range(3):
            writer.write(frame)
        writer.release()
        p = Path(out_path)
        if p.is_dir():                                   # FrameDumpWriter fallback
            assert len(list(p.glob("frame_*.png"))) == 3
        else:                                            # real encoded video
            assert p.exists() and p.stat().st_size > 0


def test_frame_dump_writer():
    import tempfile
    from pathlib import Path
    from instancedepth.utils.viz import FrameDumpWriter

    with tempfile.TemporaryDirectory() as td:
        w = FrameDumpWriter(Path(td) / "seq_frames", fps=15.0)
        assert w.isOpened()
        for _ in range(2):
            w.write(np.zeros((8, 8, 3), np.uint8))
        w.release()
        assert len(list((Path(td) / "seq_frames").glob("frame_*.png"))) == 2


def test_frame_dump_writer_stitches_with_system_ffmpeg():
    """With a stitch_target set, release() must produce a real video via the
    system ffmpeg (and clean up the PNGs); without ffmpeg on PATH it must keep
    the PNGs -- both behaviors asserted according to what this machine has."""
    import shutil
    import tempfile
    from pathlib import Path
    from instancedepth.utils.viz import FrameDumpWriter

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "clip.mp4"
        w = FrameDumpWriter(Path(td) / "clip_frames", fps=10.0, stitch_target=target)
        for i in range(4):
            w.write(np.full((32, 32, 3), i * 20, np.uint8))
        w.release()
        if shutil.which("ffmpeg"):
            assert target.exists() and target.stat().st_size > 0
            assert not (Path(td) / "clip_frames").exists()   # PNGs cleaned up
        else:
            assert len(list((Path(td) / "clip_frames").glob("frame_*.png"))) == 4


def test_open_frame_source_directory_input():
    """infer_video's universal escape hatch: a directory of image frames must
    work on any OpenCV build, in filename order, with the fallback fps."""
    import cv2
    import tempfile
    from pathlib import Path
    from scripts.infer_video import open_frame_source

    with tempfile.TemporaryDirectory() as td:
        for i in range(3):
            frame = np.full((8, 8, 3), i * 10, np.uint8)
            cv2.imwrite(str(Path(td) / f"frame_{i:03d}.png"), frame)
        frames, fps, total = open_frame_source(td, fps_fallback=12.0)
        got = list(frames)
        assert (fps, total, len(got)) == (12.0, 3, 3)
        assert [int(f[0, 0, 0]) for f in got] == [0, 10, 20]   # filename order


def test_open_frame_source_missing_path():
    from scripts.infer_video import open_frame_source
    try:
        open_frame_source("definitely/not/a/real/path.mp4", fps_fallback=30.0)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "does not exist" in str(e)
