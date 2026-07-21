"""Phase 3 -- Occlusion-Aware Depth Refinement (paper Sec. 4.2.2, Eq. 8-12).

This package
composes the frozen Phase-2 instance decoder and the (fine-tuned) Phase-1
depth branch, adds the Occlusion Pair Relation Reasoning head ``Phi_o``, and
produces occlusion-corrected per-instance + dense metric depth.
"""
