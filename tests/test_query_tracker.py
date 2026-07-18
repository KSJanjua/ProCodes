"""Unit tests for QueryInstanceTracker (videodepth/models/query_tracker.py) --
the Mask2Former-VIS identity concept via per-frame query-embedding matching
(MinVIS-style). The contracts that justify it over mask-IoU tracking:

  * identity survives arbitrary QUERY-ORDER shuffling between frames (the
    core video-Mask2Former property: the embedding IS the identity);
  * identity follows the PERSON through a crossing, even when position/IoU
    points the wrong way;
  * a person re-identifies after a long occlusion, from a new position --
    impossible for IoU-only tracking;
  * with embeds=None it degrades gracefully to IoU matching.
"""

from __future__ import annotations

import numpy as np

from videodepth.models.query_tracker import QueryInstanceTracker


def _rect(H, W, r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


def _embed(axis, dim=32):
    """Deterministic, mutually-orthogonal 'person' embeddings (cosine = 0
    between people, 1 with themselves) -- models the separation real
    Mask2Former query embeddings have between different people, without the
    accidental correlation random vectors can carry."""
    e = np.zeros(dim, np.float32)
    e[axis] = 1.0
    return e


H = W = 48
E_A, E_B = _embed(0), _embed(1)          # two people's (distinct) query embeddings


def test_ids_survive_query_order_shuffle():
    """Frame 2 presents the same people in the OPPOSITE detection order --
    the tracker must keep each person's id (the raw query index would swap)."""
    tr = QueryInstanceTracker(min_hits=1)
    mA, mB = _rect(H, W, 0, 40, 2, 18), _rect(H, W, 0, 40, 26, 46)
    _, _, ids1 = tr.update([mA, mB], [2.0, 4.0], [E_A, E_B])
    _, deps2, ids2 = tr.update([mB, mA], [4.0, 2.0], [E_B, E_A])   # shuffled order
    a1, b1 = ids1
    # person A (embed E_A, depth 2.0) keeps id a1 despite arriving second
    assert set(ids2) == {a1, b1}
    assert deps2[ids2.index(a1)] == 2.0 and deps2[ids2.index(b1)] == 4.0


def test_identity_follows_person_through_crossing_not_position():
    """The two people swap positions. IoU votes for the WRONG pairing (each
    person's new mask overlaps the other's old spot); the embedding must win."""
    tr = QueryInstanceTracker(min_hits=1)
    left, right = _rect(H, W, 0, 48, 0, 22), _rect(H, W, 0, 48, 26, 48)
    _, _, ids1 = tr.update([left, right], [2.0, 4.0], [E_A, E_B])
    idA, idB = ids1
    # crossed: A now occupies (roughly) B's old region and vice versa
    _, deps2, ids2 = tr.update([right, left], [2.0, 4.0], [E_A, E_B])
    assert ids2 == [idA, idB]                      # identity stuck to the embedding
    assert deps2 == [2.0, 4.0]


def test_reidentification_after_long_occlusion():
    """Person B vanishes for 10 frames (beyond any short IoU memory), then
    reappears at a NEW position -- the embedding re-identifies them."""
    tr = QueryInstanceTracker(min_hits=1, max_age=30)
    mA = _rect(H, W, 0, 48, 0, 20)
    mB = _rect(H, W, 0, 48, 28, 48)
    _, _, ids = tr.update([mA, mB], [2.0, 4.0], [E_A, E_B])
    idB = ids[1]
    for _ in range(10):                            # B fully occluded
        tr.update([mA], [2.0], [E_A])
    mB_new = _rect(H, W, 0, 48, 10, 30)            # returns somewhere else
    _, _, ids_back = tr.update([mA, mB_new], [2.0, 4.0], [E_A, E_B])
    assert idB in ids_back                         # same person, same id


def test_expiry_after_max_age_gives_new_id():
    tr = QueryInstanceTracker(min_hits=1, max_age=3)
    mB = _rect(H, W, 0, 48, 28, 48)
    _, _, ids = tr.update([mB], [4.0], [E_B])
    idB = ids[0]
    for _ in range(5):                             # gone longer than max_age
        tr.update([], [], [])
    _, _, ids_back = tr.update([mB], [4.0], [E_B])
    assert ids_back and ids_back[0] != idB         # track expired -> fresh id


def test_iou_only_fallback_without_embeddings():
    tr = QueryInstanceTracker(min_hits=1)
    m1 = _rect(H, W, 0, 48, 0, 24)
    _, _, ids1 = tr.update([m1], [3.0], None)
    m2 = _rect(H, W, 0, 48, 2, 26)                 # same person, small shift
    _, _, ids2 = tr.update([m2], [3.0], None)
    assert ids2 == ids1                            # matched by IoU alone


def test_min_hits_suppresses_one_frame_detections():
    tr = QueryInstanceTracker(min_hits=2)
    mA = _rect(H, W, 0, 48, 0, 20)
    masks, _, _ = tr.update([mA], [2.0], [E_A])
    assert masks == []                             # not confirmed yet
    masks2, _, ids2 = tr.update([mA], [2.0], [E_A])
    assert len(masks2) == 1 and len(ids2) == 1     # confirmed on 2nd hit
