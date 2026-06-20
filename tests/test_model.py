"""Tests for the scoring helpers in src.model.

estimate_pairs is the vectorized, chunked equivalent of looping
surprise.SVD.predict over (user, item) pairs, so Surprise's own predict is the
oracle. The chunking-invariance test is the regression guard for the OOM fix
that introduced chunking in the first place.
"""

import copy

import numpy as np
import pytest

from src.model import estimate_pairs, item_raw_ids


def test_matches_surprise_predict(tiny_model):
    """estimate_pairs == surprise predict, over a mix of known and unknown ids."""
    algo, scale = tiny_model
    rng = np.random.default_rng(1)
    n = 200
    users = rng.integers(0, 22, size=n)   # 20, 21 are unknown users
    items = rng.integers(0, 17, size=n)   # 15, 16 are unknown items

    got = estimate_pairs(algo, users, items, scale)
    expected = np.array([algo.predict(int(u), int(i)).est for u, i in zip(users, items, strict=True)])

    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_chunking_is_invariant(tiny_model):
    """Chunked scoring equals single-pass scoring — the OOM fix must not change results."""
    algo, scale = tiny_model
    rng = np.random.default_rng(2)
    n = 50
    users = rng.integers(0, 22, size=n)
    items = rng.integers(0, 17, size=n)

    one_pass = estimate_pairs(algo, users, items, scale, chunk_size=10_000)
    chunked = estimate_pairs(algo, users, items, scale, chunk_size=7)  # 8 chunks over 50 rows

    np.testing.assert_array_equal(one_pass, chunked)


def test_unknown_ids_fall_back_like_surprise(tiny_model):
    """Cold-start: drop the term we lack, fall back toward global_mean + known bias."""
    algo, scale = tiny_model
    ts = algo.trainset
    gm = ts.global_mean

    u_raw = 0
    u_inner = ts._raw2inner_id_users[u_raw]
    i_raw = next(iter(ts._raw2inner_id_items))
    i_inner = ts._raw2inner_id_items[i_raw]

    both_unknown = estimate_pairs(algo, [9999], [9999], scale)[0]
    user_only = estimate_pairs(algo, [u_raw], [9999], scale)[0]
    item_only = estimate_pairs(algo, [9999], [i_raw], scale)[0]

    assert both_unknown == pytest.approx(np.clip(gm, *scale))
    assert user_only == pytest.approx(np.clip(gm + algo.bu[u_inner], *scale))
    assert item_only == pytest.approx(np.clip(gm + algo.bi[i_inner], *scale))


def test_clamps_scores_to_rating_scale(tiny_model):
    """Scores are clipped to the rating scale, like Surprise's clamped predictions."""
    algo, scale = tiny_model
    a = copy.deepcopy(algo)               # don't mutate the shared session fixture

    u_raw = 0
    i_raw = next(iter(a.trainset._raw2inner_id_items))
    a.bu[a.trainset._raw2inner_id_users[u_raw]] = 100.0   # blow past the ceiling

    out = estimate_pairs(a, [u_raw], [i_raw], scale)[0]
    assert out == pytest.approx(scale[1])


def test_item_raw_ids_inverts_the_inner_mapping(tiny_model):
    """item_raw_ids is the exact inverse of the trainset's raw->inner item map.

    Independent on purpose: top_k_for_user's tests use this array on both sides of
    their checks, so they would not catch a bug in it; here it is cross-checked
    against the trainset's own map.
    """
    algo, _ = tiny_model
    idx2raw = item_raw_ids(algo)
    raw2inner = algo.trainset._raw2inner_id_items

    assert idx2raw.shape == (algo.trainset.n_items,)
    assert idx2raw.dtype == np.int64

    for inner in range(algo.trainset.n_items):        # inner -> raw -> inner
        assert raw2inner[int(idx2raw[inner])] == inner

    for raw, inner in raw2inner.items():              # raw -> inner -> raw
        assert int(idx2raw[inner]) == raw
