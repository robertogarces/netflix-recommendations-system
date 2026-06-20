"""Tests for top_k_for_user (src.recommend).

top_k_for_user turns a user's per-item scores into a ranked top-K: it scores the
catalog (src.model.score_all_items), masks items the user has already seen to
-inf, picks the top K with argpartition, and decodes inner ids back to raw
movie_ids. Rather than compare to a full-sort oracle (fragile when clipped scores
tie at the rating-scale boundary), we assert the *properties* that define a
correct top-K — which hold regardless of how ties are broken.
"""

import numpy as np
import pytest

from src.model import item_raw_ids, score_all_items
from src.recommend import top_k_for_user


def _maps(algo):
    """(idx2raw, raw2inner) — the inner<->raw item-id maps top_k_for_user needs."""
    return item_raw_ids(algo), algo.trainset._raw2inner_id_items


def _assert_valid_topk(ids, scores, scores_full, raw2inner, expected_len, seen=frozenset()):
    """Assert (ids, scores) is a correct top-K over scores_full, minus `seen`."""
    assert len(ids) == expected_len
    assert np.all(np.diff(scores) <= 0)                        # sorted descending

    returned = {raw2inner[int(m)] for m in ids}
    assert len(returned) == len(ids)                           # no duplicate items
    assert seen.isdisjoint({int(m) for m in ids})              # already-seen items excluded

    # Decoding is correct: each returned score is that item's own score.
    for m, s in zip(ids, scores, strict=True):
        assert s == pytest.approx(scores_full[raw2inner[int(m)]])

    # Frontier: nothing we left out outscores what we kept (>= is tie-robust).
    kept = returned | {raw2inner[m] for m in seen if m in raw2inner}
    left_out = [s for j, s in enumerate(scores_full) if j not in kept]
    if left_out:
        assert scores.min() >= max(left_out) - 1e-9


def test_warm_user_returns_correct_topk(tiny_model):
    algo, scale = tiny_model
    idx2raw, raw2inner = _maps(algo)
    scores_full, _ = score_all_items(algo, 0, scale)

    ids, scores, segment = top_k_for_user(algo, 0, 5, frozenset(), idx2raw, raw2inner, scale)

    assert segment == "warm"
    _assert_valid_topk(ids, scores, scores_full, raw2inner, expected_len=5)


def test_seen_items_are_excluded(tiny_model):
    algo, scale = tiny_model
    idx2raw, raw2inner = _maps(algo)
    scores_full, _ = score_all_items(algo, 0, scale)

    # Mark the unmasked top-3 as already seen; they must not come back.
    top_ids, _, _ = top_k_for_user(algo, 0, 3, frozenset(), idx2raw, raw2inner, scale)
    seen = {int(m) for m in top_ids}

    ids, scores, _ = top_k_for_user(algo, 0, 5, seen, idx2raw, raw2inner, scale)

    _assert_valid_topk(ids, scores, scores_full, raw2inner, expected_len=5, seen=seen)


def test_k_larger_than_catalog_returns_all_items(tiny_model):
    algo, scale = tiny_model
    idx2raw, raw2inner = _maps(algo)
    n_items = algo.trainset.n_items
    scores_full, _ = score_all_items(algo, 0, scale)

    ids, scores, _ = top_k_for_user(algo, 0, n_items + 50, frozenset(), idx2raw, raw2inner, scale)

    _assert_valid_topk(ids, scores, scores_full, raw2inner, expected_len=n_items)


def test_cold_user_still_gets_recommendations(tiny_model):
    algo, scale = tiny_model
    idx2raw, raw2inner = _maps(algo)
    scores_full, segment = score_all_items(algo, 9999, scale)
    assert segment == "cold"

    ids, scores, segment2 = top_k_for_user(algo, 9999, 5, frozenset(), idx2raw, raw2inner, scale)

    assert segment2 == "cold"
    _assert_valid_topk(ids, scores, scores_full, raw2inner, expected_len=5)
