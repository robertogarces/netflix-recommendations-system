"""Tests for precision_recall_at_k (src.evaluate).

The function is fully vectorized (lexsort + np.add.reduceat), which is fast but
easy to get subtly wrong. These tests pin it against a plain per-user reference
implementation plus a few hand-checked cases, so a future refactor of the
vectorized path cannot silently change the metric.
"""

import numpy as np
import pytest

from src.evaluate import precision_recall_at_k


def _reference(user_ids, true_ratings, est_ratings, k, threshold):
    """Obvious per-user oracle: for each user, rank their items by estimated
    rating, take the top K, count how many are relevant. Slow but clearly correct.

    Mirrors two choices of the real implementation so the two are comparable:
    precision divides by a fixed K, and users with no relevant item are dropped.
    """
    precisions, recalls = [], []
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        true_u = true_ratings[mask]
        est_u = est_ratings[mask]

        order = np.argsort(-est_u, kind="stable")   # best estimate first
        relevant = true_u[order] >= threshold
        n_relevant = int(relevant.sum())
        if n_relevant == 0:
            continue                                  # recall undefined -> excluded

        hits = int(relevant[:k].sum())
        precisions.append(hits / k)
        recalls.append(hits / n_relevant)

    return float(np.mean(precisions)), float(np.mean(recalls))


def test_matches_docstring_worked_example():
    """The k=1 example spelled out in precision_recall_at_k's docstring."""
    user_ids = np.array([1, 1, 2])
    est = np.array([4.8, 3.0, 4.0])
    true = np.array([5, 5, 2])

    precision, recall = precision_recall_at_k(user_ids, true, est, k=1, threshold=4.0)

    # User 1: top-1 is relevant -> precision 1/1, recall 1/2.
    # User 2: no relevant item -> dropped. So the means are over user 1 only.
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [1, 5, 10])
@pytest.mark.parametrize("threshold", [3.0, 4.0, 4.5])
def test_matches_reference_oracle(seed, k, threshold):
    """Vectorized result == naive per-user result across random inputs.

    Estimated ratings are continuous (rng.uniform) so there are no ties to make
    the ranking ambiguous; ties get their own dedicated test below.
    """
    rng = np.random.default_rng(seed)
    n = 500
    user_ids = rng.integers(0, 30, size=n)            # ~30 users, many repeats
    true_ratings = rng.integers(1, 6, size=n).astype(float)
    est_ratings = rng.uniform(1.0, 5.0, size=n)

    got = precision_recall_at_k(user_ids, true_ratings, est_ratings, k, threshold)
    expected = _reference(user_ids, true_ratings, est_ratings, k, threshold)

    assert got[0] == pytest.approx(expected[0])
    assert got[1] == pytest.approx(expected[1])


def test_user_with_no_relevant_items_is_excluded():
    """A user whose every true rating is below threshold must not count."""
    user_ids = np.array([1, 2, 2])
    est = np.array([5.0, 5.0, 1.0])
    true = np.array([5, 1, 1])

    precision, recall = precision_recall_at_k(user_ids, true, est, k=1, threshold=4.0)

    # Only user 1 contributes (user 2 has no relevant item): 1/1 and 1/1.
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_precision_divides_by_fixed_k():
    """Precision@K uses a fixed K denominator, even if the user has < K items."""
    user_ids = np.array([1, 1])
    est = np.array([5.0, 4.5])
    true = np.array([5, 5])

    precision, recall = precision_recall_at_k(user_ids, true, est, k=10, threshold=4.0)

    # 2 relevant items, both surfaced: precision = 2/10, recall = 2/2.
    assert precision == pytest.approx(0.2)
    assert recall == pytest.approx(1.0)


def test_ties_in_score_break_by_input_order():
    """Equal estimates keep their input order (stable sort), which decides top-K."""
    user_ids = np.array([1, 1])
    est = np.array([4.0, 4.0])      # tie
    true = np.array([5, 1])         # only the first item is relevant

    precision, recall = precision_recall_at_k(user_ids, true, est, k=1, threshold=4.0)

    # Stable order keeps item 0 first, so the single top item is the relevant one.
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
