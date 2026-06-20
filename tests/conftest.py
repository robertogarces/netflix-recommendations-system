"""Shared pytest fixtures.

`tiny_model` fits a small Surprise SVD on a handful of synthetic ratings, so the
scoring tests (estimate_pairs, and later score_all_items / item_raw_ids) have a
real fitted model with predictable known/unknown ids — without touching the
100M-row dataset.
"""

import numpy as np
import pandas as pd
import pytest
from surprise import SVD, Dataset, Reader

RATING_SCALE = (1.0, 5.0)


@pytest.fixture(scope="session")
def tiny_model():
    """(algo, rating_scale): a deterministic SVD fitted on ~100 synthetic ratings.

    Users 0..19 and items 0..14 are known; ids like 9999 are guaranteed unknown,
    for the cold-start paths. Read-only — a test that needs to mutate the model
    should deepcopy it first.
    """
    rng = np.random.default_rng(0)
    n_users, n_items = 20, 15
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=int(rng.integers(3, 8)), replace=False)
        for i in items:
            rows.append((u, int(i), float(rng.integers(1, 6))))
    df = pd.DataFrame(rows, columns=["user", "item", "rating"])

    reader = Reader(rating_scale=RATING_SCALE)
    trainset = Dataset.load_from_df(df, reader).build_full_trainset()
    algo = SVD(n_factors=5, n_epochs=10, random_state=42)
    algo.fit(trainset)
    return algo, RATING_SCALE
