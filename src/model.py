"""SVD recommender backed by the Surprise library.

A thin wrapper around surprise.SVD so the train / evaluate / recommend / dashboard
stages share one tiny model interface. Surprise learns Funk matrix factorization:

    ŷ(u, i) = global_mean + b_u + b_i + q_i · p_u

and stores the result as plain numpy arrays (pu, qi, bu, bi) plus the trainset's
global_mean and raw-id↔inner-id mappings. That lets us score in a couple of
vectorized numpy operations instead of Surprise's per-pair Python `predict` loop,
which matters when the test set has millions of rows.

Two id spaces — the one friction to keep in mind: the data speaks "raw" ids
(customer_id, movie_id), but Surprise indexes its factor arrays by compact
"inner" ids (0..n-1). So every scoring helper translates raw→inner on the way in
(_raw_to_inner) and inner→raw on the way out (item_raw_ids). The placeholder/mask
dance in estimate_pairs exists only to make that translation vectorizable.

Unknown users/items (cold start) degrade exactly as Surprise's own estimator
does: drop whichever term we lack and fall back toward global_mean + known bias.
"""

import logging

import numpy as np
import pandas as pd
from surprise import dump

logger = logging.getLogger(__name__)


def save_model(path, algo) -> None:
    """Persist a fitted Surprise algo (factors, biases and trainset) to disk."""
    dump.dump(str(path), algo=algo)


def load_model(path):
    """Load a fitted Surprise algo saved by save_model."""
    _, algo = dump.load(str(path))
    return algo


# Surprise keeps raw→inner id maps as these (stable, internal) trainset dicts;
# we read them directly to vectorize what predict() does one pair at a time.
def _raw_to_inner(trainset, raw_ids: np.ndarray, kind: str) -> np.ndarray:
    """Map raw customer/movie ids to Surprise inner indices, NaN where unknown."""
    mapping = trainset._raw2inner_id_users if kind == "user" else trainset._raw2inner_id_items
    return pd.Series(raw_ids).map(mapping).to_numpy()  # float array with NaN for unknown


def estimate_pairs(algo, raw_users, raw_items, rating_scale, chunk_size: int = 2_000_000) -> np.ndarray:
    """Vectorized rating estimates for aligned (user, item) arrays.

    Mirrors surprise.SVD.estimate but over whole arrays: start from global_mean,
    add b_u where the user is known, add b_i where the item is known, and add the
    factor dot product only where BOTH are known. Result is clipped to the rating
    scale, matching how Surprise predictions are normally clamped.

    Processed in chunks of `chunk_size` rows: the factor gather algo.pu[users]
    builds an (n_rows, n_factors) array, so doing it in one shot at tens of
    millions of rows (a 50% sample is ~40M train rows) allocates tens of GB and
    OOMs. Chunking bounds peak memory to about chunk_size × n_factors floats.
    """
    ts = algo.trainset
    # The raw→inner mapping over the full arrays is cheap (one float per row);
    # only the factor gather below is memory-heavy, so that is what we chunk.
    u_inner = _raw_to_inner(ts, np.asarray(raw_users), "user")
    i_inner = _raw_to_inner(ts, np.asarray(raw_items), "item")

    out = np.empty(len(u_inner), dtype=np.float64)
    for start in range(0, len(u_inner), chunk_size):
        end = start + chunk_size
        uc, ic = u_inner[start:end], i_inner[start:end]

        uk = ~np.isnan(uc)             # user-known mask
        ik = ~np.isnan(ic)             # item-known mask
        # Unknown ids get placeholder index 0 so the fancy-indexing below never
        # errors on NaN; the uk/ik masks guarantee those placeholder rows are
        # never actually read (we only add b_u/b_i/factors where known).
        ui = np.where(uk, uc, 0).astype(np.int64)
        ii = np.where(ik, ic, 0).astype(np.int64)

        est = np.full(len(uc), ts.global_mean, dtype=np.float64)
        est[uk] += algo.bu[ui[uk]]
        est[ik] += algo.bi[ii[ik]]
        both = uk & ik
        # einsum 'ij,ij->i' is a per-row dot product of the two factor matrices.
        est[both] += np.einsum("ij,ij->i", algo.pu[ui[both]], algo.qi[ii[both]])
        out[start:end] = est

    return np.clip(out, *rating_scale)


def score_all_items(algo, raw_uid, rating_scale) -> tuple[np.ndarray, str]:
    """Score the whole catalog for one user → (scores by inner item id, segment).

    Warm user: global_mean + b_u + b_i + q_i·p_u for every item.
    Cold user (not in trainset): global_mean + b_i — a popularity ranking, which
    is exactly what the model predicts when it knows nothing about the user.
    """
    ts = algo.trainset
    scores = ts.global_mean + algo.bi                      # (n_items,) popularity baseline
    if raw_uid in ts._raw2inner_id_users:
        u = ts._raw2inner_id_users[raw_uid]
        scores = scores + algo.bu[u] + algo.qi @ algo.pu[u]  # add personalization
        segment = "warm"
    else:
        segment = "cold"
    return np.clip(scores, *rating_scale), segment


def item_raw_ids(algo) -> np.ndarray:
    """Array mapping each inner item index → its raw movie_id (for decoding top-K)."""
    ts = algo.trainset
    out = np.empty(ts.n_items, dtype=np.int64)
    for raw, inner in ts._raw2inner_id_items.items():
        out[inner] = raw
    return out


def knows_user(algo, raw_uid) -> bool:
    """Whether the user was seen in training (has a learned factor vector)."""
    return raw_uid in algo.trainset._raw2inner_id_users
