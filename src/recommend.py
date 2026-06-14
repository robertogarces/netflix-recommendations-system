"""Recommendation stage: fitted SVD model → top-K movie recommendations.

Two modes:
- batch (default):       recommend top-K for a sample of the users in Netflix's
                         qualifying.txt and save outputs/recommendations.parquet
- inspect (--user-id X): print top-K for a single user, enriched with titles

Scoring is identical in both modes — only the trigger and output differ, so an
online API would be a thin wrapper over `recommend_single_user`.

Items the user has already rated are filtered out. Cold users (not in the
trainset) fall back to popularity ranking by learned item bias — exactly what
the model predicts for an unknown user.
"""

import argparse
import logging
import yaml
import numpy as np
import polars as pl
import pandas as pd

from config.paths import PROCESSED_DATA_PATH, RAW_DATA_PATH, OUTPUTS_PATH, MODELS_PATH, CONFIG_PATH
from src.model import load_model, score_all_items, item_raw_ids

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_movie_titles() -> dict[int, str]:
    """movie_id → title. Titles may contain commas, so split on the first two
    separators only (the rest is the title)."""
    titles = {}
    with open(RAW_DATA_PATH / "movie_titles.csv", encoding="latin1") as f:
        for line in f:
            movie_id, _year, title = line.rstrip("\n").split(",", 2)
            titles[int(movie_id)] = title
    return titles


def load_qualifying_users(sample_fraction: float, seed: int) -> list[int]:
    """Unique customer_ids from Netflix's qualifying.txt, sampled for legibility.

    qualifying.txt interleaves "movie_id:" header lines with "customer_id,date"
    rows (no ratings). We only need the distinct users to recommend for.
    """
    customer_ids = set()
    with open(RAW_DATA_PATH / "qualifying.txt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith(":"):
                continue
            customer_ids.add(int(line.split(",", 1)[0]))

    ids = np.array(sorted(customer_ids))
    if sample_fraction < 1.0:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=int(len(ids) * sample_fraction), replace=False)
    return sorted(int(c) for c in ids)


def load_seen_items(customer_ids) -> dict[int, set]:
    """Map each user to the set of movie_ids they have already rated, for filtering.

    Lazily scans the processed parquet and keeps only the requested users, so the
    inspect mode (one user) stays cheap and batch mode pays the scan once.
    """
    seen = (
        pl.scan_parquet(PROCESSED_DATA_PATH / "processed_data.parquet")
        .filter(pl.col("customer_id").is_in(list(customer_ids)))
        .select(["customer_id", "movie_id"])
        .collect()
    )
    return {
        cid: set(group["movie_id"].to_list())
        for cid, group in seen.group_by("customer_id")
    }


def top_k_for_user(algo, user_id, k, seen, idx2raw, raw2inner, rating_scale):
    """Top-K (movie_ids, scores, segment) for one user, masking already-seen items."""
    scores, segment = score_all_items(algo, user_id, rating_scale)  # by inner item id

    # Mask items the user already rated by pushing their score to -inf.
    if seen:
        inner = [raw2inner[m] for m in seen if m in raw2inner]
        if inner:
            scores[inner] = -np.inf

    # argpartition grabs the top-k cheaply, then we sort just those k by score.
    k = min(k, len(scores))
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(-scores[top])]
    return idx2raw[top], scores[top], segment


def _build_records(user_id, movie_ids, scores, titles, segment) -> pd.DataFrame:
    """Tidy long table of one user's recommendations."""
    return pd.DataFrame({
        "customer_id": user_id,
        "rank":        np.arange(1, len(movie_ids) + 1),
        "movie_id":    movie_ids,
        "title":       [titles.get(int(m), "<unknown>") for m in movie_ids],
        "score":       scores,
        "segment":     segment,
    })


def recommend_single_user(algo, user_id, k, rating_scale, titles=None, idx2raw=None):
    """Top-K for one user as a DataFrame, plus the segment ("warm"/"cold").

    Shared by the CLI inspect mode and the dashboard. titles/idx2raw can be passed
    in to avoid recomputing them per call.
    """
    titles  = titles if titles is not None else load_movie_titles()
    idx2raw = idx2raw if idx2raw is not None else item_raw_ids(algo)
    raw2inner = algo.trainset._raw2inner_id_items
    seen = load_seen_items([user_id]).get(user_id, set())

    movie_ids, scores, segment = top_k_for_user(
        algo, user_id, k, seen, idx2raw, raw2inner, rating_scale
    )
    df = pd.DataFrame({
        "rank":     range(1, len(movie_ids) + 1),
        "movie_id": movie_ids,
        "title":    [titles.get(int(m), "<unknown>") for m in movie_ids],
        "score":    scores,
    })
    return df, segment


def run_batch(algo, k, rating_scale, sample_fraction, seed):
    idx2raw    = item_raw_ids(algo)
    raw2inner  = algo.trainset._raw2inner_id_items
    users_seen = algo.trainset._raw2inner_id_users  # membership test = 'user is warm'
    titles     = load_movie_titles()

    users = load_qualifying_users(sample_fraction, seed)
    warm  = sum(u in users_seen for u in users)
    logger.info(f"Qualifying users sampled: {len(users):,} ({sample_fraction:.0%}) | warm: {warm:,}")

    seen_by_user = load_seen_items(users)
    frames = []
    for user_id in users:
        seen = seen_by_user.get(user_id, set())
        movie_ids, scores, segment = top_k_for_user(
            algo, user_id, k, seen, idx2raw, raw2inner, rating_scale
        )
        frames.append(_build_records(user_id, movie_ids, scores, titles, segment))

    out = pd.concat(frames, ignore_index=True).sort_values(["customer_id", "rank"])
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_PATH / "recommendations.parquet"
    out.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(out):,} recommendations ({len(users):,} users × {k}) to {output_path}")


def run_inspect(algo, user_id, k, rating_scale):
    df, segment = recommend_single_user(algo, user_id, k, rating_scale)
    label = "personalized" if segment == "warm" else "cold start — popularity fallback"
    header = f"Top-{k} for user {user_id} ({label})"

    print(f"\n{header}\n" + "-" * len(header))
    for row in df.itertuples(index=False):
        print(f"{row.rank:>2}. {row.title:<45} (movie_id={row.movie_id}, score={row.score:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate movie recommendations from a trained SVD model.")
    parser.add_argument("--user-id", type=int, default=None,
                        help="Inspect recommendations for a single customer_id. Omit to run batch precompute.")
    parser.add_argument("--k", type=int, default=None, help="Number of recommendations (overrides config).")
    args = parser.parse_args()

    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    rec_cfg      = config["recommend"]
    rating_scale = tuple(config["data"]["rating_scale"])
    k            = args.k or rec_cfg["top_k"]

    algo = load_model(MODELS_PATH / "svd_model.pkl")
    ts = algo.trainset
    logger.info(f"Loaded model: {ts.n_users:,} users, {ts.n_items:,} items")

    if args.user_id is not None:
        run_inspect(algo, args.user_id, k, rating_scale)
    else:
        run_batch(algo, k, rating_scale,
                  sample_fraction=rec_cfg["qualifying_sample_fraction"], seed=config["seed"])


if __name__ == "__main__":
    main()
