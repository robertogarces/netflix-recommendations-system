"""Recommendation stage: trained checkpoint → top-K movie recommendations.

Two modes:
- batch (default):       recommend top-K for a sample of the users in Netflix's
                         qualifying.txt and save outputs/recommendations.parquet
- inspect (--user-id X): print top-K for a single user, enriched with titles

Scoring logic is identical in both modes — only the trigger and output differ,
so an online API would be a thin wrapper over `recommend_users`.

Items the user has already rated are filtered out. Cold users (not in the
checkpoint vocabulary) fall back to popularity ranking by learned item bias —
exactly what the model predicts for an unknown user.
"""

import argparse
import logging
import yaml
import numpy as np
import polars as pl
import pandas as pd
import torch

from config.paths import PROCESSED_DATA_PATH, RAW_DATA_PATH, OUTPUTS_PATH, MODELS_PATH, CONFIG_PATH
from src.model import get_device, load_checkpoint

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

    qualifying.txt is the set Netflix released for submission: header lines
    ("movie_id:") interleaved with "customer_id,date" rows (no ratings). We
    only need the distinct users to recommend for.
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


def load_seen_items(customer_ids, user2idx: dict, item2idx: dict, device) -> dict[int, torch.Tensor]:
    """Map each user's rated movies to a tensor of item indices, for filtering.

    Lazily scans the processed parquet and keeps only the requested users, so
    the inspect mode (one user) stays cheap and batch mode pays once.
    """
    seen = (
        pl.scan_parquet(PROCESSED_DATA_PATH / "processed_data.parquet")
        .filter(pl.col("customer_id").is_in(list(customer_ids)))
        .select(["customer_id", "movie_id"])
        .collect()
        .to_pandas()
    )
    seen["user_idx"] = seen["customer_id"].map(user2idx)
    seen["item_idx"] = seen["movie_id"].map(item2idx)
    seen = seen.dropna(subset=["user_idx", "item_idx"])

    return {
        int(uidx): torch.tensor(group.to_numpy(), dtype=torch.long, device=device)
        for uidx, group in seen.groupby("user_idx")["item_idx"]
    }


@torch.no_grad()
def recommend_users(
    model, user_indices, seen_by_user, k, batch_size, num_items, device, rating_scale
) -> tuple[np.ndarray, np.ndarray]:
    """Score the full catalog for each user, mask seen items, return top-K.

    At ~17.7k items, scoring the whole catalog per user is cheap; two-stage
    candidate generation only pays off at millions of items.

    Returns (item_idx, scores), both shaped (len(user_indices), k).
    """
    item_ids = torch.arange(num_items, device=device)
    rec_items, rec_scores = [], []

    for start in range(0, len(user_indices), batch_size):
        batch = user_indices[start:start + batch_size]
        b = len(batch)

        users = torch.tensor(batch, device=device).repeat_interleave(num_items)
        items = item_ids.repeat(b)
        scores = model(users, items).clamp(*rating_scale).view(b, num_items)

        for row, uidx in enumerate(batch):
            seen = seen_by_user.get(int(uidx))
            if seen is not None:
                scores[row, seen] = float("-inf")

        top_scores, top_idx = torch.topk(scores, k, dim=1)
        rec_items.append(top_idx.cpu().numpy())
        rec_scores.append(top_scores.cpu().numpy())

    return np.concatenate(rec_items), np.concatenate(rec_scores)


@torch.no_grad()
def cold_start_topk(model, k, num_items, device) -> tuple[np.ndarray, np.ndarray]:
    """Popularity ranking for an unknown user: global_mean + item_bias."""
    item_ids = torch.arange(num_items, device=device)
    scores = model.global_mean + model.item_bias(item_ids).squeeze(-1)
    top_scores, top_idx = torch.topk(scores, k)
    return top_idx.cpu().numpy(), top_scores.cpu().numpy()


def _build_records(customer_ids, rec_items, rec_scores, idx2movie, titles, k, segment) -> pd.DataFrame:
    """Flatten (users × k) score arrays into a tidy long table."""
    n = len(customer_ids)
    movie_col = np.array([idx2movie[i] for i in rec_items.ravel()])
    return pd.DataFrame({
        "customer_id": np.repeat(customer_ids, k),
        "rank":        np.tile(np.arange(1, k + 1), n),
        "movie_id":    movie_col,
        "title":       [titles.get(m, "<unknown>") for m in movie_col],
        "score":       rec_scores.ravel(),
        "segment":     segment,
    })


def run_batch(model, checkpoint, k, batch_size, num_items, device, rating_scale, sample_fraction, seed):
    user2idx, item2idx = checkpoint["user2idx"], checkpoint["item2idx"]
    idx2movie = {idx: mid for mid, idx in item2idx.items()}
    titles = load_movie_titles()

    users = load_qualifying_users(sample_fraction, seed)
    warm = [u for u in users if u in user2idx]
    cold = [u for u in users if u not in user2idx]
    logger.info(f"Qualifying users sampled: {len(users):,} ({sample_fraction:.0%}) "
                f"| warm: {len(warm):,} | cold: {len(cold):,}")

    frames = []

    if warm:
        logger.info(f"Loading seen items for {len(warm):,} warm users ...")
        seen_by_user = load_seen_items(warm, user2idx, item2idx, device)
        warm_indices = [user2idx[u] for u in warm]
        logger.info("Scoring personalized recommendations ...")
        rec_items, rec_scores = recommend_users(
            model, warm_indices, seen_by_user, k, batch_size, num_items, device, rating_scale
        )
        frames.append(_build_records(warm, rec_items, rec_scores, idx2movie, titles, k, "warm"))

    if cold:
        # Unknown users share one popularity list (global_mean + item_bias)
        cold_items, cold_scores = cold_start_topk(model, k, num_items, device)
        rec_items  = np.tile(cold_items, (len(cold), 1))
        rec_scores = np.tile(cold_scores, (len(cold), 1))
        frames.append(_build_records(cold, rec_items, rec_scores, idx2movie, titles, k, "cold"))

    out = pd.concat(frames, ignore_index=True).sort_values(["customer_id", "rank"])

    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_PATH / "recommendations.parquet"
    out.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(out):,} recommendations ({len(users):,} users × {k}) to {output_path}")


def recommend_single_user(
    model, checkpoint, user_id, k, num_items, device, rating_scale
) -> tuple[pd.DataFrame, str]:
    """Top-K for one user as a DataFrame, plus the segment ("warm"/"cold").

    Shared by the CLI inspect mode and the dashboard — both wrap this.
    """
    user2idx, item2idx = checkpoint["user2idx"], checkpoint["item2idx"]
    idx2movie = {idx: mid for mid, idx in item2idx.items()}
    titles = load_movie_titles()

    if user_id in user2idx:
        seen_by_user = load_seen_items([user_id], user2idx, item2idx, device)
        rec_items, rec_scores = recommend_users(
            model, [user2idx[user_id]], seen_by_user, k, 1, num_items, device, rating_scale
        )
        rec_items, rec_scores = rec_items[0], rec_scores[0]
        segment = "warm"
    else:
        rec_items, rec_scores = cold_start_topk(model, k, num_items, device)
        segment = "cold"

    movie_ids = [idx2movie[i] for i in rec_items]
    df = pd.DataFrame({
        "rank":     range(1, len(movie_ids) + 1),
        "movie_id": movie_ids,
        "title":    [titles.get(m, "<unknown>") for m in movie_ids],
        "score":    rec_scores,
    })
    return df, segment


def run_inspect(model, checkpoint, user_id, k, num_items, device, rating_scale):
    df, segment = recommend_single_user(model, checkpoint, user_id, k, num_items, device, rating_scale)
    label = "personalized" if segment == "warm" else "cold start — popularity fallback"
    header = f"Top-{k} for user {user_id} ({label})"

    print(f"\n{header}\n" + "-" * len(header))
    for row in df.itertuples(index=False):
        print(f"{row.rank:>2}. {row.title:<45} (movie_id={row.movie_id}, score={row.score:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate movie recommendations from a trained NCF checkpoint.")
    parser.add_argument("--user-id", type=int, default=None,
                        help="Inspect recommendations for a single customer_id. Omit to run batch precompute.")
    parser.add_argument("--k", type=int, default=None, help="Number of recommendations (overrides config).")
    args = parser.parse_args()

    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    rec_cfg      = config["recommend"]
    rating_scale = tuple(config["data"]["rating_scale"])
    k            = args.k or rec_cfg["top_k"]

    device = get_device()
    model, checkpoint = load_checkpoint(MODELS_PATH / "ncf_model.pt", device)
    num_items = len(checkpoint["item2idx"])
    logger.info(f"Loaded checkpoint: {len(checkpoint['user2idx']):,} users, {num_items:,} items | device: {device}")

    if args.user_id is not None:
        run_inspect(model, checkpoint, args.user_id, k, num_items, device, rating_scale)
    else:
        run_batch(
            model, checkpoint, k, rec_cfg["user_batch_size"], num_items, device, rating_scale,
            sample_fraction=rec_cfg["qualifying_sample_fraction"], seed=config["seed"],
        )


if __name__ == "__main__":
    main()
