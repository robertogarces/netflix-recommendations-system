"""Recommendation stage: trained checkpoint → top-K movie recommendations.

Two modes:
- batch (default):       precompute top-K for every user the model knows and
                         save artifacts/recommendations.parquet
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

from config.paths import PROCESSED_DATA_PATH, RAW_DATA_PATH, ARTIFACTS_PATH, MODELS_PATH, CONFIG_PATH
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


def run_batch(model, checkpoint, k, batch_size, num_items, device, rating_scale):
    user2idx, item2idx = checkpoint["user2idx"], checkpoint["item2idx"]
    idx2movie = {idx: mid for mid, idx in item2idx.items()}
    titles = load_movie_titles()

    logger.info(f"Loading seen items for {len(user2idx):,} users ...")
    seen_by_user = load_seen_items(user2idx.keys(), user2idx, item2idx, device)

    # Stable order: recommend for users in ascending index order
    customer_ids = [cid for cid, _ in sorted(user2idx.items(), key=lambda kv: kv[1])]
    user_indices = list(range(len(customer_ids)))

    logger.info(f"Scoring top-{k} recommendations ...")
    rec_items, rec_scores = recommend_users(
        model, user_indices, seen_by_user, k, batch_size, num_items, device, rating_scale
    )

    # Flatten (users × k) into a tidy long table
    n_users = len(customer_ids)
    customer_col = np.repeat(customer_ids, k)
    rank_col     = np.tile(np.arange(1, k + 1), n_users)
    movie_col    = np.array([idx2movie[i] for i in rec_items.ravel()])
    out = pd.DataFrame({
        "customer_id": customer_col,
        "rank":        rank_col,
        "movie_id":    movie_col,
        "title":       [titles.get(m, "<unknown>") for m in movie_col],
        "score":       rec_scores.ravel(),
    })

    output_path = ARTIFACTS_PATH / "recommendations.parquet"
    out.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(out):,} recommendations ({n_users:,} users × {k}) to {output_path}")


def run_inspect(model, checkpoint, user_id, k, num_items, device, rating_scale):
    user2idx, item2idx = checkpoint["user2idx"], checkpoint["item2idx"]
    idx2movie = {idx: mid for mid, idx in item2idx.items()}
    titles = load_movie_titles()

    if user_id in user2idx:
        seen_by_user = load_seen_items([user_id], user2idx, item2idx, device)
        rec_items, rec_scores = recommend_users(
            model, [user2idx[user_id]], seen_by_user, k, 1, num_items, device, rating_scale
        )
        rec_items, rec_scores = rec_items[0], rec_scores[0]
        header = f"Top-{k} for user {user_id} (personalized)"
    else:
        rec_items, rec_scores = cold_start_topk(model, k, num_items, device)
        header = f"Top-{k} for user {user_id} (cold start — popularity fallback)"

    print(f"\n{header}\n" + "-" * len(header))
    for rank, (item_idx, score) in enumerate(zip(rec_items, rec_scores), start=1):
        movie_id = idx2movie[item_idx]
        print(f"{rank:>2}. {titles.get(movie_id, '<unknown>'):<45} (movie_id={movie_id}, score={score:.3f})")
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
        run_batch(model, checkpoint, k, rec_cfg["user_batch_size"], num_items, device, rating_scale)


if __name__ == "__main__":
    main()
