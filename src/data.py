"""Shared data utilities for the train and evaluate stages.

Keeping sampling, splitting, and tensor construction in one place guarantees
both stages see the data in exactly the same way.
"""

import logging
import numpy as np
import pandas as pd
import torch

from config.paths import PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


def load_processed(sample_fraction: float, seed: int) -> pd.DataFrame:
    """Load the processed dataset, optionally sub-sampling a fraction of users.

    Why sample by user instead of by row?
    Random row sampling gives every user roughly `sample_fraction` of their
    ratings — a user with 200 ratings ends up with 20. Embeddings trained on
    20 ratings per user barely learn anything useful. By sampling *users*
    instead, each sampled user keeps their full history, so the embeddings
    have enough signal to learn meaningful representations. The trade-off is
    that we see fewer users total, but that's acceptable for a 10% sample.
    """
    df = pd.read_parquet(PROCESSED_DATA_PATH / "processed_data.parquet")
    if sample_fraction < 1.0:
        users = df["customer_id"].unique()
        rng = np.random.default_rng(seed)
        # rng.choice without replacement: pick sample_fraction% of distinct user IDs
        sampled = rng.choice(users, size=int(len(users) * sample_fraction), replace=False)
        df = df[df["customer_id"].isin(sampled)]
    logger.info(
        f"Loaded {len(df):,} rows | {df['customer_id'].nunique():,} users "
        f"({sample_fraction:.0%} user sample)"
    )
    return df


def temporal_split(
    df: pd.DataFrame, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way temporal split: train on the past, validate on the middle, test on the future.

    Why temporal and not random?
    A random split leaks future information into training (e.g. a user's
    December rating can be used to predict their January rating). Temporal
    splits simulate the real deployment scenario: the model trains on historical
    data and must generalize to future ratings it has never seen.

    The downside: users who appear for the first time near the test boundary
    end up with no training history — they become "cold users" in the test set.
    src.train flags these rows and src.evaluate reports their metrics separately.

    Split boundaries are date quantiles over the entire dataset (not per-user),
    so the train/val/test row fractions match val_size / test_size exactly.
    """
    dates = df["date"].sort_values()

    # Find the date at the quantile corresponding to the end of the training window.
    # Example with val_size=0.1, test_size=0.2: training covers the first 70% of dates.
    val_date  = dates.iloc[int(len(df) * (1 - val_size - test_size))]
    test_date = dates.iloc[int(len(df) * (1 - test_size))]

    train = df[df["date"] < val_date]
    val   = df[(df["date"] >= val_date) & (df["date"] < test_date)]
    test  = df[df["date"] >= test_date]
    return train, val, test


def add_index_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map raw customer/movie IDs to compact 0..N-1 indices for embedding lookups.

    Why compact indices?
    PyTorch Embedding tables are allocated as [num_entities, emb_size]. If we
    used raw customer_ids (up to ~2.6M in the full dataset), the embedding
    table would have 2.6M rows even when a 10% sample only touches ~26k users.
    That wastes ~100x memory and makes Adam's internal momentum/variance buffers
    proportionally larger. pd.factorize remaps IDs to 0..N-1, so the table is
    exactly as large as the number of entities actually seen in this run.

    Why pd.factorize instead of a dict comprehension or .map()?
    factorize is implemented in C and handles tens of millions of rows in ~1s.
    A Python loop or .map() over a dict costs ~30s at this scale.

    sort=True makes the mapping deterministic: the same input always produces
    the same indices, regardless of the order rows appear in the DataFrame.

    The returned dicts (user2idx, item2idx) are saved inside the checkpoint so
    each model run carries its own vocabulary and can be loaded without re-running
    preprocessing or knowing the original sampling fraction.
    """
    # factorize returns: (integer codes array, unique values array)
    # codes[i] is the 0-based index of df["customer_id"].iloc[i] in user_ids
    user_codes, user_ids = pd.factorize(df["customer_id"], sort=True)
    item_codes, item_ids = pd.factorize(df["movie_id"], sort=True)

    # Attach the compact indices as new columns (does not modify the original df)
    df = df.assign(user_idx=user_codes, item_idx=item_codes)

    # Invert the arrays to build lookup dicts: raw_id → compact_index
    # enumerate(user_ids) gives (compact_idx, raw_id), so we swap them.
    user2idx = {uid: idx for idx, uid in enumerate(user_ids.tolist())}
    item2idx = {mid: idx for idx, mid in enumerate(item_ids.tolist())}
    return df, user2idx, item2idx


def to_tensors(df: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the three model-input columns to PyTorch tensors on the target device.

    Why load everything onto the device upfront?
    At 10% of the dataset the full sample is ~300MB — it fits in device memory
    (MPS / GPU). Keeping tensors device-resident means each training batch is a
    pure on-device index operation with no CPU→device transfer overhead per batch.
    If device memory were tight we would stream mini-batches from CPU, but for
    this scale the upfront transfer is the right trade-off.
    """
    return (
        torch.from_numpy(df["user_idx"].to_numpy()).long().to(device),   # user embedding indices
        torch.from_numpy(df["item_idx"].to_numpy()).long().to(device),   # item embedding indices
        torch.from_numpy(df["rating"].to_numpy()).float().to(device),    # target ratings (float32)
    )
