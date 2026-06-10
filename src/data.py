"""Shared data utilities for the train and evaluate stages.

Keeping sampling, splitting and tensor construction in one place guarantees
both stages see the data the same way.
"""

import logging
import numpy as np
import pandas as pd
import torch

from config.paths import PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


def load_processed(sample_fraction: float, seed: int) -> pd.DataFrame:
    """Load the processed dataset, optionally sampling a fraction of users.

    Sampling is done by user (keeping each sampled user's full history) rather
    than by row: random row sampling dilutes every user's history to a handful
    of ratings, which caps what the embeddings can learn regardless of model
    capacity.
    """
    df = pd.read_parquet(PROCESSED_DATA_PATH / "processed_data.parquet")
    if sample_fraction < 1.0:
        users = df["customer_id"].unique()
        rng = np.random.default_rng(seed)
        sampled = rng.choice(users, size=int(len(users) * sample_fraction), replace=False)
        df = df[df["customer_id"].isin(sampled)]
    logger.info(
        f"Loaded {len(df):,} rows | {df['customer_id'].nunique():,} users "
        f"({sample_fraction:.0%} user sample)"
    )
    return df


def temporal_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at the (1 - test_size) date quantile: train on the past, test on
    the future. Prevents leakage and mirrors deployment conditions."""
    sorted_idx = df["date"].argsort()
    split_idx  = int(len(df) * (1 - test_size))
    split_date = df["date"].iloc[sorted_idx.iloc[split_idx]]
    return df[df["date"] < split_date], df[df["date"] >= split_date]


def add_index_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Add compact, contiguous user_idx/item_idx columns and return the mappings.

    Compact tables matter for speed: Adam updates the *entire* embedding table
    every step, so sizing it to all 463k catalog users when a 5% sample only
    contains ~23k would make every step ~8x slower. The mappings are saved
    with the checkpoint — the model owns its vocabulary.

    Uses pd.factorize (C implementation) instead of a Python dict .map(),
    which costs ~30s on 30M rows. sort=True keeps indices deterministic.
    """
    user_codes, user_ids = pd.factorize(df["customer_id"], sort=True)
    item_codes, item_ids = pd.factorize(df["movie_id"], sort=True)
    df = df.assign(user_idx=user_codes, item_idx=item_codes)
    user2idx = {uid: idx for idx, uid in enumerate(user_ids.tolist())}
    item2idx = {mid: idx for idx, mid in enumerate(item_ids.tolist())}
    return df, user2idx, item2idx


def to_tensors(df: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (users, items, ratings) tensors resident on the device.

    The full sample fits in device memory (~100MB at 5%), so batching becomes
    pure on-device indexing with no per-batch host transfers.
    """
    return (
        torch.from_numpy(df["user_idx"].to_numpy()).long().to(device),
        torch.from_numpy(df["item_idx"].to_numpy()).long().to(device),
        torch.from_numpy(df["rating"].to_numpy()).float().to(device),
    )
