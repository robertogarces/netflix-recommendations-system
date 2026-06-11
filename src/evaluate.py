"""Evaluation stage: trained checkpoint + held-out test set → metrics.

Loads the self-contained checkpoint produced by src.train, rebuilds the model,
and reports test RMSE, Precision@K and Recall@K to the log and mlflow.
"""

import json
import logging
import numpy as np
import pandas as pd
import yaml
import mlflow
import torch

from config.paths import OUTPUTS_PATH, MODELS_PATH, CONFIG_PATH
from src.model import get_device, load_checkpoint

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def precision_recall_at_k(
    user_ids: np.ndarray,
    true_ratings: np.ndarray,
    est_ratings: np.ndarray,
    k: int = 10,
    threshold: float = 4.0,
) -> tuple[float, float]:
    """Mean Precision@K and Recall@K across users, fully vectorized.

    Precision@K: of the top-K items we recommended, what fraction were relevant?
                 = (relevant items in top-K) / K

    Recall@K:    of all relevant items the user has in the test set, what fraction
                 did we surface in the top-K?
                 = (relevant items in top-K) / (total relevant items for that user)

    Users with no relevant items (no true rating >= threshold) are excluded from
    recall because recall would be 0/0 (undefined denominator). They are also
    excluded from precision to keep both metrics on the same set of users.

    Vectorized strategy:
    Instead of a Python loop over each user, we sort the entire array so each
    user's items are contiguous and ordered by predicted score, then use
    np.add.reduceat to sum within each user's block — equivalent to a SQL
    GROUP BY SUM but in one NumPy call.
    """
    # np.lexsort sorts by the LAST key first (primary), then earlier keys (secondary).
    # Primary sort:   user_ids ascending  → groups all items for the same user together
    # Secondary sort: -est_ratings        → within each user, highest predicted score first
    order  = np.lexsort((-est_ratings, user_ids))
    uids   = user_ids[order]
    true_r = true_ratings[order]

    # np.unique with return_index=True gives the position in `uids` where each
    # unique user first appears — these are the segment boundaries for reduceat.
    # counts tells us how many items each user has (used to compute per-item rank).
    _, start_idx, counts = np.unique(uids, return_index=True, return_counts=True)

    # A rating is "relevant" if it meets or exceeds the threshold (e.g. 4.5 stars).
    relevant = true_r >= threshold

    # Compute each row's 0-based rank within its user's block.
    # np.repeat(start_idx, counts) repeats each user's start position once per item
    # that belongs to that user. Subtracting from the global index gives local rank.
    # Example: user A has items at global positions [3,4,5] → ranks [0,1,2]
    rank     = np.arange(len(uids)) - np.repeat(start_idx, counts)
    in_top_k = rank < k  # True for positions 0..k-1 within each user (the top-K)

    # np.add.reduceat(arr, indices): sum arr in non-overlapping segments defined by
    # indices. It's equivalent to [arr[s:e].sum() for s,e in zip(indices, indices[1:]+[len])].
    # Much faster than a Python loop because it's a single C-level pass over the array.
    n_relevant       = np.add.reduceat(relevant,            start_idx)  # total relevant per user
    n_relevant_top_k = np.add.reduceat(relevant & in_top_k, start_idx)  # relevant hits in top-K

    # Exclude users with no relevant items (recall is undefined for them).
    mask = n_relevant > 0
    precision = n_relevant_top_k[mask] / k                       # hits / K
    recall    = n_relevant_top_k[mask] / n_relevant[mask]        # hits / total relevant

    return float(precision.mean()), float(recall.mean())


def predict(model, users, items, batch_size, rating_scale, device) -> np.ndarray:
    """Batched inference with predictions clamped to the rating scale."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            end = start + batch_size
            u = users[start:end].to(device)
            i = items[start:end].to(device)
            preds.append(model(u, i).clamp(*rating_scale).cpu())
    return torch.cat(preds).numpy()


def main():
    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    rating_scale = tuple(config["data"]["rating_scale"])
    eval_cfg     = config["evaluation"]
    batch_size   = config["ncf"]["batch_size"]

    device = get_device()
    model, checkpoint = load_checkpoint(MODELS_PATH / "ncf_model.pt", device)
    logger.info(f"Loaded checkpoint: {len(checkpoint['user2idx']):,} users, "
                f"{len(checkpoint['item2idx']):,} items | device: {device}")

    # Test set already carries user_idx/item_idx from the training run
    test_df = pd.read_parquet(OUTPUTS_PATH / "test_set.parquet")
    logger.info(f"Test set: {len(test_df):,} rows")

    users = torch.from_numpy(test_df["user_idx"].to_numpy()).long()
    items = torch.from_numpy(test_df["item_idx"].to_numpy()).long()
    true_ratings = test_df["rating"].to_numpy()

    est_ratings = predict(model, users, items, batch_size, rating_scale, device)

    # Segment by cold users (no training history — flagged by src.train).
    # Their error floor is global_mean + item_bias; no CF model can beat it,
    # so the warm segment is the number that model iterations can move.
    sq_error = (est_ratings - true_ratings) ** 2
    warm = ~test_df["is_cold_user"].to_numpy()

    rmse      = float(np.sqrt(sq_error.mean()))
    rmse_warm = float(np.sqrt(sq_error[warm].mean()))
    rmse_cold = float(np.sqrt(sq_error[~warm].mean()))

    k, threshold = eval_cfg["top_n"], eval_cfg["relevance_threshold"]
    user_ids = test_df["user_idx"].to_numpy()
    precision, recall = precision_recall_at_k(user_ids, true_ratings, est_ratings, k, threshold)
    precision_warm, recall_warm = precision_recall_at_k(
        user_ids[warm], true_ratings[warm], est_ratings[warm], k, threshold
    )

    metrics = {
        "test_rmse":           rmse,
        "test_rmse_warm":      rmse_warm,
        "test_rmse_cold":      rmse_cold,
        "precision_at_k":      precision,
        "precision_at_k_warm": precision_warm,
        "recall_at_k":         recall,
        "recall_at_k_warm":    recall_warm,
    }

    logger.info(f"Test RMSE:      {rmse:.4f}")
    logger.info(f"  warm users:   {rmse_warm:.4f}  ({warm.mean():.0%} of rows)")
    logger.info(f"  cold users:   {rmse_cold:.4f}  ({(~warm).mean():.0%} of rows)")
    logger.info(f"Precision@{k}:   {precision:.4f}  (warm: {precision_warm:.4f})")
    logger.info(f"Recall@{k}:      {recall:.4f}  (warm: {recall_warm:.4f})")

    # Persist a readable report for the dashboard (alongside mlflow)
    report = {**metrics, "k": k, "threshold": threshold,
              "warm_fraction": float(warm.mean()), "n_test_rows": int(len(test_df))}
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_PATH / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics to outputs/metrics.json")

    # Attach test metrics to the training run that produced this checkpoint,
    # so one mlflow run holds the full story: params, training curves, test.
    run_id = checkpoint.get("mlflow_run_id")
    if run_id is None:
        mlflow.set_experiment("Netflix_NCF")
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "NCF_Evaluation"):
        # Tags (not params) — re-evaluating with a different k must not clash
        mlflow.set_tags({"eval_k": k, "eval_threshold": threshold})
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
