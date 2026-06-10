"""Evaluation stage: trained checkpoint + held-out test set → metrics.

Loads the self-contained checkpoint produced by src.train, rebuilds the model,
and reports test RMSE, Precision@K and Recall@K to the log and mlflow.
"""

import logging
import numpy as np
import pandas as pd
import yaml
import mlflow
import torch

from config.paths import ARTIFACTS_PATH, MODELS_PATH, CONFIG_PATH
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
    """
    Mean Precision@K and Recall@K across users, fully vectorized.

    Users with no relevant items (no true rating >= threshold) are excluded,
    since recall is undefined for them.
    """
    # Sort by user, then by estimated rating descending within each user
    order  = np.lexsort((-est_ratings, user_ids))
    uids   = user_ids[order]
    true_r = true_ratings[order]

    # Group boundaries per user (uids is sorted, so start indices are increasing)
    _, start_idx, counts = np.unique(uids, return_index=True, return_counts=True)

    relevant = true_r >= threshold
    rank     = np.arange(len(uids)) - np.repeat(start_idx, counts)  # 0-based rank within user
    in_top_k = rank < k

    n_relevant       = np.add.reduceat(relevant, start_idx)
    n_relevant_top_k = np.add.reduceat(relevant & in_top_k, start_idx)

    mask = n_relevant > 0
    precision = n_relevant_top_k[mask] / k
    recall    = n_relevant_top_k[mask] / n_relevant[mask]

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
    test_df = pd.read_parquet(ARTIFACTS_PATH / "test_set.parquet")
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

    logger.info(f"Test RMSE:      {rmse:.4f}")
    logger.info(f"  warm users:   {rmse_warm:.4f}  ({warm.mean():.0%} of rows)")
    logger.info(f"  cold users:   {rmse_cold:.4f}  ({(~warm).mean():.0%} of rows)")
    logger.info(f"Precision@{k}:   {precision:.4f}  (warm: {precision_warm:.4f})")
    logger.info(f"Recall@{k}:      {recall:.4f}  (warm: {recall_warm:.4f})")

    # Attach test metrics to the training run that produced this checkpoint,
    # so one mlflow run holds the full story: params, training curves, test.
    run_id = checkpoint.get("mlflow_run_id")
    if run_id is None:
        mlflow.set_experiment("Netflix_NCF")
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "NCF_Evaluation"):
        # Tags (not params) — re-evaluating with a different k must not clash
        mlflow.set_tags({"eval_k": k, "eval_threshold": threshold})
        mlflow.log_metrics({
            "test_rmse":           rmse,
            "test_rmse_warm":      rmse_warm,
            "test_rmse_cold":      rmse_cold,
            "precision_at_k":      precision,
            "precision_at_k_warm": precision_warm,
            "recall_at_k":         recall,
            "recall_at_k_warm":    recall_warm,
        })


if __name__ == "__main__":
    main()
