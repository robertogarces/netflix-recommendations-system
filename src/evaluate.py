"""Evaluation stage: fitted SVD model + held-out test set → metrics.

Loads the Surprise model produced by src.train, scores the test set, and reports
test RMSE (overall / warm / cold), Precision@K and Recall@K to the log, to
outputs/metrics.json (for the dashboard) and to mlflow.
"""

import json
import logging
import numpy as np
import pandas as pd
import yaml
import mlflow

from config.paths import OUTPUTS_PATH, MODELS_PATH, CONFIG_PATH
from src.model import load_model, estimate_pairs

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
    Recall@K:    of all relevant items the user has in the test set, what fraction
                 did we surface in the top-K?

    Users with no relevant items (no true rating >= threshold) are excluded from
    both metrics because recall would be 0/0.

    Vectorized strategy: sort the whole array so each user's items are contiguous
    and ordered by predicted score, then use np.add.reduceat to sum within each
    user's block — a GROUP BY SUM in one NumPy pass instead of a Python loop.
    """
    # lexsort: primary key user_ids (groups users), secondary -est (best first).
    order  = np.lexsort((-est_ratings, user_ids))
    uids   = user_ids[order]
    true_r = true_ratings[order]

    _, start_idx, counts = np.unique(uids, return_index=True, return_counts=True)
    relevant = true_r >= threshold

    # 0-based rank of each row within its user's block; top-K = rank < k.
    rank     = np.arange(len(uids)) - np.repeat(start_idx, counts)
    in_top_k = rank < k

    n_relevant       = np.add.reduceat(relevant,            start_idx)
    n_relevant_top_k = np.add.reduceat(relevant & in_top_k, start_idx)

    mask = n_relevant > 0
    precision = n_relevant_top_k[mask] / k
    recall    = n_relevant_top_k[mask] / n_relevant[mask]
    return float(precision.mean()), float(recall.mean())


def main():
    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    rating_scale = tuple(config["data"]["rating_scale"])
    eval_cfg     = config["evaluation"]

    algo = load_model(MODELS_PATH / "svd_model.pkl")
    ts = algo.trainset
    logger.info(f"Loaded model: {ts.n_users:,} users, {ts.n_items:,} items")

    test_df = pd.read_parquet(OUTPUTS_PATH / "test_set.parquet")
    logger.info(f"Test set: {len(test_df):,} rows")

    true_ratings = test_df["rating"].to_numpy()
    est_ratings  = estimate_pairs(algo, test_df["customer_id"], test_df["movie_id"], rating_scale)

    # Segment by cold users (no training history — flagged by src.train). Their
    # error floor is global_mean + item_bias; no CF model can beat it, so the warm
    # segment is the number that model iterations can actually move.
    sq_error = (est_ratings - true_ratings) ** 2
    warm = ~test_df["is_cold_user"].to_numpy()

    rmse      = float(np.sqrt(sq_error.mean()))
    rmse_warm = float(np.sqrt(sq_error[warm].mean()))
    rmse_cold = float(np.sqrt(sq_error[~warm].mean()))

    k, threshold = eval_cfg["top_n"], eval_cfg["relevance_threshold"]
    user_ids = test_df["customer_id"].to_numpy()
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

    # Persist a readable report for the dashboard (alongside mlflow).
    report = {"model_type": "svd", **metrics, "k": k, "threshold": threshold,
              "warm_fraction": float(warm.mean()), "n_test_rows": int(len(test_df))}
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_PATH / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics to outputs/metrics.json")

    # Attach test metrics to the training run, so one mlflow run holds the full
    # story: params, train RMSE, test metrics.
    run_id = getattr(algo, "mlflow_run_id", None)
    if run_id is None:
        mlflow.set_experiment("Netflix_SVD")
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "SVD_Evaluation"):
        mlflow.set_tags({"eval_k": k, "eval_threshold": threshold})
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
