"""Evaluation stage: fitted SVD model + held-out test set → metrics.

Loads the Surprise model produced by src.train, scores the test set, and reports
test RMSE (overall / warm / cold), Precision@K and Recall@K to the log, to
outputs/metrics.json (for the dashboard) and to mlflow.
"""

import json
import logging

import mlflow
import numpy as np
import pandas as pd
import yaml

from config.paths import CONFIG_PATH, MODELS_PATH, OUTPUTS_PATH
from src.model import estimate_pairs, load_model

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

    Why vectorized? The obvious version loops per user ("sort that user's items,
    take the top K, count the relevant ones"). Instead we sort the WHOLE array so
    every user's rows form one contiguous, score-ordered block, then sum within
    each block with np.add.reduceat — a GROUP BY SUM in a single NumPy pass. At
    millions of test rows the Python loop is the bottleneck; this avoids it.

    Worked example, k=1, relevant = (true >= threshold):

        row user est true                 after lexsort (user asc, est desc):
         0   A   4.8  5  relevant            A block -> [(4.8,5), (3.0,5)]
         1   A   3.0  5  relevant            B block -> [(4.0,2)]
         2   B   4.0  2

        start_idx = [0, 2]        counts = [2, 1]
        rank      = [0, 1, 0]     in_top_k (k=1) = [T, F, T]
        relevant  = [T, T, F]
        n_relevant       = reduceat(relevant,            [0,2]) = [2, 0]
        n_relevant_top_k = reduceat(relevant & in_top_k, [0,2]) = [1, 0]
        -> B dropped (0 relevant); A: precision = 1/1, recall = 1/2
    """
    # lexsort sorts by the LAST key first: user_ids (primary) groups each user's
    # rows together, -est_ratings (secondary) orders them best-score-first.
    order  = np.lexsort((-est_ratings, user_ids))
    uids   = user_ids[order]
    true_r = true_ratings[order]

    # uids is now sorted, so each user is one contiguous block. start_idx = where
    # each block begins; counts = its length. These drive the segmented sums.
    _, start_idx, counts = np.unique(uids, return_index=True, return_counts=True)
    relevant = true_r >= threshold

    # rank = each row's 0-based position within its block (global index minus the
    # block's start). Since rows are score-ordered, rank < k marks the top-K.
    rank     = np.arange(len(uids)) - np.repeat(start_idx, counts)
    in_top_k = rank < k

    # np.add.reduceat(arr, start_idx) sums arr in the segments [start_idx[i]:start_idx[i+1]]
    # — i.e. one total per user, in a single C-level pass.
    n_relevant       = np.add.reduceat(relevant,            start_idx)  # relevant items per user
    n_relevant_top_k = np.add.reduceat(relevant & in_top_k, start_idx)  # relevant AND in top-K

    mask = n_relevant > 0  # drop users with no relevant items (recall undefined)
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
