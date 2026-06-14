"""Training stage: processed data → fitted Surprise SVD model.

Outputs:
- models/svd_model.pkl       — the fitted Surprise algo (factors, biases, trainset)
- outputs/test_set.parquet   — held-out temporal split, consumed by src.evaluate
"""

import logging
import yaml
import mlflow
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

from config.paths import PROCESSED_DATA_PATH, OUTPUTS_PATH, MODELS_PATH, CONFIG_PATH
from src.model import save_model, estimate_pairs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed(sample_fraction: float, seed: int) -> pd.DataFrame:
    """Load the processed dataset, optionally sub-sampling a fraction of users.

    Why sample by user instead of by row?
    Random row sampling gives every user roughly `sample_fraction` of their
    ratings — a user with 200 ratings ends up with 20, too few to learn a
    reliable factor vector. By sampling *users* instead, each sampled user keeps
    their full history, so the model has enough signal per user. The trade-off is
    fewer users total, which is acceptable for a 10% sample.
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


def temporal_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into past (train) and future (test) by a global date quantile.

    Why temporal and not random?
    A random split leaks future information into training (e.g. a user's December
    rating helps predict their January rating). A temporal split mirrors the real
    deployment scenario: train on history, generalize to ratings not yet seen.

    The downside: users appearing for the first time after the cut-off have no
    training history — they become "cold users" in the test set, whose error is a
    floor no collaborative-filtering model can beat. src.evaluate reports their
    metrics separately.

    The boundary is a date quantile over the whole dataset (not per user), so the
    test row fraction matches test_size.
    """
    dates = df["date"].sort_values()
    test_date = dates.iloc[int(len(df) * (1 - test_size))]
    train = df[df["date"] < test_date]
    test  = df[df["date"] >= test_date]
    return train, test


def main():
    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    seed         = config["seed"]
    rating_scale = tuple(config["data"]["rating_scale"])
    training_cfg = config["training"]
    svd_cfg      = config["svd"]

    df = load_processed(training_cfg["data_sample_fraction"], seed)

    # Temporal split: train on the past, test on the future (untouched here).
    train_df, test_df = temporal_split(df, training_cfg["test_size"])

    # Flag test rows from users with no training history (they joined after the
    # split date). Their error is a cold-start floor no CF model can beat, so
    # evaluate reports warm/cold segments separately.
    train_user_ids = set(train_df["customer_id"].unique())
    test_df = test_df.assign(is_cold_user=~test_df["customer_id"].isin(train_user_ids))
    cold_fraction = test_df["is_cold_user"].mean()
    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,} rows | "
                f"cold-user test rows: {cold_fraction:.1%}")

    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(OUTPUTS_PATH / "test_set.parquet", index=False)
    logger.info("Saved held-out test set to outputs/test_set.parquet")

    # Surprise wants a (user, item, rating) frame wrapped in a Dataset. Reader
    # carries the rating scale; build_full_trainset turns the whole frame into the
    # internal trainset (raw↔inner id maps + rating index) that SVD trains on.
    reader   = Reader(rating_scale=rating_scale)
    data     = Dataset.load_from_df(train_df[["customer_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    logger.info(f"Trainset: {trainset.n_users:,} users, {trainset.n_items:,} items")

    algo = SVD(
        n_factors=svd_cfg["n_factors"],
        n_epochs=svd_cfg["n_epochs"],
        lr_all=svd_cfg["lr_all"],
        reg_all=svd_cfg["reg_all"],
        random_state=seed,
        verbose=True,  # print "Processing epoch N" so long fits show live progress
    )

    mlflow.set_experiment("Netflix_SVD")
    with mlflow.start_run(run_name="SVD_Training") as run:
        logger.info(f"Fitting SVD ({svd_cfg['n_epochs']} epochs) ...")
        algo.fit(trainset)

        # Train vs held-out RMSE side by side: a big gap means the model is
        # memorizing the training ratings. Both are computed vectorized over the
        # learned factors (see src.model.estimate_pairs), not Surprise's per-row
        # predict loop. The full evaluation (warm/cold, P@K) lives in src.evaluate;
        # this is just a fast overfitting check at the end of training.
        def rmse(df):
            est = estimate_pairs(algo, df["customer_id"], df["movie_id"], rating_scale)
            return float(np.sqrt(np.mean((est - df["rating"].to_numpy()) ** 2)))

        train_rmse, test_rmse = rmse(train_df), rmse(test_df)
        logger.info(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} "
                    f"(gap {test_rmse - train_rmse:+.4f})")

        mlflow.log_params({
            **svd_cfg,
            "seed":                 seed,
            "test_size":            training_cfg["test_size"],
            "data_sample_fraction": training_cfg["data_sample_fraction"],
            "num_users":            trainset.n_users,
            "num_items":            trainset.n_items,
            "n_train_rows":         len(train_df),
            "n_test_rows":          len(test_df),
            "cold_user_fraction":   round(float(cold_fraction), 4),
        })
        mlflow.log_metrics({"train_rmse": train_rmse, "test_rmse": test_rmse})
        mlflow.log_artifact(str(CONFIG_PATH / "settings.yaml"))

        # Stash the run id on the algo so evaluate.py can attach test metrics to
        # THIS run — one mlflow run then holds params, train and test together.
        algo.mlflow_run_id = run.info.run_id

        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_PATH / "svd_model.pkl"
        save_model(model_path, algo)
        mlflow.log_artifact(str(model_path))
        logger.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
