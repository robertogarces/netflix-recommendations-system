"""Training stage: processed data → trained NCF checkpoint.

Outputs:
- models/ncf_model.pt        — state_dict + index mappings + model config
- artifacts/test_set.parquet — held-out temporal split, consumed by src.evaluate
"""

import logging
import yaml
import mlflow
import torch
from tqdm import tqdm

from config.paths import OUTPUTS_PATH, MODELS_PATH, CONFIG_PATH
from src.data import load_processed, temporal_split, add_index_columns, to_tensors
from src.model import NCF, get_device

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validation_rmse(model, users, items, ratings, batch_size, rating_scale) -> float:
    """Compute RMSE on a dataset in mini-batches, accumulating error on the device.

    Why accumulate on-device and call .item() only once?
    On MPS and CUDA, each .item() (or any operation that reads a tensor back to
    the CPU) forces a host-device synchronization — the CPU blocks until the GPU
    catches up. Calling it once per batch would turn the validation loop into a
    chain of sync points and kill throughput. Instead we keep sq_error as a
    scalar tensor on the device and sync only at the end.

    torch.zeros(()) creates a 0-dimensional (scalar) tensor — no batch dimension,
    just a single number that lives on the device.
    """
    model.eval()
    sq_error = torch.zeros((), device=users.device)  # scalar on device, avoids per-batch sync
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            end = start + batch_size
            est = model(users[start:end], items[start:end]).clamp(*rating_scale)
            sq_error += torch.sum((est - ratings[start:end]) ** 2)
    return (sq_error.item() / len(users)) ** 0.5  # single host-device sync here


def main():
    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    seed         = config["seed"]
    rating_scale = tuple(config["data"]["rating_scale"])
    training_cfg = config["training"]
    ncf_cfg      = config["ncf"]

    torch.manual_seed(seed)

    df = load_processed(training_cfg["data_sample_fraction"], seed)

    # The checkpoint owns its vocabulary: mappings are built from this run's
    # data and saved with the model (see add_index_columns for rationale).
    df, user2idx, item2idx = add_index_columns(df)

    # Three-way split: val drives early stopping, test stays untouched until
    # src.evaluate so reported test metrics carry no model-selection bias.
    train_df, val_df, test_df = temporal_split(
        df, training_cfg["val_size"], training_cfg["test_size"]
    )
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} rows")

    # Flag test rows from users with no training history (joined after the
    # split date). Their RMSE is a cold-start floor no CF model can beat, so
    # evaluate reports warm/cold segments separately.
    train_user_ids = set(train_df["customer_id"].unique())
    test_df = test_df.assign(is_cold_user=~test_df["customer_id"].isin(train_user_ids))
    cold_fraction = test_df["is_cold_user"].mean()
    logger.info(f"Cold-user test rows: {cold_fraction:.1%}")

    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(OUTPUTS_PATH / "test_set.parquet", index=False)
    logger.info("Saved held-out test set to outputs/test_set.parquet")

    device = get_device()
    logger.info(f"Device: {device}")

    train_users, train_items, train_ratings = to_tensors(train_df, device)
    val_users,   val_items,   val_ratings   = to_tensors(val_df, device)

    global_mean = train_ratings.mean().item()
    model = NCF(
        num_users=len(user2idx),
        num_items=len(item2idx),
        emb_size=ncf_cfg["emb_size"],
        hidden_dims=tuple(ncf_cfg["hidden_dims"]),
        dropout=ncf_cfg["dropout"],
        global_mean=global_mean,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ncf_cfg["lr"], weight_decay=ncf_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    loss_fn   = torch.nn.MSELoss()
    batch_size = ncf_cfg["batch_size"]

    mlflow.set_experiment("Netflix_NCF")
    with mlflow.start_run(run_name="NCF_Training") as run:
        mlflow.log_params({
            **ncf_cfg,
            "seed":                 seed,
            "val_size":             training_cfg["val_size"],
            "test_size":            training_cfg["test_size"],
            "data_sample_fraction": training_cfg["data_sample_fraction"],
            "num_users":            len(user2idx),
            "num_items":            len(item2idx),
            "n_train_rows":         len(train_df),
            "n_val_rows":           len(val_df),
            "n_test_rows":          len(test_df),
            "cold_user_fraction":   round(float(cold_fraction), 4),
        })
        # Full config snapshot for exact reproducibility of this run
        mlflow.log_artifact(str(CONFIG_PATH / "settings.yaml"))

        best_val_rmse = float("inf")
        patience   = ncf_cfg["early_stopping_patience"]
        no_improve = 0
        model_path = MODELS_PATH / "ncf_model.pt"
        MODELS_PATH.mkdir(parents=True, exist_ok=True)

        def save_checkpoint():
            # Self-contained bundle: every artifact evaluate.py and recommend.py
            # need to reconstruct the exact model without re-running preprocessing.
            # - state_dict:    the trained weights
            # - user2idx/item2idx: mapping from raw IDs to embedding indices
            # - global_mean:   frozen scalar baked into the model buffer
            # - ncf_config:    hyperparams needed to rebuild the architecture
            # - mlflow_run_id: lets evaluate.py log test metrics into THIS run
            torch.save({
                "state_dict":     model.state_dict(),
                "user2idx":       user2idx,
                "item2idx":       item2idx,
                "global_mean":    global_mean,
                "ncf_config":     ncf_cfg,
                "mlflow_run_id":  run.info.run_id,
            }, model_path)

        n_train = len(train_users)
        for epoch in range(1, ncf_cfg["epochs"] + 1):
            model.train()
            # Shuffle training indices each epoch so the model doesn't memorize
            # batch order. randperm on-device avoids a CPU→device copy per step.
            perm = torch.randperm(n_train, device=device)

            # Accumulate squared error as a device-resident scalar (see validation_rmse
            # for why we avoid per-batch .item() calls).
            train_sq_error = torch.zeros((), device=device)
            for start in tqdm(range(0, n_train, batch_size), desc=f"Epoch {epoch}/{ncf_cfg['epochs']}"):
                b = perm[start:start + batch_size]  # shuffled indices for this mini-batch

                # MSELoss returns the *mean* squared error over the batch.
                # We multiply back by len(b) to accumulate the *sum*, so we can
                # compute a correct epoch-level RMSE at the end.
                loss = loss_fn(model(train_users[b], train_items[b]), train_ratings[b])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_sq_error += loss.detach() * len(b)  # sum, not mean
            train_rmse = (train_sq_error.item() / n_train) ** 0.5

            val_rmse = validation_rmse(model, val_users, val_items, val_ratings, batch_size, rating_scale)

            # ReduceLROnPlateau: if val_rmse stops improving, halve the learning rate.
            # This lets the model make large updates early and fine-tune later.
            scheduler.step(val_rmse)

            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch}: train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}  lr={lr:.1e}")
            mlflow.log_metrics({"train_rmse": train_rmse, "val_rmse": val_rmse, "lr": lr}, step=epoch)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                save_checkpoint()   # overwrite checkpoint only when val improves
                no_improve = 0
            else:
                no_improve += 1
                # Early stopping: if val_rmse hasn't improved for `patience` epochs,
                # the best checkpoint is already saved — stop to avoid overfitting.
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Best val RMSE: {best_val_rmse:.4f} | checkpoint: {model_path}")
        mlflow.log_metric("best_val_rmse", best_val_rmse)
        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    main()
