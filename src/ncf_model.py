import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle 
import mlflow
import mlflow.pytorch

from config.paths import PROCESSED_DATA_PATH, MODELS_PATH, ARTIFACTS_PATH
from utils.pytorch_utils import RatingsDataset, NCF, get_device
from utils.metrics import get_top_n, precision_recall_at_k
from utils.files_management import load_data

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def train_ncf_model(config):

    model_cfg = config["model"]
    data = load_data(PROCESSED_DATA_PATH / "processed_data.parquet", model_cfg['data_sample_fraction'])

    user2idx = {u: i for i, u in enumerate(data['customer_id'].unique())}
    item2idx = {m: i for i, m in enumerate(data['movie_id'].unique())}
    data['user_idx'] = data['customer_id'].map(user2idx)
    data['item_idx'] = data['movie_id'].map(item2idx)

    split_idx = int(len(data) * (1 - model_cfg['test_size']))
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    train_loader = DataLoader(RatingsDataset(train_df), batch_size=model_cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(RatingsDataset(test_df), batch_size=model_cfg["batch_size"])

    device = get_device()
    model = NCF(len(user2idx), len(item2idx), emb_size=model_cfg["emb_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience, counter = 3, 0

    logger.info("Starting NCF Model training")

    # Log params
    mlflow.log_params({
        "emb_size": model_cfg["emb_size"],
        "batch_size": model_cfg["batch_size"],
        "lr": model_cfg["lr"],
        "weight_decay": model_cfg["weight_decay"],
        "epochs": model_cfg["epochs"],
        "test_size": model_cfg["test_size"],
        "data_sample_fraction": model_cfg["data_sample_fraction"]
            }
        )

    for epoch in range(model_cfg["epochs"]):
        model.train()
        total_loss = 0
        for u, i, r in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(r)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, r in test_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i)
                val_loss += loss_fn(pred, r).item() * len(r)
        val_loss /= len(test_loader.dataset)

        train_rmse = train_loss ** 0.5
        val_rmse = val_loss ** 0.5

        logger.info(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}")
        logger.info(f"Epoch {epoch+1}: Val RMSE = {val_rmse:.4f}")

        mlflow.log_metric("train_rmse", train_rmse, step=epoch+1)
        mlflow.log_metric("val_rmse", val_rmse, step=epoch+1)

        ncf_model_path = MODELS_PATH / 'ncf_model.pt'

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ncf_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    logger.info("Best NCF Model saved to {ncf_model_path}")
    
    user2idx_path = ARTIFACTS_PATH / "user2idx.pkl"
    item2idx_path = ARTIFACTS_PATH / "item2idx.pkl"

    with open(user2idx_path, "wb") as f:
        pickle.dump(user2idx, f)

    with open(item2idx_path, "wb") as f:
        pickle.dump(item2idx, f)

    mlflow.log_artifact(user2idx_path)
    mlflow.log_artifact(item2idx_path)

    # Load model and evaluate on testing
    model.load_state_dict(torch.load(MODELS_PATH / "ncf_model.pt"))
    model.eval()

    test_loss = 0
    all_predictions = []

    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            test_loss += loss_fn(pred, r).item() * len(r)

            # Save preds
            u_cpu = u.cpu().numpy()
            i_cpu = i.cpu().numpy()
            r_cpu = r.cpu().numpy()
            pred_cpu = pred.cpu().numpy()
            for uid, iid, true_r, est in zip(u_cpu, i_cpu, r_cpu, pred_cpu):
                all_predictions.append((uid, iid, true_r, est, None))  

    test_loss /= len(test_loader.dataset)
    test_rmse = test_loss ** 0.5
    logger.info(f"Final test RMSE: {test_rmse:.4f}")
    mlflow.log_metric("test_rmse", test_rmse)

  #  top_n = get_top_n(all_predictions, n=model_cfg['top_n'])
    precision, recall = precision_recall_at_k(all_predictions, k=model_cfg['top_n'], threshold=model_cfg['threshold'])

    logger.info(f"Precision@{model_cfg['top_n']}: {precision:.4f}")
    logger.info(f"Recall@{model_cfg['top_n']}: {recall:.4f}")

    mlflow.log_metric("precision_at_k", precision)
    mlflow.log_metric("recall_at_k", recall)

    mlflow.pytorch.log_model(model, artifact_path="model")

