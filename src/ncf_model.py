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
from utils.metrics import precision_recall_at_k
from utils.files_management import load_data

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def train_ncf_model(config):

    training_cfg = config["training"]
    eval_cfg     = config["evaluation"]
    ncf_cfg      = config["ncf"]

    data = load_data(PROCESSED_DATA_PATH / "processed_data.parquet", training_cfg['data_sample_fraction'])

    with open(ARTIFACTS_PATH / "user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open(ARTIFACTS_PATH / "item2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)

    data = data[data['customer_id'].isin(user2idx) & data['movie_id'].isin(item2idx)].copy()
    data['user_idx'] = data['customer_id'].map(user2idx)
    data['item_idx'] = data['movie_id'].map(item2idx)

    sorted_idx = data['date'].argsort()
    split_idx  = int(len(data) * (1 - training_cfg['test_size']))
    split_date = data['date'].iloc[sorted_idx.iloc[split_idx]]
    train_df   = data[data['date'] < split_date]
    test_df    = data[data['date'] >= split_date]

    train_loader = DataLoader(RatingsDataset(train_df), batch_size=ncf_cfg["batch_size"], shuffle=True)
    test_loader  = DataLoader(RatingsDataset(test_df),  batch_size=ncf_cfg["batch_size"])

    device = get_device()
    model = NCF(len(user2idx), len(item2idx), emb_size=ncf_cfg["emb_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ncf_cfg["lr"], weight_decay=ncf_cfg["weight_decay"])
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience, counter = 3, 0

    logger.info("Starting NCF Model training")

    mlflow.log_params({
        "emb_size":              ncf_cfg["emb_size"],
        "batch_size":            ncf_cfg["batch_size"],
        "lr":                    ncf_cfg["lr"],
        "weight_decay":          ncf_cfg["weight_decay"],
        "epochs":                ncf_cfg["epochs"],
        "test_size":             training_cfg["test_size"],
        "data_sample_fraction":  training_cfg["data_sample_fraction"],
    })

    for epoch in range(ncf_cfg["epochs"]):
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

    logger.info(f"Best NCF Model saved to {ncf_model_path}")

    mlflow.log_artifact(ARTIFACTS_PATH / "user2idx.pkl")
    mlflow.log_artifact(ARTIFACTS_PATH / "item2idx.pkl")

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

    top_n     = eval_cfg['top_n']
    threshold = eval_cfg['relevance_threshold']
    precision, recall = precision_recall_at_k(all_predictions, k=top_n, threshold=threshold)

    logger.info(f"Precision@{top_n}: {precision:.4f}")
    logger.info(f"Recall@{top_n}:    {recall:.4f}")

    mlflow.log_metric("precision_at_k", precision)
    mlflow.log_metric("recall_at_k", recall)

    mlflow.pytorch.log_model(model, artifact_path="model")

