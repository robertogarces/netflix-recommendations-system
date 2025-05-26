import logging
import pickle
import pandas as pd
import optuna
import yaml
import mlflow
import mlflow.sklearn
from surprise import Dataset, Reader, SVD, accuracy

from config.paths import PROCESSED_DATA_PATH, CONFIG_PATH, MODELS_PATH
from utils.data_split import temporal_train_test_split
from utils.metrics import get_top_n, precision_recall_at_k
from utils.files_management import load_data

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')


def prepare_datasets(df: pd.DataFrame, test_size: float):
    logger.info(f"Splitting data into train/test with test_size={test_size}")
    train_df, test_df = temporal_train_test_split(df, test_size=test_size)
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    trainset = Dataset.load_from_df(train_df[['customer_id', 'movie_id', 'rating']], reader).build_full_trainset()
    testset = list(zip(test_df['customer_id'], test_df['movie_id'], test_df['rating']))
    return trainset, testset

def objective(trial, trainset, testset, config):
    params = {
        'n_factors': trial.suggest_int('n_factors', config['n_factors_min'], config['n_factors_max']),
        'n_epochs': trial.suggest_int('n_epochs', config['n_epochs_min'], config['n_epochs_max']),
        'lr_all': trial.suggest_float('lr_all', config['lr_all_min'], config['lr_all_max'], log=True),
        'reg_all': trial.suggest_float('reg_all', config['reg_all_min'], config['reg_all_max'], log=True)
    }

    model = SVD(**params)
    model.fit(trainset)
    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    return rmse

def train_svd_model(config):
    model_cfg = config["model"]
    svd_hyperparams = config["svd_hyperparams"]

    data = load_data(PROCESSED_DATA_PATH / "processed_data.parquet", model_cfg['data_sample_fraction'])
    trainset, testset = prepare_datasets(data, model_cfg['test_size'])

    mlflow.set_experiment("SVD_Recommender")
    with mlflow.start_run(run_name="SVD_Training"):
        mlflow.log_param("data_sample_fraction", model_cfg["data_sample_fraction"])
        mlflow.log_param("test_size", model_cfg["test_size"])

        if model_cfg.get("optimize", False):
            logger.info("Starting hyperparameter optimization with Optuna")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, trainset, testset, svd_hyperparams),
                           n_trials=model_cfg['n_trials'])

            best_params = study.best_params
            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", study.best_value)

            logger.info(f"Best RMSE: {study.best_value:.4f}")
            logger.info(f"Best params: {best_params}")

            with open(CONFIG_PATH / "best_params.yaml", "w") as f:
                yaml.dump(best_params, f)
                logger.info("Best parameters saved to best_params.yaml")
        else:
            try:
                with open(CONFIG_PATH / "best_params.yaml") as f:
                    best_params = yaml.safe_load(f)
                    logger.info("Loaded best parameters from best_params.yaml")
            except FileNotFoundError:
                logger.warning("Best parameters file not found. Using default parameters.")
                best_params = {
                    'n_factors': 100,
                    'n_epochs': 20,
                    'lr_all': 0.005,
                    'reg_all': 0.02
                }

        mlflow.log_params(best_params)

        logger.info("Training final SVD model")
        final_model = SVD(**best_params)
        final_model.fit(trainset)
        predictions = final_model.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        mlflow.log_metric("rmse", rmse)
        logger.info(f"RMSE: {rmse:.4f}")

        top_n = get_top_n(predictions, n=model_cfg['top_n'])
        precision, recall = precision_recall_at_k(predictions, k=model_cfg['top_n'], threshold=model_cfg['threshold'])
        logger.info(f"Precision@{model_cfg['top_n']}: {precision:.4f}")
        logger.info(f"Recall@{model_cfg['top_n']}: {recall:.4f}")

        mlflow.log_metric(f"precision_at_{model_cfg['top_n']}", precision)
        mlflow.log_metric(f"recall_at_{model_cfg['top_n']}", recall)

        model_output_path = MODELS_PATH / "svd_model.pkl"
        with open(model_output_path, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"Model saved to {model_output_path}")
        mlflow.log_artifact(model_output_path)

        best_params_path = CONFIG_PATH / "best_params.yaml"
        if best_params_path.exists():
            mlflow.log_artifact(best_params_path)
