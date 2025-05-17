import sys
import logging
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from surprise import Dataset, Reader, SVD, accuracy
import pickle
import yaml

sys.path.append('../')

from config.paths import PROCESSED_DATA_PATH, CONFIG_PATH, MODELS_PATH
from utils.data_split import temporal_train_test_split
from utils.metrics import get_top_n, precision_recall_at_k
from utils.config_loader import load_config

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str, sample_fraction: float):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)

    # As the dataset contains 100M rows we might use just a proportion of it so it's easier to process 
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42)
        logger.info(f"Sampled {sample_fraction*100}% of data, resulting in {len(data)} rows")
    else:
        logger.info(f"Using 100% of data, {len(data)} rows")
    return data

def prepare_datasets(df: pd.DataFrame, test_size: float):
    logger.info(f"Splitting data into train/test with test_size={test_size}")
    # Train and testing are divided according to the date of the rating. Test size is defined at config/settings.yaml
    train_df, test_df = temporal_train_test_split(df, test_size=test_size)
    # Format the datasets so they can be trained with a SVD
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    trainset = Dataset.load_from_df(train_df[['customer_id', 'movie_id', 'rating']], reader).build_full_trainset()
    testset = list(zip(test_df['customer_id'], test_df['movie_id'], test_df['rating']))
    return trainset, testset

# Optimize SVD model hyper params with Optuna
def objective(trial, trainset, testset, config):
    params = {
        'n_factors': trial.suggest_int('n_factors', config['n_factors_min'], config['n_factors_max']),
        'n_epochs': trial.suggest_int('n_epochs', config['n_epochs_min'], config['n_epochs_max']),
        'lr_all': trial.suggest_float('lr_all', config['lr_all_min'], config['lr_all_max'], log=True),
        'reg_all': trial.suggest_float('reg_all', config['reg_all_min'], config['reg_all_max'], log=True)
    }

    # Start a ML Flow process so all of the experiments are recorded
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = SVD(**params)
        model.fit(trainset)

        preds = model.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)

        mlflow.log_metric("rmse", rmse)
        return rmse

def main():
    config = load_config(CONFIG_PATH / "settings.yaml")
    model_cfg = config["model"]
    svd_hyperparams = config["svd_hyperparams"]

    data = load_data(PROCESSED_DATA_PATH / "processed_data.parquet", model_cfg['data_sample_fraction'])
    trainset, testset = prepare_datasets(data, model_cfg['test_size'])

    mlflow.set_experiment("Netflix_SVD_Optimization")

    with mlflow.start_run(run_name="SVD-Optuna-Tuning"):
        # At config/settings.yaml we can choose wheter we optimize the SVD model. This file also contains other settings for both training and optimization
        if model_cfg.get("optimize", False):
            logger.info("Starting hyperparameter optimization with Optuna")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, trainset, testset, svd_hyperparams),
                           n_trials=model_cfg['n_trials'])

            best_params = study.best_params
            logger.info(f"Best RMSE: {study.best_value:.4f}")
            logger.info(f"Best params: {best_params}")

            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", study.best_value)

            # Save best parameters to YAML
            with open(CONFIG_PATH / "best_params.yaml", "w") as f:
                yaml.dump(best_params, f)
                logger.info("Best parameters saved to best_params.yaml")
        # If we don't want to optimize the model, it's going to be trained with the last saved best parameters. If there's not a best params file it is going to be trained with the parameters below
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

        # Train and evaluate final model
        logger.info(f"Training SVD Model with best parameters")
        final_model = SVD(**best_params)
        final_model.fit(trainset)
        predictions = final_model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        logger.info(f"RMSE: {rmse:.4f}")
        mlflow.log_metric("final_rmse", rmse)

        # Precision/Recall
        top_n = get_top_n(predictions, n=model_cfg['top_n'])
        precision, recall = precision_recall_at_k(predictions, k=model_cfg['top_n'], threshold=model_cfg['threshold'])

        logger.info(f"Precision@{model_cfg['top_n']}: {precision:.4f}")
        logger.info(f"Recall@{model_cfg['top_n']}: {recall:.4f}")
        mlflow.log_metric(f"precision_at_{model_cfg['top_n']}", precision)
        mlflow.log_metric(f"recall_at_{model_cfg['top_n']}", recall)

        # Log model to MLflow and save locally
        mlflow.sklearn.log_model(final_model, "svd_model")
        model_output_path = MODELS_PATH / "svd_model.pkl"
        with open(model_output_path, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    main()
