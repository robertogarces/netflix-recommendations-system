import sys
import logging
import pickle
import pandas as pd
import torch
from utils.pytorch_utils import NCF, get_device

sys.path.append('../')

from config.paths import RAW_DATA_PATH, ARTIFACTS_PATH, MODELS_PATH, FINAL_DATA_PATH, CONFIG_PATH
from utils.files_management import load_model, load_netflix_data
from utils.data_processing import convert_columns_to_string, filter_unseen
from utils.config_loader import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_predictions(df: pd.DataFrame, model_type: str, model, user2idx=None, item2idx=None) -> pd.DataFrame:
    """
    Generates predicted ratings using either a Surprise model or an NCF PyTorch model.
    """
    if model_type == "svd":
        predictions = []
        for _, row in df.iterrows():
            uid = str(row['customer_id'])
            iid = str(row['movie_id'])
            pred = model.predict(uid, iid)
            predictions.append(pred.est)
        df["pred_rating"] = predictions

    elif model_type == "ncf":
        device = get_device()
        model.to(device)
        model.eval()

        # Filtrar usuarios e items que están fuera del vocabulario
        df = df[df['customer_id'].isin(user2idx.keys()) & df['movie_id'].isin(item2idx.keys())]

        # Mapear a índices
        df['user_idx'] = df['customer_id'].map(user2idx)
        df['item_idx'] = df['movie_id'].map(item2idx)

        # Convertir a tensores y predecir en batch si es necesario
        user_tensor = torch.tensor(df['user_idx'].values, dtype=torch.long).to(device)
        item_tensor = torch.tensor(df['item_idx'].values, dtype=torch.long).to(device)

        with torch.no_grad():
            preds = model(user_tensor, item_tensor).cpu().numpy()

        df["pred_rating"] = preds

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return df

def main():
    # Load test data
    data_path = RAW_DATA_PATH / "qualifying.txt"
    logger.info(f"Loading test data from {data_path}")
    df = load_netflix_data(file_path=data_path, has_ratings=False, verbose=True)

    # Process the dataset
    df = convert_columns_to_string(df, ['customer_id', 'movie_id'])

    # Load valid user and movie IDs
    logger.info("Loading valid user and movie IDs")
    with open(ARTIFACTS_PATH / "valid_users.pkl", "rb") as f:
        valid_users = pickle.load(f)
    with open(ARTIFACTS_PATH / "valid_movies.pkl", "rb") as f:
        valid_movies = pickle.load(f)

    # Filter out the users/movies that were removed at the preprocessing step
    df = filter_unseen(df, valid_users=valid_users, valid_movies=valid_movies)

    config = load_config(CONFIG_PATH / "settings.yaml")
    model_type = config["model"]["type"].lower()

    if model_type == "svd":
        logger.info("Loading trained SVD model")
        model = load_model(MODELS_PATH / "svd_model.pkl")
        df = generate_predictions(df, model_type="svd", model=model)

    elif model_type == "ncf":
        logger.info("Loading trained NCF model")
        
        # Load dictionaries 
        with open(ARTIFACTS_PATH / "user2idx.pkl", "rb") as f:
            user2idx = pickle.load(f)
        with open(ARTIFACTS_PATH / "item2idx.pkl", "rb") as f:
            item2idx = pickle.load(f)

        # Get embedding size from config
        emb_size = config["model"]["emb_size"]

        # Create model with correct dimensions
        num_users = len(user2idx)
        num_items = len(item2idx)
        device = get_device()

        model = NCF(num_users=num_users, num_items=num_items, emb_size=emb_size).to(device)

        # Load trained weights
        model.load_state_dict(torch.load(MODELS_PATH / "ncf_model.pt", map_location=device))


        df = generate_predictions(df, model_type="ncf", model=model, user2idx=user2idx, item2idx=item2idx)

    else:
        raise ValueError("Invalid model type")


    logger.info(f"Generated {len(df)} predictions")

    # Load movie titles
    movie_titles_path = RAW_DATA_PATH / "movie_titles_fixed.csv"
    logger.info(f"Loading movie titles from {movie_titles_path}")
    movie_titles = pd.read_csv(
        movie_titles_path,
        sep=';',
        encoding='latin1',
        header=None,
        names=['id', 'year', 'title']
    )
    movie_titles['id'] = movie_titles['id'].astype(str)

    # Merge for interpretability
    final_df = pd.merge(df, movie_titles, how='left', left_on='movie_id', right_on='id')
    logger.info("Merged predictions with movie titles")
    logger.info(final_df.head())

    # Save results
    output_path = FINAL_DATA_PATH / "predictions.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()

