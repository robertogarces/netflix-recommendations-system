import sys
import logging
import pickle
import pandas as pd

sys.path.append('../')

from config.paths import RAW_DATA_PATH, ARTIFACTS_PATH, MODELS_PATH, FINAL_DATA_PATH
from utils.files_management import load_model, load_netflix_data
from utils.data_processing import convert_columns_to_string, filter_unseen

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Generates predicted ratings for a given dataframe using a Surprise model.

    Parameters:
        df (pd.DataFrame): DataFrame with 'customer_id' and 'movie_id'.
        model: Trained Surprise model.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'pred_rating' column.
    """
    predictions = []
    for _, row in df.iterrows():
        uid = str(row['customer_id'])  # surprise requires string IDs
        iid = str(row['movie_id'])

        pred = model.predict(uid, iid)
        predictions.append(pred.est)

    df["pred_rating"] = predictions
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

    # Filter out unknown users/movies
    df = filter_unseen(df, valid_users=valid_users, valid_movies=valid_movies)

    # Load trained model
    logger.info("Loading trained SVD model")
    model = load_model(MODELS_PATH / "svd_model.pkl")

    # Generate predictions
    logger.info("Generating predictions")
    df = generate_predictions(df, model)

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






