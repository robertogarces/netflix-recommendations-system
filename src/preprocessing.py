import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('../')

from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH
from utils.files_management import fix_csv_with_commas_in_text, load_multiple_netflix_files
from utils.data_processing import filter_sparse_users_and_movies, filter_valid_ratings, convert_columns_to_string

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
combined_data_path_list = [
    RAW_DATA_PATH / f"combined_data_{i}.txt" for i in range(1, 5)
]
concatenated_data = RAW_DATA_PATH / "data.parquet"
movie_titles_path = RAW_DATA_PATH / "movie_titles.csv"
movie_titles_fixed_path = RAW_DATA_PATH / "movie_titles_fixed.csv"
processed_data_path = PROCESSED_DATA_PATH / "processed_data.parquet"

def load_and_concatenate_raw_data(paths: list, save_path: Path) -> pd.DataFrame:
    logger.info("Loading and combining raw data...")
    return load_multiple_netflix_files(file_paths=paths, save_path=save_path, verbose=True)

def preprocess_ratings_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Converting ID columns to string...")
    df = convert_columns_to_string(df, ['customer_id', 'movie_id'])
    logger.info("Filtering valid ratings (1-5)...")
    df = filter_valid_ratings(df, min_rating=1, max_rating=5)
    logger.info("Filtering sparse users and movies...")
    df = filter_sparse_users_and_movies(df, min_movie_ratings=50, min_user_ratings=10)
    return df

def main():
    df = load_and_concatenate_raw_data(combined_data_path_list, concatenated_data)
    fix_csv_with_commas_in_text(movie_titles_path, movie_titles_fixed_path)
    df = preprocess_ratings_data(df)
    logger.info(f"Saving processed data to: {processed_data_path}")
    df.to_parquet(processed_data_path)

if __name__ == "__main__":
    main()
