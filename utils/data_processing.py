import pandas as pd
import numpy as np

import pickle
from pathlib import Path

def filter_sparse_users_and_movies(
    df: pd.DataFrame,
    min_movie_ratings: int = 50,
    min_user_ratings: int = 10,
    save_dir: str = "artifacts"
) -> pd.DataFrame:
    """
    Filters out movies and users with very few ratings to reduce noise,
    and saves valid movie and user IDs.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        min_movie_ratings (int): Minimum ratings a movie must have to be kept.
        min_user_ratings (int): Minimum ratings a user must have to be kept.
        save_dir (str): Directory to save valid IDs.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Filter movies
    movie_counts = df['movie_id'].value_counts()
    valid_movies = set(movie_counts[movie_counts >= min_movie_ratings].index)
    df = df[df['movie_id'].isin(valid_movies)]

    # Filter users
    user_counts = df['customer_id'].value_counts()
    valid_users = set(user_counts[user_counts >= min_user_ratings].index)
    df = df[df['customer_id'].isin(valid_users)]

    # Save valid sets
    with open(save_path / "valid_movies.pkl", "wb") as f:
        pickle.dump(valid_movies, f)

    with open(save_path / "valid_users.pkl", "wb") as f:
        pickle.dump(valid_users, f)

    return df


def filter_valid_ratings(
    df: pd.DataFrame,
    min_rating: int =1, 
    max_rating: int =5
    ) -> pd.DataFrame:
    """
    Filter a DataFrame to keep only rows where 'rating' is between a min and a max rating (inclusive).

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'rating' column.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df['rating'].between(min_rating, max_rating)]


def convert_columns_to_string(
    df: pd.DataFrame, 
    columns: list[str]
    ) -> pd.DataFrame:
    """
    Convert specified columns of a DataFrame to string dtype using NumPy only if they are not already strings.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list of str): List of column names to convert.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to string dtype if needed.
    """
    for col in columns:
        if df[col].dtype != 'string':
            df[col] = np.array(df[col], dtype=str)
    return df


def filter_unseen(df, valid_users, valid_movies):
    mask_users = df["customer_id"].isin(valid_users)
    mask_movies = df["movie_id"].isin(valid_movies)
    filtered_df = df[mask_users & mask_movies].copy()

    dropped = len(df) - len(filtered_df)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with unknown users or movies.")

    return filtered_df