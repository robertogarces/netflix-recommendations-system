import pandas as pd
import numpy as np

def filter_sparse_users_and_movies(
    df: pd.DataFrame,
    min_movie_ratings: int = 50,
    min_user_ratings: int = 10
) -> pd.DataFrame:
    """
    Filters out movies and users with very few ratings to reduce noise.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Count movie ratings
    movie_counts = df['movie_id'].value_counts()
    valid_movies = movie_counts.index[movie_counts >= min_movie_ratings]

    # Filter once by movies
    df = df[df['movie_id'].isin(valid_movies)]

    # Count user ratings on the filtered set
    user_counts = df['customer_id'].value_counts()
    valid_users = user_counts.index[user_counts >= min_user_ratings]

    # Final filter by users
    df = df[df['customer_id'].isin(valid_users)]

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