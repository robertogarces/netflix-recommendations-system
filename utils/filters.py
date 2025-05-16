import pandas as pd
import numpy as np

def filter_sparse_users_and_movies(
    df: pd.DataFrame, 
    min_movie_ratings: int = 50, 
    min_user_ratings: int = 10
    ) -> pd.DataFrame:
    """
    Filters out movies and users with very few ratings to reduce noise.

    Parameters:
        df (pd.DataFrame): The original ratings DataFrame. Must contain 'movie_id' and 'customer_id' columns.
        min_movie_ratings (int): Minimum number of ratings required for a movie to be kept.
        min_user_ratings (int): Minimum number of ratings required for a user to be kept.

    Returns:
        pd.DataFrame: Filtered DataFrame with less sparse movies and users.
    """
    # Convert columns to NumPy arrays for faster processing
    movie_ids = df['movie_id'].values
    user_ids = df['customer_id'].values

    # Get counts using NumPy (faster than pandas.value_counts)
    movie_unique, movie_counts = np.unique(movie_ids, return_counts=True)
    user_unique, user_counts = np.unique(user_ids, return_counts=True)

    # Create sets of allowed IDs for fast lookup
    valid_movies = set(movie_unique[movie_counts >= min_movie_ratings])
    valid_users = set(user_unique[user_counts >= min_user_ratings])

    # Use NumPy boolean indexing
    mask = np.isin(movie_ids, list(valid_movies)) & np.isin(user_ids, list(valid_users))
    return df[mask]



def filter_valid_ratings(
    df: pd.DataFrame,
    min_rating: int =1, 
    max_rating: int =5
    ) -> pd.DataFrame:
    """
    Quickly filter a DataFrame to keep only rows where 'rating' is between 1 and 5 inclusive.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'rating' column.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df['rating'].between(min_rating, max_rating)]
