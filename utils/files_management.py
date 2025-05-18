from pathlib import Path
import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')


def fix_csv_with_commas_in_text(
    input_path: Path, 
    output_path: Path
    ) -> None:
    """
    Fixes a malformed CSV file where fields are separated by commas,
    but the third field (typically a text column) may contain commas itself.

    This function replaces only the first two commas in each line with semicolons (;),
    preserving commas that appear within the text of the third column.

    Parameters:
    - input_path (Path): Path to the original CSV file.
    - output_path (Path): Path where the corrected CSV will be saved.

    Example:
    >>> fix_csv_with_commas_in_text(
            input_path=Path("data/movie_titles.csv"),
            output_path=Path("data/movie_titles_fixed.csv")
        )
    """
    with open(input_path, 'r', encoding='latin1') as infile, \
         open(output_path, 'w', encoding='latin1') as outfile:
        
        for line in infile:
            line = line.rstrip('\n')

            # Split line into a maximum of 3 parts to isolate the first two separators
            parts = line.split(',', 2)

            if len(parts) == 3:
                # Replace the first two commas with semicolons
                new_line = ';'.join(parts)
            else:
                # If the line doesn't contain exactly 3 parts, write it as-is
                new_line = line

            outfile.write(new_line + '\n')


def load_netflix_data(
    file_path: str, 
    has_ratings: bool = True, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load Netflix data from raw format into a cleaned DataFrame.

    Parameters:
        file_path (str): Path to the raw Netflix data file.
        has_ratings (bool): Whether the file includes ratings (training) or not (test).
        verbose (bool): If True, prints number of loaded rows and a sample.

    Returns:
        pd.DataFrame: A DataFrame with appropriate columns depending on `has_ratings`.
    """
    data = []
    current_movie_id = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                try:
                    if has_ratings:
                        customer_id, rating, date = line.split(",")
                        data.append((
                            current_movie_id,
                            int(customer_id),
                            float(rating),
                            date
                        ))
                    else:
                        customer_id, date = line.split(",")
                        data.append((
                            current_movie_id,
                            int(customer_id),
                            date
                        ))
                except ValueError:
                    continue

    if has_ratings:
        df = pd.DataFrame(data, columns=["movie_id", "customer_id", "rating", "date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    else:
        df = pd.DataFrame(data, columns=["movie_id", "customer_id", "date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    if verbose:
        print(f"[INFO] Loaded {len(df):,} rows from {file_path}")
        print(df.head())

    return df


def load_multiple_netflix_files(
    file_paths: list[str], 
    save_path: str = None, 
    verbose: bool = True,
    has_ratings: bool = True
    ) -> pd.DataFrame:
    """
    Load and merge multiple Netflix data files into one DataFrame.

    Parameters:
        file_paths (list of str): List of raw Netflix data file paths.
        save_path (str, optional): If provided, saves the merged DataFrame to this path as a .parquet file.
        verbose (bool): If True, prints progress messages.

    Returns:
        pd.DataFrame: Merged DataFrame from all provided files.
    """
    dfs = []
    for path in file_paths:
        if verbose:
            print(f"\n[INFO] Processing: {path}")
        df = load_netflix_data(path, verbose=verbose, has_ratings=has_ratings)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    if save_path:
        merged_df.to_parquet(save_path, index=False)
        if verbose:
            print(f"\n[INFO] Merged data saved to: {save_path}")

    if verbose:
        print(f"\n[INFO] Total rows loaded: {len(merged_df):,}")

    return merged_df


def load_model(path: Path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_data(data_path: str, sample_fraction: float):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)

    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42)
        logger.info(f"Sampled {sample_fraction*100}% of data, resulting in {len(data)} rows")
    else:
        logger.info(f"Using 100% of data, {len(data)} rows")
    return data
