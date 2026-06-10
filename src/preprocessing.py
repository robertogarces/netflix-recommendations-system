"""Preprocessing stage: raw combined_data_N.txt files → clean training parquet.

Outputs:
- data/processed/processed_data.parquet — filtered ratings for model training
- artifacts/valid_users.pkl / valid_movies.pkl — the data contract: IDs that
  survived filtering, used to reject unknown IDs at inference time

Index mappings are NOT built here — each training run builds compact mappings
from its own (possibly sampled) data and ships them inside the checkpoint.
"""

import logging
import pickle
import yaml
import polars as pl

from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH, ARTIFACTS_PATH, CONFIG_PATH

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_raw_file(path) -> pl.DataFrame:
    """
    Parse one combined_data_N.txt file, fully vectorized.

    The raw format interleaves movie header lines ("123:") with rating lines
    ("customer_id,rating,date"). Instead of a Python loop, we read every line
    as a single string column and use Polars expressions: extract the movie id
    from header lines, forward-fill it onto the rating lines below it, then
    split the rating lines into columns.
    """
    return (
        pl.read_csv(path, has_header=False, separator="\x01", quote_char=None,
                    new_columns=["line"])
        .lazy()
        .with_columns(is_header=pl.col("line").str.ends_with(":"))
        .with_columns(
            movie_id=pl.when(pl.col("is_header"))
            .then(pl.col("line").str.strip_chars_end(":").cast(pl.Int32, strict=False))
            .otherwise(None)
            .forward_fill()
        )
        .filter(~pl.col("is_header"))
        .with_columns(parts=pl.col("line").str.split_exact(",", 2))
        .select(
            "movie_id",
            customer_id=pl.col("parts").struct.field("field_0").cast(pl.Int32, strict=False),
            rating=pl.col("parts").struct.field("field_1").cast(pl.Float32, strict=False),
            date=pl.col("parts").struct.field("field_2").str.to_date("%Y-%m-%d", strict=False),
        )
        .drop_nulls()
        .collect()
    )


def parse_raw_files(num_files: int) -> pl.DataFrame:
    chunks = []
    for i in range(1, num_files + 1):
        path = RAW_DATA_PATH / f"combined_data_{i}.txt"
        logger.info(f"Parsing {path.name} ...")
        chunks.append(parse_raw_file(path))
    df = pl.concat(chunks)
    logger.info(f"Parsed {len(df):,} rows total")
    return df


def filter_sparse(
    df: pl.DataFrame,
    min_movie_ratings: int,
    min_user_ratings: int,
) -> tuple[pl.DataFrame, list, list]:
    # Movies filter first — removing sparse movies can push some users below threshold
    valid_movies = (
        df.group_by("movie_id").len()
        .filter(pl.col("len") >= min_movie_ratings)
        .select("movie_id")
    )
    df = df.join(valid_movies, on="movie_id", how="semi")
    logger.info(f"After movie filter: {len(df):,} rows | {len(valid_movies):,} movies kept")

    valid_users = (
        df.group_by("customer_id").len()
        .filter(pl.col("len") >= min_user_ratings)
        .select("customer_id")
    )
    df = df.join(valid_users, on="customer_id", how="semi")
    logger.info(f"After user filter:  {len(df):,} rows | {len(valid_users):,} users kept")

    return df, valid_movies["movie_id"].to_list(), valid_users["customer_id"].to_list()


def main():
    with open(CONFIG_PATH / "settings.yaml") as f:
        config = yaml.safe_load(f)
    cfg = config["preprocessing"]

    df = parse_raw_files(config["data"]["num_raw_files"])
    df, valid_movies, valid_users = filter_sparse(
        df, cfg["min_movie_ratings"], cfg["min_user_ratings"]
    )

    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    for name, obj in [
        ("valid_movies.pkl", set(valid_movies)),
        ("valid_users.pkl",  set(valid_users)),
    ]:
        with open(ARTIFACTS_PATH / name, "wb") as f:
            pickle.dump(obj, f)
    logger.info("Saved artifacts: valid_movies, valid_users")

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / "processed_data.parquet"
    df.write_parquet(output_path)
    logger.info(f"Saved processed data: {output_path} | shape: {df.shape}")


if __name__ == "__main__":
    main()
