"""Preprocessing stage: raw combined_data_N.txt files → clean training parquet.

Outputs:
- data/processed/processed_data.parquet — filtered ratings for model training
- artifacts/valid_users.pkl / valid_movies.pkl — the data contract: IDs that
  survived filtering, used to reject unknown IDs at inference time

Raw customer/movie IDs are kept as-is here; Surprise builds its own internal
index mappings from the training data when the model is fit (see src.train).
"""

import logging
import pickle

import polars as pl
import yaml

from config.paths import ARTIFACTS_PATH, CONFIG_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_raw_file(path) -> pl.DataFrame:
    """Parse one combined_data_N.txt file into a clean DataFrame, fully vectorized.

    The raw file format mixes two line types with no structural separator:

        123:                          ← movie header: declares movie_id for the lines below
        1488844,3,2005-09-06          ← rating: customer_id, rating, date
        822109,5,2005-05-13
        ...
        124:                          ← next movie block starts
        ...

    Strategy: read every line as a raw string, then use Polars expressions to
    detect headers, forward-fill the movie_id down to rating rows, then parse
    the rating rows into typed columns — all without a Python loop.

    A Python loop at ~100M rows would take minutes; this approach runs in seconds
    because Polars executes the expressions in Rust across all rows at once.
    """
    return (
        # Read the file treating each line as a single string in column "line".
        # separator="\x01" (ASCII SOH, never appears in the file) prevents Polars
        # from splitting lines on commas at this stage — we need the raw line first.
        # quote_char=None disables quote handling (movie titles contain quotes).
        pl.read_csv(path, has_header=False, separator="\x01", quote_char=None,
                    new_columns=["line"])

        # Switch to lazy mode: Polars builds a query plan and can push filters
        # down (e.g. skip parsing columns we won't use). .collect() runs it all.
        .lazy()

        # Step 1 — detect header lines. Headers end with ":" (e.g. "123:"),
        # rating lines do not. This boolean column drives the next step.
        .with_columns(is_header=pl.col("line").str.ends_with(":"))

        # Step 2 — extract movie_id from header lines only, then forward-fill.
        # pl.when(...).then(...).otherwise(None) is a vectorized if/else:
        #   - header rows → strip the trailing ":" and cast to Int32
        #   - rating rows → None (will be filled by the next movie_id above them)
        # .forward_fill() propagates the last non-null value downward, so every
        # rating row inherits the movie_id of the header that precedes it.
        .with_columns(
            movie_id=pl.when(pl.col("is_header"))
            .then(pl.col("line").str.strip_chars_end(":").cast(pl.Int32, strict=False))
            .otherwise(None)
            .forward_fill()
        )

        # Step 3 — discard the header rows; only rating rows remain.
        .filter(~pl.col("is_header"))

        # Step 4 — split "customer_id,rating,date" into exactly 3 parts.
        # split_exact(",", 2) produces a Struct with fixed fields: field_0, field_1, field_2.
        # Unlike split() (which returns a variable-length List), split_exact is O(n)
        # and lets us access each part by name in the next step.
        .with_columns(parts=pl.col("line").str.split_exact(",", 2))

        # Step 5 — unpack the Struct fields and cast each to its proper type.
        # .struct.field("field_N") extracts the Nth piece from the split above.
        # strict=False: rows that fail to cast (malformed lines) become null
        # instead of raising an error; drop_nulls() removes them below.
        .select(
            "movie_id",
            customer_id=pl.col("parts").struct.field("field_0").cast(pl.Int32, strict=False),
            rating=pl.col("parts").struct.field("field_1").cast(pl.Float32, strict=False),
            date=pl.col("parts").struct.field("field_2").str.to_date("%Y-%m-%d", strict=False),
        )
        .drop_nulls()  # remove any malformed rows that produced nulls during casting
        .collect()     # execute the full lazy query plan and materialize into memory
    )


def parse_raw_files(num_files: int) -> pl.DataFrame:

    # Parse each raw file into a DataFrame, then concatenate them all together.
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
    """Remove movies and users that have too few ratings to learn from.

    Why filter movies before users?
    A movie with only 5 ratings is too noisy for the model to learn a reliable
    embedding. Removing sparse movies first can push some users below the user
    threshold — which is correct: if a user's history is mostly sparse movies,
    their preference signal is unreliable too. Filtering users first would
    artificially retain those sparse movies.

    Uses a semi-join instead of .isin(): Polars implements it as a hash join
    (O(n)), whereas a naive membership test over a Python list is O(n × m).
    """
    # Build a table of movies that meet the minimum rating count, then keep
    # only the rows whose movie_id appears in that table (semi-join = filter by
    # existence, not by value — no columns from valid_movies are added).
    valid_movies = (
        df.group_by("movie_id").len()
        .filter(pl.col("len") >= min_movie_ratings)
        .select("movie_id")
    )
    df = df.join(valid_movies, on="movie_id", how="semi")
    logger.info(f"After movie filter: {len(df):,} rows | {len(valid_movies):,} movies kept")

    # Same pattern for users, applied after the movie filter.
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
