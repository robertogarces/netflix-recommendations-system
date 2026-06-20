"""Tests for src.preprocessing.

Two functions, both exercised without the real 100M-row dataset:
- parse_raw_file: reads Netflix's mixed "movie_id:" header / rating-row format and
  forward-fills the movie_id down to each rating. Driven by a tiny raw file written
  to a tmp_path.
- filter_sparse: drops movies and users below the rating-count thresholds, in that
  order — which can cascade (removing a sparse movie can push a user under).
"""

import datetime

import polars as pl

from src.preprocessing import filter_sparse, parse_raw_file


def test_parse_forward_fills_movie_id(tmp_path):
    """Each rating row inherits the movie_id of the header above it; types are cast."""
    raw = tmp_path / "combined_data_1.txt"
    raw.write_text(
        "1:\n"
        "1488844,3,2005-09-06\n"
        "822109,5,2005-05-13\n"
        "2:\n"
        "885013,4,2005-10-19\n"
        "3:\n"
        "30878,4,2005-12-26\n"
    )

    df = parse_raw_file(raw)

    assert df.height == 4                                          # header rows dropped
    assert df["movie_id"].to_list() == [1, 1, 2, 3]               # the forward-fill
    assert df["customer_id"].to_list() == [1488844, 822109, 885013, 30878]
    assert df["rating"].to_list() == [3.0, 5.0, 4.0, 4.0]
    assert df["date"].to_list()[0] == datetime.date(2005, 9, 6)

    assert df.schema["movie_id"] == pl.Int32
    assert df.schema["customer_id"] == pl.Int32
    assert df.schema["rating"] == pl.Float32
    assert df.schema["date"] == pl.Date


def test_parse_drops_malformed_rows(tmp_path):
    """Rows that fail to cast (non-numeric id) become null and are dropped."""
    raw = tmp_path / "combined_data_1.txt"
    raw.write_text(
        "1:\n"
        "123,4,2005-01-01\n"
        "abc,4,2005-01-01\n"      # non-numeric customer_id -> null -> dropped
        "456,5,2005-02-02\n"
    )

    df = parse_raw_file(raw)

    assert df.height == 2
    assert df["customer_id"].to_list() == [123, 456]


def _ratings_df(pairs):
    """Build a ratings frame from (movie_id, customer_id) pairs."""
    return pl.DataFrame({
        "movie_id": [m for m, _ in pairs],
        "customer_id": [c for _, c in pairs],
        "rating": [4.0] * len(pairs),
    })


# movie 3 has a single rating; user 102's only other rating is on movie 3.
_CASCADE = [(1, 100), (2, 100), (1, 101), (2, 101), (1, 102), (3, 102)]


def test_filter_sparse_removes_below_threshold():
    """A movie with fewer than min_movie_ratings is removed."""
    out, valid_movies, _ = filter_sparse(_ratings_df(_CASCADE), min_movie_ratings=2, min_user_ratings=2)

    assert set(valid_movies) == {1, 2}                 # movie 3 (1 rating) dropped
    assert set(out["movie_id"].to_list()) == {1, 2}


def test_movie_filter_cascades_to_users():
    """Removing sparse movie 3 leaves user 102 with one rating, below the user
    threshold, so the movie-first order drops user 102 too."""
    out, _, valid_users = filter_sparse(_ratings_df(_CASCADE), min_movie_ratings=2, min_user_ratings=2)

    assert set(valid_users) == {100, 101}
    assert 102 not in valid_users
    assert out.height == 4
    assert set(out["customer_id"].to_list()) == {100, 101}
