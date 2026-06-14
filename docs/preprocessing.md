# Preprocessing

`src/preprocessing.py` turns the raw Netflix Prize files into one clean, typed,
filtered table the rest of the pipeline trains on. It runs once and is the slowest
stage, because it touches the full ~100M ratings.

```
data/raw/combined_data_{1..4}.txt  ──►  data/processed/processed_data.parquet
                                         artifacts/valid_users.pkl
                                         artifacts/valid_movies.pkl
```

## The raw format, and why it needs parsing
The ratings ship as four text files (`combined_data_1..4.txt`, ~2 GB total). They
are **not** CSVs you can read directly — each file interleaves *two different line
types* with no shared column layout:

```
1:                      ← movie header: "movie_id:" — declares the movie for the rows below it
1488844,3,2005-09-06    ← rating row: customer_id, rating, date  (no movie_id!)
822109,5,2005-05-13
885013,4,2005-10-19
...
2:                      ← next movie block begins
2059652,4,2005-09-05
...
```

The catch: a rating row carries no movie id. The movie is only stated once, in the
header line above the block, and applies to **every** row until the next header.
So parsing means: detect header lines, and **propagate** each movie id down onto
its rating rows. That "forward-fill from a header" step is the whole problem — a
plain CSV reader can't do it.

## The parsing strategy (`parse_raw_file`)
Each file is processed as a sequence of vectorized column operations (no Python
loop over rows):

1. **Read every line as a single string.** `pl.read_csv(..., separator="\x01")`
   uses a byte that never appears in the data, so Polars does *not* split on commas
   yet — we need the whole raw line first. (`quote_char=None` disables quote
   handling.)
2. **Flag headers.** `is_header = line.str.ends_with(":")`.
3. **Extract + forward-fill the movie id.** On header rows, strip the trailing `:`
   and cast to int; on rating rows, leave null — then `.forward_fill()` propagates
   the last seen movie id downward:

   | line | is_header | movie_id (after forward-fill) |
   |------|-----------|-------------------------------|
   | `1:` | true | 1 |
   | `1488844,3,2005-09-06` | false | 1 |
   | `822109,5,2005-05-13` | false | 1 |
   | `2:` | true | 2 |
   | `2059652,4,2005-09-05` | false | 2 |

4. **Drop the header rows** (`filter(~is_header)`) — only ratings remain.
5. **Split + type the rest.** `str.split_exact(",", 2)` cuts each line into exactly
   three fields, which are cast to `customer_id: Int32`, `rating: Float32`,
   `date: Date`. Malformed rows fail the cast → become null → `drop_nulls()` removes
   them.

The four files are parsed this way and concatenated (`parse_raw_files`).

## Why Polars instead of pandas
This stage is where Polars earns its place (the rest of the pipeline, working on
the *sampled* data, uses pandas). At ~100M rows:

- **No Python loop.** Header detection, the forward-fill, and the split are native
  Polars expressions executed in Rust across all rows at once. The naive
  "loop line by line and remember the current movie id" would take minutes; this
  runs in seconds.
- **Lazy + columnar.** `.lazy()` lets Polars build a query plan and stream the work
  column-by-column (Arrow memory), instead of materializing intermediate pandas
  frames. That keeps peak RAM manageable on a 2 GB input.
- **Multithreaded** by default, vs pandas' single-threaded eager execution.

This isn't dogma — for a small file pandas would be fine. It's specifically the
100M-row, forward-fill-shaped ETL that makes Polars the right tool here. Downstream,
once the data is sampled to a few million rows, the pipeline switches to pandas
(which Surprise consumes directly).

## Filtering sparse users and movies (`filter_sparse`)
Raw data has a long tail of barely-rated movies and barely-active users that carry
almost no learnable signal. Two thresholds (`config/settings.yaml → preprocessing`)
prune them:

- Drop movies with `< min_movie_ratings` (50).
- *Then* drop users with `< min_user_ratings` (10).

**Order matters:** movies are filtered first, because removing sparse movies can
push a user below the user threshold — which is correct (a user whose history is
mostly obscure movies has an unreliable taste signal). Filtering users first would
keep those sparse movies around.

Both filters use Polars **semi-joins** (a hash join, O(n)) rather than `.isin()`
over a Python list (O(n·m)).

The surviving id sets are saved as `artifacts/valid_users.pkl` / `valid_movies.pkl`
— the "data contract" used to reject unknown ids at inference time.

## Output
`data/processed/processed_data.parquet` — one row per rating, four typed columns:

| column | type | example |
|--------|------|---------|
| `movie_id` | Int32 | 1 |
| `customer_id` | Int32 | 1488844 |
| `rating` | Float32 | 3.0 |
| `date` | Date | 2005-09-06 |

```
shape: (100_394_331, 4)
┌──────────┬─────────────┬────────┬────────────┐
│ movie_id ┆ customer_id ┆ rating ┆ date       │
╞══════════╪═════════════╪════════╪════════════╡
│ 1        ┆ 1488844     ┆ 3.0    ┆ 2005-09-06 │
│ 1        ┆ 822109      ┆ 5.0    ┆ 2005-05-13 │
│ ...      ┆ ...         ┆ ...    ┆ ...        │
└──────────┴─────────────┴────────┴────────────┘
```

~100.4M ratings survive filtering, across ~464k users and ~17.7k movies. Parquet
(columnar, compressed) keeps this ~550 MB on disk and fast to scan column-wise —
`train` reads only the columns it needs.

> Note: index mappings (user→row, movie→row) are **not** built here. Each training
> run builds them from its own (sampled) data, so preprocessing stays independent
> of the experiment configuration. See [model.md](model.md) for the inner↔raw ids.
