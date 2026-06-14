# Recommendation

The serving stage: turn the fitted SVD into delivered top-K movie lists. *How a
single score is computed* lives in [model.md](model.md); this doc is the
*application* logic around it — who we recommend for, what we filter out, the two
run modes, and the output.

## One scoring core, several entry points
Every flow funnels through `top_k_for_user` (score the catalog → mask seen items →
take top-K) and its wrapper `recommend_single_user`. The CLI inspect mode, the batch
precompute, and the Streamlit dashboard are all thin shells over that same core —
and an online API would be too. Change the ranking logic once, and every surface
gets it.

## Two modes
**Batch (default)** — `python -m src.recommend`
Precomputes top-K for a sample of the qualifying users → `outputs/recommendations.parquet`.

**Inspect** — `python -m src.recommend --user-id <id> [--k <n>]`
Prints one user's top-K with titles, labeled personalized (warm) or cold-start
fallback. For spot-checking a single user; `--k` overrides the configured count.

## Who we recommend for: `qualifying.txt`
Netflix shipped `qualifying.txt` as the Prize's submission target — the
`(user, movie, date)` triples whose ratings were withheld, the set competitors had
to predict. We recommend for the **users** in that file (the held-out population),
not the training users.

The file interleaves `movie_id:` header lines with `customer_id,date` rows (no
ratings); `load_qualifying_users` just collects the distinct customer ids. There are
hundreds of thousands, so `qualifying_sample_fraction` (0.01) subsamples them —
purely for legibility and runtime of the demo output, **not** a modeling choice.

## Filtering already-seen items
A recommender should not surface a movie the user has already rated.
`load_seen_items` lazily scans the processed parquet (Polars) and returns, per
requested user, the set of `movie_id`s they have rated. `top_k_for_user` maps those
to inner ids and sets their score to `-inf` before selecting the top-K, so a seen
item can never be returned. One scan covers the whole batch; the inspect path scans
just the one user.

## Warm vs cold at serving time
- **Warm user** (seen in training): the full catalog is scored with
  `μ + b_u + b_i + q_iᵀ p_u` → a genuinely personalized ranking.
- **Cold user** (unknown): no `p_u`/`b_u`, so the score collapses to `μ + b_i` — a
  popularity ranking. The output's `segment` column records which path was taken.
  See [model.md](model.md#cold-start) for why popularity is the right fallback.

Top-K selection uses `np.argpartition` (an O(n) partial selection) and then sorts
only those K — cheap even when scoring all ~17k items per user. At catalog sizes
where brute force stops being cheap this becomes an ANN / maximum-inner-product
problem; see the [retrieval note](model.md#retrieval-note) in model.md.

## Output: `recommendations.parquet`
A tidy long table — one row per `(user, rank)`:

| column | meaning |
|--------|---------|
| `customer_id` | the user |
| `rank` | 1..K, best first |
| `movie_id` | recommended movie |
| `title` | resolved from `movie_titles.csv` |
| `score` | predicted rating, clamped to `[1, 5]` |
| `segment` | `warm` (personalized) or `cold` (popularity) |

At the default 1% qualifying sample × K=10 this is a compact, inspectable file; raise
`qualifying_sample_fraction` for a fuller run.
