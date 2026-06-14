# Evaluation

How the model is judged, and — more importantly — how to read each number without
fooling yourself.

---

## Temporal split (train past, test future)
`train` splits ratings by **date**: the oldest `1 − test_size` go to train, the
newest `test_size` (0.2) become the test set. This mirrors deployment — fit on
history, predict ratings that haven't happened yet — and avoids the leakage a
random split causes (a user's *later* rating helping predict their *earlier* one,
which never happens in production).

The price of realism is **cold users**: people whose first rating falls after the
cut-off have no training history at all.

### Warm vs cold users
`train` flags each test row `is_cold_user`. The distinction is not cosmetic:

- **Warm users** have a learned factor vector → genuinely personalized predictions.
  This is the number model changes can move.
- **Cold users** fall back to `μ + b_i` (popularity) — the model has never seen
  them. Their error is a fixed floor, independent of how good the factors are.

At the current 10% sample, ~43% of test rows are cold. So **every metric is reported
overall *and* warm-only**, and the warm figure is the one to optimize.

---

## Two families of metric, measuring different things
The product goal is twofold — *predict ratings* and *rank the right movies at the
top* — so we track two kinds of metric, and they do **not** move together:

- **RMSE** is a *regression* metric: how close is the predicted rating to the true
  one. It is what SGD optimizes.
- **Precision@K / Recall@K** are *ranking/retrieval* metrics: did the relevant items
  end up at the top. This is closer to the actual product objective.

**RMSE judges the predicted *value*; ranking judges only the *order*.** A model can
nail the order while getting every value wrong — so the two diverge. Popularity is
the canonical case. Take one user's three test movies, scored by pure popularity
(`μ + b_i`, the same score the model gives a cold user):

| movie | true rating | predicted (popularity) |
|-------|-------------|------------------------|
| A (very popular) | **5** | 4.1 |
| B (popular) | 2 | 3.8 |
| C (niche) | 1 | 3.3 |

- **RMSE ≈ 1.76** — terrible: every prediction sits around 3.3–4.1, nowhere near the
  true 1 / 2 / 5.
- **Ranking is perfect** — sorted by score, A (the only relevant, 5-star item) lands
  at rank 1, so Precision@1 = Recall@1 = 1.0. Ranking only needed A to *outscore* B
  and C, not to be predicted accurately.

It ranks well because a user's 5-star movies tend to *be* popular movies, so sorting
by popularity floats their likely favourites up. This is exactly why **cold users —
who get pure popularity — crater on RMSE yet stay respectable on Precision/Recall**:
note the overall Precision@10 below (which includes cold users) actually *exceeds*
the warm-only figure.

---

## RMSE
```
RMSE = sqrt( mean( (r̂ − r)² ) )        predictions clamped to [1, 5] first
```

- **Units are stars.** Warm RMSE 0.93 ≈ the typical prediction is off by ~0.9 stars
  on a 1–5 scale.
- **Why RMSE, not MAE.** The square penalizes large misses harder than small ones,
  and it is the Netflix Prize's official metric — so it is the historical yardstick
  for this dataset.
- **Reference points (orientation, *not* a comparison).** Netflix's own *Cinematch*
  scored ≈0.951 and the 2009 grand prize was ≈0.857 — both on Netflix's fixed
  quiz/probe set. Our numbers use a temporal split on a 10% sample, so they are
  **not** comparable; treat 0.85–0.95 only as a rough sense of "good" on this data.
- **Read the decomposition, not the headline.** The overall RMSE (0.996) is dragged
  up by the cold floor; the warm RMSE (0.928) is the honest measure of
  personalization quality. Reporting only the overall number would understate the
  model and hide where the error actually lives.

---

## Precision@K and Recall@K
For each user we rank their test items by predicted score, take the top **K**
(=`top_n`=10), and count how many are *relevant*.

```
relevant(u, i)  :=  true_rating(u, i) ≥ relevance_threshold        (= 4.5)

Precision@K = (relevant items in the top-K) / K
Recall@K    = (relevant items in the top-K) / (all relevant items the user has)
```

- **Precision@K** — "of the 10 we put at the top, how many would the user have
  loved." Caps at 1.0 only if the user has ≥K relevant items.
- **Recall@K** — "of everything the user would have loved, how much did we surface
  in the top 10." A short top-K mechanically limits recall when a user has many
  relevant items.
- Raising K trades precision for recall (more slots → more hits, but a lower hit
  *rate*).

### Three caveats that change the interpretation
1. **`threshold = 4.5` means 5-star-only.** Netflix ratings are integers 1–5, so
   `≥ 4.5` collapses to *exactly 5*. These metrics therefore measure "did we surface
   the user's **5-star** movies," a strict bar. Lowering the threshold to 4.0 would
   count 4s as relevant — a softer, more forgiving definition. Worth a deliberate
   choice, not a default.
2. **Ranking is over the user's *rated test items*, not the full catalog.** We sort
   the items the user actually rated in the test window — not all ~17k movies.
   That's the standard offline shortcut (ground truth exists only for rated items),
   but it means these numbers are *optimistic* relative to true top-K-over-catalog,
   and they say nothing about items the user never rated (ratings are
   missing-not-at-random).
3. **Users with zero relevant items are excluded** (recall would be 0/0). With the
   5-star bar, the metric is implicitly over the subpopulation that has at least one
   5-star test rating.

---

## Current numbers (10% user sample)
| Metric | Overall | Warm |
|--------|---------|------|
| RMSE | 0.9958 | 0.9277 |
| RMSE (cold) | 1.0791 | — |
| Precision@10 | 0.4275 | 0.3957 |
| Recall@10 | 0.5662 | 0.6390 |

Train RMSE 0.8313 (gap +0.165). Test set ~1.99M rows, ~57% warm / ~43% cold.

How to read this: the model predicts a warm user's rating to within ~0.93 stars,
and roughly **2 of every 5** of its top-10 picks are movies that user rated 5 stars
in the held-out future. Note the ranking metrics barely separate warm from overall
— consistent with the point above that popularity is a weak *predictor* but a decent
*ranker*. Why the RMSE is near its floor regardless of tuning is covered in
[decisions.md](decisions.md#4-overfitting-analysis).
