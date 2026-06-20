# Testing

What the test suite guards, how to run it, and what each test pins down. The
suite focuses on the **vectorized numerical code** — the parts that are fast but
easy to get subtly wrong — not the I/O glue around them.

---

## Running the tests
```
make install-dev      # installs pytest (+ ruff, pre-commit) and registers hooks
make test             # or: pytest
```
Configuration lives in `pytest.ini`: `pythonpath = .` puts the repo root on
`sys.path` so tests can `import src` without installing the project as a package,
and `testpaths = tests` is where pytest looks.

---

## Layout
```
tests/
  conftest.py         # shared fixtures (tiny_model: a small fitted SVD)
  test_metrics.py     # precision_recall_at_k (src/evaluate.py)
  test_model.py       # estimate_pairs, item_raw_ids (src/model.py)
  test_recommend.py   # top_k_for_user        (src/recommend.py)
```
`conftest.py` holds fixtures pytest injects by name — currently `tiny_model`, a
small Surprise SVD fitted on ~100 synthetic ratings (session-scoped, read-only),
so the scoring tests get a real fitted model without touching the 100M-row dataset.

---

## Approach: pin vectorized code against a simple oracle
The risky functions here replace an obvious per-row Python loop with a single
vectorized NumPy pass (`lexsort`, `np.add.reduceat`, `argpartition`, `einsum`).
They are fast at millions of rows, but the index arithmetic is easy to get wrong
in a way that still *runs* and returns plausible-looking numbers. So the pattern is:

1. **Oracle test** — reimplement the function the slow, obviously-correct way (a
   plain per-user loop) and assert the vectorized result equals it across many
   randomized inputs. The naive version is the spec.
2. **Golden cases** — one or two tiny, hand-computed examples, so a failure points
   at a concrete expectation (and so the worked examples in the docstrings stay true).
3. **Edge cases** — the boundaries where the index math or the metric definition
   bites: empty groups, `k` larger than a group, ties.

When an exact oracle would be fragile — e.g. clipped scores that tie at the
rating-scale boundary — a test instead asserts the **properties** that define a
correct result (sorted, complete, nothing left out beats anything kept). Property
checks hold no matter how ties happen to break.

---

## Coverage

### `precision_recall_at_k` — `tests/test_metrics.py`
The metric is explained in [evaluation.md](evaluation.md#precisionk-and-recallk); the
implementation (`src/evaluate.py`) sorts the whole array once with `lexsort` and sums
per-user with `np.add.reduceat` instead of looping. Five tests:

| Test | Pins | Why it matters |
|------|------|----------------|
| `test_matches_docstring_worked_example` | the k=1 example from the docstring → `1.0 / 0.5` | a readable anchor; if a refactor breaks it, the worked example in the code has become a lie |
| `test_matches_reference_oracle` | vectorized == naive per-user loop over 5 seeds × 3 `k` × 3 `threshold` (45 cases) | the core guarantee — the `lexsort`/`reduceat` trick computes exactly what the obvious loop does |
| `test_user_with_no_relevant_items_is_excluded` | users with zero relevant items are dropped, not scored as 0 | recall is `0/0` for them; counting them as 0 would silently deflate the metric |
| `test_precision_divides_by_fixed_k` | precision = hits / `k`, even when a user has fewer than `k` items | locks the convention — "fixing" it to divide by `min(k, n_items)` would move every reported number |
| `test_ties_in_score_break_by_input_order` | equal scores keep input order (stable sort), which decides top-K | makes top-K membership under ties deterministic; relies on `lexsort` being stable |

The last three double as **executable documentation of deliberate choices**: the
fixed-`k` denominator, the exclusion of no-relevant users, and stable tie-breaking
are decisions, not accidents — the tests are where they are written down and defended.

---

### `estimate_pairs` — `tests/test_model.py`
`estimate_pairs` (`src/model.py`) is the vectorized, chunked equivalent of looping
`surprise.SVD.predict` over `(user, item)` pairs: it scores millions of rows with a
few NumPy passes instead of a per-row Python loop. **Surprise's own `predict` is the
oracle**, and the `tiny_model` fixture supplies a real fitted model with known ids
(users 0–19, items 0–14) and guaranteed-unknown ids (e.g. 9999). Four tests:

| Test | Pins | Why it matters |
|------|------|----------------|
| `test_matches_surprise_predict` | vectorized output == `algo.predict(u, i).est` over 200 known/unknown pairs | validates the whole formula (`μ + b_u + b_i + q_i·p_u`), the clamping and the unknown-id fallback against the library itself |
| `test_chunking_is_invariant` | `chunk_size=7` (8 chunks) == single pass, bit-for-bit | regression guard for the OOM fix — chunking must be a *memory* optimization only, never change a result (catches off-by-one at the chunk seams) |
| `test_unknown_ids_fall_back_like_surprise` | both unknown → `μ`; warm user only → `μ + b_u`; warm item only → `μ + b_i` | an executable spec of the three cold-start branches, readable without running Surprise in your head |
| `test_clamps_scores_to_rating_scale` | a prediction forced past the ceiling clips to the scale max | makes the clamp actually fire (via an inflated bias on a `deepcopy`, so the shared fixture stays clean) |

The two oracle-style tests are complementary: `test_matches_surprise_predict` runs in
a single chunk (200 rows, well under the 2M default), so it checks the **math**; only
`test_chunking_is_invariant` exercises the **chunk boundaries**.

---

### `top_k_for_user` — `tests/test_recommend.py`
`top_k_for_user` (`src/recommend.py`) turns a user's catalog scores into a ranked
top-K: it scores every item (`score_all_items`), masks already-seen items to `-inf`,
selects the top K with `argpartition`, and decodes inner ids back to raw `movie_id`s.
An exact full-sort oracle is fragile here (clipped scores can tie at the scale
boundary), so the helper `_assert_valid_topk` checks the **properties** of a correct
top-K instead: right length, sorted descending, no seen items, scores that decode to
the right item, and a frontier check (nothing left out outscores anything kept). Four
tests drive it through the helper:

| Test | Pins | Why it matters |
|------|------|----------------|
| `test_warm_user_returns_correct_topk` | a warm user gets a valid, correctly-decoded top-K, segment `"warm"` | the core path: argpartition ranking + inner→raw decoding |
| `test_seen_items_are_excluded` | items passed as `seen` never appear, and the rest is still a valid top-K | the seen-masking (raw→inner→ -inf) that keeps recommendations fresh |
| `test_k_larger_than_catalog_returns_all_items` | `k` beyond the catalog size returns every item, once each | the `k = min(k, n_items)` clamp, with no duplicates |
| `test_cold_user_still_gets_recommendations` | an unknown user gets segment `"cold"` and a valid popularity-ranked top-K | cold users still receive K results, ranked by the popularity fallback |

---

### `item_raw_ids` — `tests/test_model.py`
`item_raw_ids` (`src/model.py`) builds the inner→raw item-id array used to decode
top-K results back to `movie_id`s. One test, `test_item_raw_ids_inverts_the_inner_mapping`,
checks it is the **exact inverse** of the trainset's raw→inner map in both directions,
plus its shape and dtype. It is deliberately independent: `top_k_for_user`'s tests use
this array on both sides of their assertions, so a bug *in* it would hide there; this
test cross-checks it against the trainset's own map instead.
