# Model

The recommender is a **biased matrix factorization** model (Funk SVD), trained
with SGD by the [Surprise](https://surpriselib.com/) library. This document covers
what the model is, how it learns, what it can and cannot represent, and how it is
scored and served.

---

## 1. Intuition: latent factors

Every user and every movie is described by a vector of `f` learned numbers — a
**latent factor** vector. You can read the dimensions as unnamed "taste axes"
(e.g. *grittiness*, *dialogue-heavy*, *blockbuster-ness*); the model discovers them
from ratings, they are not labeled. A user's vector `p_u` says how much they lean
toward each axis; a movie's vector `q_i` says how much it expresses each axis. Their
dot product `q_iᵀ p_u` is high when a user's tastes line up with a movie's content,
and that is the personalized part of the score.

On top of the dot product sit three **offsets** that absorb everything not specific
to a user–movie *pair*: a global mean, a per-user bias, and a per-movie bias.

---

## 2. The prediction

```
r̂(u, i) = μ + b_u + b_i + q_iᵀ p_u
```

| term | shape | what it absorbs |
|------|-------|-----------------|
| `μ` | scalar | global mean rating (here **3.59**) — the dumbest possible baseline |
| `b_u` | scalar/user | "this user rates everything ~0.4 high" |
| `b_i` | scalar/movie | "this movie is universally loved / polarizing" |
| `q_iᵀ p_u` | dot of two `f`-vectors | the *interaction*: does this user like **this kind** of movie |

The biases matter more than they look: they let the factors model only the
*residual* signal above a sensible baseline, instead of wasting capacity
re-learning that some users are generous and some movies are hits. Predictions are
finally clamped to the rating scale `[1, 5]`.

---

## 3. How it learns

### Objective
Surprise fits the parameters by minimizing **regularized squared error over the
observed ratings only** — not the full user×movie matrix:

```
min  Σ        (r_ui − μ − b_u − b_i − q_iᵀ p_u)²  +  λ (b_u² + b_i² + ‖p_u‖² + ‖q_i‖²)
   (u,i) ∈ R
```

> **Naming caveat (worth internalizing):** despite the name, this is *not* the
> linear-algebra SVD decomposition. Classic SVD requires a fully-observed matrix;
> our rating matrix is ~99% missing. This is gradient-descent matrix factorization
> over the *observed* entries — the "SVD" label is historical, from Simon Funk's
> 2006 Netflix Prize blog post. Treating it as the algebraic SVD will mislead you.

### Optimization (SGD)
The model is fit by stochastic gradient descent: one pass = one epoch; each epoch
visits every training rating once. For each observed `r_ui`, with prediction error
`e_ui = r_ui − r̂_ui`, the four parameters touched by that rating are nudged along
the negative gradient:

```
e_ui = r_ui − r̂_ui
b_u ← b_u + γ · (e_ui − λ · b_u)
b_i ← b_i + γ · (e_ui − λ · b_i)
p_u ← p_u + γ · (e_ui · q_i − λ · p_u)
q_i ← q_i + γ · (e_ui · p_u − λ · q_i)
```

with learning rate `γ = lr_all` and regularization `λ = reg_all`. Read the factor
updates as: *move each user vector toward the movie vectors they rated highly, and
vice-versa, shrinking both toward zero a little each step* (the `−λ·θ` term is the
gradient of the L2 penalty — weight decay).

Initialization: factors are drawn `~ N(0, 0.1²)`, biases start at 0, and `μ` is the
fixed training-set mean. The run is seeded (`random_state = seed`) for
reproducibility.

---

## 4. Hyperparameters (`config/settings.yaml → svd`)

| Param | Value | Lever it controls |
|-------|-------|-------------------|
| `n_factors` (`f`) | 50 | **capacity** — more dimensions fit finer structure but overfit faster |
| `reg_all` (`λ`) | 0.05 | **regularization** — shrinks factors/biases; the main overfitting knob |
| `lr_all` (`γ`) | 0.005 | SGD step size |
| `n_epochs` | 20 | passes over the data; more = more fitting (a soft capacity knob too) |

`n_factors` and `reg_all` together set the bias–variance tradeoff. The Surprise
defaults (`f=100`, `λ=0.02`) overfit hard on this data — train RMSE 0.71 vs test
~1.0. Halving the factors and ~2.5×-ing the regularization (50 / 0.05) closes the
train/test gap to a healthy ~0.16 with **no loss in test accuracy**. The full
analysis — and why test RMSE is near its floor regardless — is in
[decisions.md](decisions.md#4-overfitting-analysis).

---

## 5. What it captures — and what it does not

**Captures:** linear user–item affinity (the dot product) plus additive user/movie
offsets. That is a strong, well-understood baseline and most of the achievable
signal on explicit-rating data.

**Blind spots** (by construction):

- **No temporal dynamics.** The rating *date* is used only to make the train/test
  split — the model has no notion that tastes or a movie's reception drift over
  time. It predicts a single static `r̂(u, i)`.
- **No content/side features.** A movie is just an id with a learned vector; there
  is no genre, cast, or text to fall back on for a brand-new item or user.
- **Symmetric cold start.** An unseen user or item therefore drops to `μ + b`
  (popularity) — there is no feature-based warm start. ~43% of test rows are cold
  users; see [evaluation.md](evaluation.md).
- **No implicit feedback.** *Which* movies a user chose to rate is itself signal;
  SVD ignores it. SVD++ adds it — we tried it and it overfit under the temporal
  split (see [decisions.md](decisions.md)).

These are not bugs; they are the model class. They define the ceiling and tell you
where the next gain would come from (features, sequence models), not more SVD tuning.

---

## 6. Scoring & serving (`src/model.py`)

`model.py` is a thin layer that turns the learned arrays (`pu`, `qi`, `bu`, `bi`,
`μ`) into predictions. We bypass Surprise's per-pair Python `predict()` and score
with vectorized NumPy, because at evaluation/serving scale the Python loop is the
bottleneck.

### Two scoring paths

- **`estimate_pairs(...)`** — pointwise: `r̂` for aligned `(user, item)` arrays.
  Used by evaluate over millions of test rows. The factor gather `pu[users]` builds
  an `(n_rows, f)` array, so it is processed in **chunks** to keep peak memory
  bounded — a one-shot gather over ~40M rows would allocate tens of GB and OOM.
- **`score_all_items(...)`** — full-catalog: every item's score for one user, as a
  single matmul `qi @ p_u` (BLAS gemv) plus the bias vector. Used by recommend.

### Complexity & footprint (current 10% model)

| Quantity | Cost / size |
|----------|-------------|
| Train | `O(n_ratings · f · n_epochs)` ≈ 7.9M · 50 · 20 ≈ **8×10⁹** SGD updates (~4 min, single-threaded) |
| Parameters | `(n_users + n_items)·(f + 1) + 1` ≈ **2.9M** floats (~23 MB) |
| Inference, 1 pair | `O(f)` = 50 multiply-adds |
| Inference, 1 user × full catalog | `O(n_items · f)` ≈ 17,209 · 50 ≈ **0.86M** ops — trivial |

> **Serving smell:** the model *itself* is ~23 MB, but `models/svd_model.pkl` is
> **246 MB**, because `surprise.dump` pickles the entire trainset (every rating's
> id index) alongside the parameters. For real serving you would persist only
> `pu/qi/bu/bi/μ` + the id maps and drop the trainset — a ~10× artifact shrink.

### Two id spaces
The data speaks **raw** ids (`customer_id`, `movie_id`); Surprise indexes its
arrays by compact **inner** ids (`0..n-1`). Every helper translates raw→inner in
(`_raw_to_inner`) and inner→raw out (`item_raw_ids`). This is the only real friction
in the module; the placeholder/mask logic in `estimate_pairs` exists solely to make
that translation vectorizable over unknown ids.

### Cold start
An unknown user has no `p_u`/`b_u`, so the score collapses to `μ + b_i` for every
item — a **popularity ranking**, exactly what `SVD.estimate` returns when it knows
nothing about the user. It is the error floor no collaborative-filtering model can
beat for that user.

### Retrieval note
Brute-force scoring the whole catalog per user is fine here (17k items). At
**millions** of items it would not be: since the only user-dependent term is
`q_iᵀ p_u`, top-K retrieval becomes a **maximum-inner-product search** over `qi`,
which you would serve with an ANN index (FAISS/ScaNN) and then re-rank with the
biases. We brute-force because two-stage retrieval only pays off well past this
catalog size.
