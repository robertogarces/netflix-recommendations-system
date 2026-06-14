# Decisions & experiment log

The project's decision record: *why* it looks the way it does, and which
alternatives were tried and dropped. The code shows the *what*; this captures the
*why* — including approaches no longer in the tree.

> **Provenance — what is verifiable.** The codebase today is a single Surprise SVD.
> Two earlier approaches are referenced below, with different evidence status:
> - The original **NeuMF / NCF** model is in git history (`git show d824627:src/model.py`),
>   and its test metrics were committed (`git show d824627:outputs/metrics.json`).
>   That row is reproducible.
> - The **SVD / SVD++** exploration happened *during* the migration. That code was
>   never committed (the SVD++ model existed only in a working tree) and its MLflow
>   runs are not tracked in the repo. Those numbers are **indicative** — recorded as
>   the rationale for the decision, not a reproducible benchmark.
>
> Everything from section 3 on (the Surprise migration, the overfitting tuning) is
> reproducible from the current tree.

---

## 1. From notebooks to a pipeline
The project began as exploratory notebooks (`notebooks/`: EDA, preprocessing,
model). The first model was a neural one — **NeuMF (NCF)**: a generalized
matrix-factorization branch plus an MLP over the concatenated user/item embeddings,
in PyTorch, later refactored into the staged `src/` pipeline.

## 2. Why plain SVD — not NCF, not SVD++
While the pipeline was still PyTorch, three collaborative-filtering models were
trained on the same temporal split to see what the added complexity actually buys.

> Indicative results — PyTorch implementations, 3-way split (val 0.1 / test 0.2),
> 10% sample. Only the **NCF** row is committed (`d824627`); the SVD / SVD++ rows are
> from exploratory runs that were not retained, so they are not reproducible from
> this repo. The *current* Surprise SVD's real, reproducible numbers live in
> [evaluation.md](evaluation.md).

| Model | test RMSE | warm | cold |
|-------|-----------|------|------|
| NCF (NeuMF: MF + MLP) | 1.0264 | 0.9380 | 1.1001 |
| SVD (matrix factorization) | 1.0177 | 0.9503 | 1.0748 |
| SVD++ (SVD + implicit feedback) | 1.1106 | 0.9539 | 1.2352 |

Findings (the durable part):

- **NCF's MLP barely helps.** On warm users it edged plain SVD by only ~1.3% RMSE —
  a small return for a non-linear network and the training/serving complexity that
  comes with it.
- **SVD++ overfit under the temporal split.** Its implicit-feedback term assumes the
  items being scored are in the user's interaction set `N(u)`. A temporal split puts
  *future* items in the test set, absent from `N(u)`, so the assumption breaks and
  the extra capacity hurt — worst on cold RMSE, where the implicit factors stole
  calibration away from the item biases. (Implicit feedback paid off in the original
  Netflix Prize precisely because the qualifying items *were* in `N(u)`.)

**Decision: plain SVD.** Competitive with NCF, far simpler, and better-behaved than
SVD++ on this data — a stronger baseline for much less machinery.

## 3. Migrating to Surprise
With SVD chosen, the hand-rolled PyTorch model (manual SGD loop, device handling,
embedding tables) was replaced by Surprise's `SVD` — same math, a fraction of the
code (commit `7f0f0f6`). The library owns the factorization; the repo keeps only
loading, scoring, and metrics. PyTorch, Docker, DVC and the conda env were removed
at the same time (commit `df4d26a`), to be rebuilt when actually needed.

## 4. Overfitting analysis
*(Reproducible: re-run `src.train` at different `svd` settings.)*

Surprise's defaults (`n_factors=100`, `reg_all=0.02`) gave a train/test RMSE gap of
+0.28 (train 0.71, test ~1.0). A sweep over `n_factors` × `reg_all` (on a 2%
subsample) showed two things:

- The gap closes easily with more regularization: `50 / 0.05` → gap +0.16, train 0.83.
- **But test RMSE barely moved** (~1.0 across every setting).

So the "overfitting" was **cosmetic**: the model memorized the training ratings, but
that was not costing test accuracy. Test error here is floored by **temporal drift**
(predicting the future) and the **~43% cold-start users**, not by under-regularization.
The tuned `50 / 0.05` is kept anyway — same test accuracy, but an honest, leaner,
better-calibrated model (commit `7f214f5`).

The real lever for a lower test RMSE is **more data** (raise `data_sample_fraction`),
not more regularization — though Surprise is single-threaded, so large fractions are
slow, which is exactly why `estimate_pairs` chunks its scoring (see [model.md](model.md)).
