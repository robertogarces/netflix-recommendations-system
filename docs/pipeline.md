# Pipeline

Four stages, each a standalone module run with `python -m src.<stage>`. They
communicate through files on disk (paths in `config/paths.py`), so any stage can
be re-run independently without re-running the others.

Each stage below links to its deep-dive doc; for the *why* behind the design — the
model choice and the overfitting analysis — see [decisions.md](decisions.md).

```
raw txt ──preprocess──▶ processed.parquet ──train──▶ svd_model.pkl ──evaluate──▶ metrics.json
                                               │                 └─────────────▶ test_set.parquet
                                               └──────recommend──▶ recommendations.parquet
```

## 1. Preprocess — `python -m src.preprocessing`
Parses the raw `combined_data_{1..4}.txt` files (interleaved movie-header and
rating-row format) into one clean, typed, filtered table.

- **Reads:** `data/raw/combined_data_*.txt`
- **Writes:** `data/processed/processed_data.parquet`, `artifacts/valid_users.pkl`, `artifacts/valid_movies.pkl`
- Full dataset after filtering: ~100M ratings, ~464k users, ~17.7k movies.

→ **Deep dive: [preprocessing.md](preprocessing.md)** — the raw format, the parsing strategy, why Polars over pandas, and the sparsity filters.

## 2. Train — `python -m src.train`
Samples a fraction of users, splits temporally, fits Surprise `SVD`, saves the model.

- **Reads:** `data/processed/processed_data.parquet`
- **Writes:** `models/svd_model.pkl`, `outputs/test_set.parquet`
- Samples `data_sample_fraction` (0.1) of **users** — each keeps their full history — then a **temporal** split holds out the newest `test_size` (0.2) of ratings as test.
- Prints per-epoch progress and `Train RMSE | Test RMSE (gap)`; logs params and metrics to MLflow (experiment `Netflix_SVD`).

→ **Deep dive: [model.md](model.md)** — the SVD model, its hyperparameters and why, and how scoring works.

## 3. Evaluate — `python -m src.evaluate`
Scores the held-out test set and reports quality metrics.

- **Reads:** `models/svd_model.pkl`, `outputs/test_set.parquet`
- **Writes:** `outputs/metrics.json` (+ MLflow, attached to the training run)
- Reports RMSE (overall / warm / cold) and Precision@K / Recall@K.

→ **Deep dive: [evaluation.md](evaluation.md)** — the temporal split, warm/cold segmentation, and how to read each metric.

## 4. Recommend — `python -m src.recommend`
Generates top-K recommendations from the fitted model. Two modes:

- **Batch (default):** top-K for a sample of `qualifying.txt` users → `outputs/recommendations.parquet`.
- **Inspect:** `--user-id <id>` prints one user's top-K enriched with movie titles. `--k <n>` overrides the count.

→ **Deep dive: [recommendation.md](recommendation.md)** — the two modes, qualifying users, seen-item filtering, warm/cold serving, and the output schema.

## Dashboard — `streamlit run app/dashboard.py`
A single-page Streamlit app: model performance (read from `metrics.json`) plus a
live user explorer that reuses `src.recommend` — so the CLI, the dashboard, and a
future API all share one scoring core.

## Run it end to end
```bash
python -m src.preprocessing   # once; slow (parses ~100M raw rows)
python -m src.train
python -m src.evaluate
python -m src.recommend
```
