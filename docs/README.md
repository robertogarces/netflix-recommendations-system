# Netflix SVD Recommender — Documentation

A movie recommender on the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
using Funk matrix factorization (Surprise `SVD`). It predicts ratings and ranks
each user's top movies.

| Doc | What's inside |
|-----|---------------|
| [pipeline.md](pipeline.md) | The four stages (preprocess → train → evaluate → recommend), how to run them, inputs/outputs |
| [preprocessing.md](preprocessing.md) | The raw Netflix format, the parsing strategy, why Polars, the sparsity filters |
| [model.md](model.md) | The SVD model, hyperparameters and why, how scoring works |
| [evaluation.md](evaluation.md) | Temporal split, warm/cold users, metrics and current numbers |
| [recommendation.md](recommendation.md) | Serving: the two modes, qualifying users, seen-item filtering, output |
| [decisions.md](decisions.md) | Why this stack: the NCF / SVD / SVD++ comparison and the overfitting analysis |
| [testing.md](testing.md) | The test suite: how to run it, the oracle approach, and what each test pins |

Code lives in `src/` (one module per stage) plus `app/dashboard.py`. All
configuration is centralized in `config/settings.yaml`; file paths in `config/paths.py`.
