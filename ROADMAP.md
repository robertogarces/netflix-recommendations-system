# Roadmap

Forward-looking plan for the Netflix Prize recommender (Surprise SVD). The model
and the 4-stage pipeline work; what is missing is the surrounding engineering:
reproducibility, tests, versioning, serving, containers, and docs.

## Current state (2026-06-14)

**Working**
- 4-stage pipeline: `preprocessing` → `train` → `evaluate` → `recommend`
  (`python -m src.<stage>`), wired by files on disk.
- Streamlit dashboard (`app/dashboard.py`): metrics panel + per-user explorer.
- `requirements.txt` — current and clean (Surprise/Polars/MLflow/Streamlit, no
  stale PyTorch).
- MLflow logging to local `mlruns/`.
- Config in `config/settings.yaml`, paths in `config/paths.py`.
- Focused docs in `docs/`.

**Missing / to rebuild**
- No `Makefile`, no tests, no DVC, no serving API, no `Dockerfile`, no
  `docker-compose`, no CI, no lint/pre-commit.
- Root `readme.md` still describes the old PyTorch/DVC/Docker stack.

## Guiding principles

- Keep it simple and honest — match the repo's existing ethos (no filler infra).
- Once a serving API exists, the dashboard should consume it over HTTP instead of
  importing `src.recommend` directly — true decoupling, same "thin wrapper" story
  the CLI/dashboard already tell.

## Phase 0 — Reproducible base

1. **Dependencies.** `requirements.txt` already exists; split into
   `requirements.txt` (runtime) + `requirements-dev.txt` (pytest, ruff, dvc).
   Optionally use `uv` as the installer (`uv pip install -r ...`) without changing
   the file format. _Done when:_ a fresh env installs and runs the pipeline.
2. **Makefile.** Thin wrappers over the existing entry points: `setup`,
   `preprocess`, `train`, `evaluate`, `recommend`, `dashboard`, `test`, `lint`,
   `clean`. _Done when:_ `make train` (etc.) reproduces the manual commands.
3. **Lint/format + pre-commit.** `ruff` (lint + format) behind a pre-commit hook.
   _Done when:_ `pre-commit run --all-files` is clean.

## Phase 1 — Trust the code

4. **Tests (pytest).** Cover the bug-prone, hard-to-eyeball pieces first:
   - `estimate_pairs` chunked output == single-shot output.
   - `precision_recall_at_k` vs. a naive reference implementation.
   - inner↔raw id round-trip (`item_raw_ids` / `_raw2inner`).
   - preprocessing header detection + forward-fill on a tiny fixture.
   - warm/cold segmentation in `top_k_for_user` (seen items masked to -inf).

   _Done when:_ `pytest` is green and runs on small fixtures, not the 100M dataset.

## Phase 2 — Version data & pipeline

5. **DVC.** Wire the stages as a `dvc.yaml` DAG (preprocess → train → evaluate →
   recommend) with `deps`/`outs`/`params` (params sourced from `settings.yaml`).
   Track `data/raw` (large), `data/processed`, `models/`, `outputs/`.
   Remote: start with a **local remote dir** (reproducible demo); cloud (S3/GDrive)
   optional later. _Done when:_ `dvc repro` reruns only what changed.

## Phase 3 — Serving

6. **FastAPI service.** Thin wrapper over `recommend_single_user`. Endpoints:
   `GET /recommend/{user_id}?k=`, `GET /health`, and (nice-to-have)
   `GET /movies/{id}/similar` using cosine similarity of learned `q_i` factors.
   Load the model once at startup; structured logging. _Done when:_ `uvicorn`
   serves and `/recommend` returns top-K JSON.
7. **Dashboard → API.** Refactor `app/dashboard.py` to call the serving API over
   HTTP instead of importing `src.recommend`. _Done when:_ the dashboard works
   against a running API and degrades gracefully if it is down.

## Phase 4 — Containerize

8. **Dockerfile.** One image usable both as the pipeline
   (`entrypoint python -m src.<stage>`) and as the API (`uvicorn`). Depends on
   Phase 0. _Done when:_ `docker build` + `docker run` serves the API.
9. **docker-compose.** Services: `api`, `dashboard`, `mlflow` (tracking UI).
   Volumes for `models/` and `outputs/`. _Done when:_ `docker compose up` brings
   up API + dashboard + MLflow.

## Phase 5 — Polish

10. **CI (GitHub Actions).** Lint + tests on PR; optional Docker build.
    _Done when:_ green check on push.
11. **Root README.** Rewrite for the rebuilt stack (Surprise SVD, DVC, FastAPI,
    Docker). Drop the old PyTorch/DVC/Docker badges. _Done when:_ the quickstart
    in the README actually works.

## Dashboard content backlog

- "More like this" — item-item similarity from `q_i` cosine (closest thing to
  explainability for a latent-factor model).
- Compare personalized recommendations vs. the popularity baseline.
- (Metrics panel + per-user rating history already exist.)

## Future features (deferred)
- **Item-item similarity endpoint** (`GET /movies/{id}/similar`) — "more like this"
  via cosine of the learned `q_i` factors. (Was "D5"; deferred to a later pass.)
- **Collapse series by season** — recommend "The Sopranos" once, not "Season 1 / 2 /
  3" as separate items. Needs grouping seasons of the same show (title parsing or a
  mapping) before ranking.

## Open decisions

- **`relevance_threshold = 4.5`** with integer ratings means only 5★ count as
  relevant. Reconsider `4.0` if the intent is "4 or 5". Affects evaluation
  numbers, so decide before re-running benchmarks.
- **DVC remote:** local dir vs. cloud (decide in Phase 2).
- **Deps installer:** pip vs. uv (cosmetic; format stays `requirements.txt`).
- **Docker:** single image vs. separate `api` / `pipeline` images.
