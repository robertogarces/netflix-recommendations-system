# Serving (API)

An HTTP service that loads the trained SVD model once and returns recommendations
on demand. It is a thin wrapper over `src.recommend.recommend_single_user` — the
same scoring core as the batch and CLI modes ([recommendation.md](recommendation.md)),
exposed over HTTP for online use.

---

## Running it
```
make serve            # uvicorn app.api:app --reload
```
- Interactive docs (Swagger): http://localhost:8000/docs
- Runnable client demo: `python examples/demo.py` (with the API running)

---

## Endpoints
| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | service status + model metadata (`n_users`, `n_items`) |
| `GET` | `/recommend/{user_id}?k=10` | top-K for one user, with a `segment` of `warm` or `cold` |
| — | `/docs`, `/openapi.json` | auto-generated Swagger UI / OpenAPI schema |

```bash
curl localhost:8000/health
curl "localhost:8000/recommend/1488844?k=5"     # warm: personalized
curl "localhost:8000/recommend/999999999?k=5"   # unknown -> cold (popularity)
```
```json
{
  "user_id": 1488844, "segment": "warm", "k": 5,
  "items": [{"rank": 1, "movie_id": 3928, "title": "Nip/Tuck: Season 2", "score": 4.48}, "..."]
}
```

---

## Clients
Two things consume the API:
- **`examples/demo.py`** — a zero-dependency example client that prints a warm vs
  cold user side by side.
- **The Streamlit dashboard** (`app/dashboard.py`, `make dashboard`) — sends the
  `customer_id` you type to `/recommend` and renders the response. The model is owned
  only by the API; the dashboard is a thin presentation client. (Its Model Performance
  and rating-history panels still read from disk — there is no API endpoint for those
  yet, so the decoupling is partial.)

---

## Why an API for a fixed dataset? (honest framing)
For a *static* catalog with a closed user set, **batch precompute is the correct
engineering choice** — and the pipeline already produces it
(`outputs/recommendations.parquet`). Online serving exists to handle what this
project does not actually have: too many users to precompute, fresh/changing data,
real-time requests. So this API exists to **demonstrate the online-serving
pattern**, not because batch is insufficient here — and that caveat is stated
plainly rather than hidden.

The realistic scenario it stands in for: a "Top Picks for You" row that calls
`/recommend/{user_id}` when a user opens the app — personalized for known users, an
instant popularity fallback for brand-new ones, recomputed per visit so it only
scores the users who actually show up. *That* is where online serving beats batch;
this service is the shape of it.

---

## Design notes
- **Model loaded once, at startup.** The ~235 MB model is loaded in a FastAPI
  `lifespan` handler and kept in `app.state`; loading it per request would be fatal.
- **Unknown user is not a 404.** The model degrades to a popularity ranking for
  users it never saw, so an unknown id returns `200` with `segment: "cold"`.
- **Sync endpoints.** Handlers are plain `def` (not `async`): they do blocking work
  (a Polars scan for already-seen items, NumPy scoring), so FastAPI runs them in a
  threadpool instead of blocking the event loop.
- **Seen-item filtering is per-request.** `recommend_single_user` scans the
  processed parquet to drop items the user already rated — fine for a demo, but a
  real service would serve these from a precomputed store, not scan on every call.
