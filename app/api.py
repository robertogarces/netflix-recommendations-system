"""FastAPI serving layer for the SVD recommender.

A thin HTTP wrapper over src.recommend.recommend_single_user: the model is loaded
once at startup and kept in memory, and each request scores one user on demand.
This is the online-serving counterpart to the batch/CLI modes — same scoring core.

Run:  uvicorn app.api:app --reload   (or: make serve)
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

import yaml
from fastapi import FastAPI, Query
from pydantic import BaseModel

from config.paths import CONFIG_PATH, MODELS_PATH
from src.model import item_raw_ids, load_model
from src.recommend import load_movie_titles, recommend_single_user

with open(CONFIG_PATH / "settings.yaml") as f:
    _config = yaml.safe_load(f)
RATING_SCALE = tuple(_config["data"]["rating_scale"])
DEFAULT_K = _config["recommend"]["top_k"]


class RecItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float


class RecResponse(BaseModel):
    user_id: int
    segment: Literal["warm", "cold"]
    k: int
    items: list[RecItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_users: int
    n_items: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and its lookups once, before the API serves any request."""
    algo = load_model(MODELS_PATH / "svd_model.pkl")
    app.state.algo = algo
    app.state.titles = load_movie_titles()
    app.state.idx2raw = item_raw_ids(algo)
    yield


app = FastAPI(title="Netflix SVD Recommender", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check plus a little model metadata."""
    ts = app.state.algo.trainset
    return HealthResponse(status="ok", model_loaded=True, n_users=ts.n_users, n_items=ts.n_items)


@app.get("/recommend/{user_id}", response_model=RecResponse)
def recommend(user_id: int, k: int = Query(default=DEFAULT_K, ge=1, le=50)) -> RecResponse:
    """Top-K for one user. Unknown users are not an error — they get the cold-start
    (popularity) fallback, flagged by `segment`."""
    df, segment = recommend_single_user(
        app.state.algo, user_id, k, RATING_SCALE,
        titles=app.state.titles, idx2raw=app.state.idx2raw,
    )
    items = [
        RecItem(rank=int(r.rank), movie_id=int(r.movie_id), title=str(r.title), score=float(r.score))
        for r in df.itertuples(index=False)
    ]
    return RecResponse(user_id=user_id, segment=segment, k=k, items=items)
