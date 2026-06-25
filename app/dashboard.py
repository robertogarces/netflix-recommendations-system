"""Streamlit dashboard for the SVD recommender.

Single page with two sections:
- Model Performance: test metrics from the latest evaluate run (outputs/metrics.json).
- User Explorer:     type a customer_id -> live top-K recommendations, warm/cold
                     status, and the user's own rating history for context.

Recommendations come from the serving API over HTTP (app/api.py), so the dashboard
is a thin presentation client and the model is owned by a single service. Point it
elsewhere with API_URL=http://host:port. Metrics and rating history are still read
from disk (no API endpoint for those yet).

Run:  make serve        # the API, in one terminal
      make dashboard    # this dashboard, in another
"""

import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import polars as pl
import streamlit as st
import yaml

# Streamlit runs this file directly, so put the project root on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.paths import CONFIG_PATH, OUTPUTS_PATH, PROCESSED_DATA_PATH
from src.recommend import load_movie_titles

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAMPLE_USER = 1488844  # a warm user, for a meaningful default view

st.set_page_config(page_title="Netflix SVD Recommender", page_icon="\U0001f3ac", layout="wide")


@st.cache_data
def load_config_cached():
    with open(CONFIG_PATH / "settings.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_metrics():
    path = OUTPUTS_PATH / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def get_user_history(user_id: int) -> pd.DataFrame:
    hist = (
        pl.scan_parquet(PROCESSED_DATA_PATH / "processed_data.parquet")
        .filter(pl.col("customer_id") == user_id)
        .select(["movie_id", "rating", "date"])
        .collect()
        .to_pandas()
    )
    if not hist.empty:
        titles = load_movie_titles()
        hist["title"] = hist["movie_id"].map(titles)
        hist = hist.sort_values("rating", ascending=False)
    return hist


def fetch_health() -> dict:
    return httpx.get(f"{API_URL}/health", timeout=10).json()


def fetch_recommendations(user_id: int, k: int) -> dict:
    resp = httpx.get(f"{API_URL}/recommend/{user_id}", params={"k": k}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# --- Shared resources ---
config = load_config_cached()
k_default = config["recommend"]["top_k"]

try:
    health = fetch_health()
except httpx.HTTPError:
    st.error(f"Recommendation API not reachable at {API_URL}. Start it with `make serve`.")
    st.stop()

st.title("\U0001f3ac Netflix Recommender — SVD Matrix Factorization (Surprise)")
st.caption(f"Model knows {health['n_users']:,} users and {health['n_items']:,} movies")

# --- Model Performance ---
st.subheader("\U0001f4ca Model Performance")
metrics = load_metrics()
if metrics is None:
    st.warning("No metrics yet. Run `python -m src.evaluate` to generate outputs/metrics.json.")
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSE Overall", f"{metrics['test_rmse']:.4f}")
    c2.metric("RMSE Warm", f"{metrics['test_rmse_warm']:.4f}")
    c3.metric("RMSE Cold", f"{metrics['test_rmse_cold']:.4f}")
    c4.metric("Precision@K", f"{metrics['precision_at_k']:.4f}", help=f"warm: {metrics['precision_at_k_warm']:.4f}")
    c5.metric("Recall@K", f"{metrics['recall_at_k']:.4f}", help=f"warm: {metrics['recall_at_k_warm']:.4f}")
    st.caption(
        f"K={metrics['k']} · relevance ≥ {metrics['threshold']} · "
        f"{metrics['warm_fraction']:.0%} of {metrics['n_test_rows']:,} test rows are warm users."
    )

st.divider()

# --- User Explorer ---
st.subheader("\U0001f50d User Explorer")
col_id, col_k = st.columns([3, 1])
user_id = col_id.number_input(
    "Customer ID", min_value=1, value=SAMPLE_USER, step=1,
    help="A warm user was seen in training; an unknown user gets the cold-start fallback.",
)
k = col_k.number_input("How many", min_value=1, max_value=50, value=k_default, step=1)

try:
    rec = fetch_recommendations(int(user_id), int(k))
except httpx.HTTPError as exc:
    st.error(f"API error fetching recommendations: {exc}")
    st.stop()

segment = rec["segment"]
recs = pd.DataFrame(rec["items"])

if segment == "warm":
    st.success("**Warm user** — personalized from the user's learned factor vector.")
else:
    st.warning("**Cold user** — not in the model's vocabulary; popularity fallback (global mean + item bias).")

history = get_user_history(int(user_id))
if not history.empty:
    history["date"] = pd.to_datetime(history["date"]).dt.strftime("%d/%B/%Y")
history_caption = f"{len(history):,} ratings · showing highest-rated" if not history.empty else "no ratings found"

left, right = st.columns(2)
with left:
    st.markdown(f"**Top-{int(k)} recommendations**")
    st.caption(f"{int(k)} results from the API")
    st.dataframe(recs[["rank", "title", "score"]], hide_index=True, width="stretch")
with right:
    st.markdown("**User's rating history**")
    st.caption(history_caption)
    if history.empty:
        st.info("No rating history for this user in the processed dataset.")
    else:
        st.dataframe(history[["title", "rating", "date"]].head(15), hide_index=True, width="stretch")
