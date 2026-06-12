"""Streamlit dashboard for the NCF recommender.

Single page with two sections:
- Model Performance: test metrics from the latest evaluate run (outputs/metrics.json).
- User Explorer:     type a customer_id -> live top-K recommendations, warm/cold
                     status, and the user's own rating history for context.

Live scoring reuses src.recommend -- the dashboard is just another wrapper over
the same core, exactly like the CLI and a future API would be.

Run:  streamlit run app/dashboard.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st
import yaml

# Streamlit runs this file directly, so put the project root on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.paths import PROCESSED_DATA_PATH, MODELS_PATH, OUTPUTS_PATH, CONFIG_PATH
from src.model import get_device, load_checkpoint
from src.recommend import recommend_single_user, load_movie_titles

st.set_page_config(page_title="Netflix NCF Recommender", page_icon="\U0001f3ac", layout="wide")


@st.cache_resource
def load_model():
    device = get_device()
    model, checkpoint = load_checkpoint(MODELS_PATH / "ncf_model.pt", device)
    sample_warm_user = min(checkpoint["user2idx"])  # guarantees the default view is personalized
    return model, checkpoint, device, sample_warm_user


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


# --- Shared resources ---
try:
    model, checkpoint, device, sample_warm_user = load_model()
except FileNotFoundError:
    st.error("No checkpoint at models/ncf_model.pt. Run `python -m src.train` first.")
    st.stop()

config       = load_config_cached()
k_default    = config["recommend"]["top_k"]
rating_scale = tuple(config["data"]["rating_scale"])
num_items    = len(checkpoint["item2idx"])

st.title("\U0001f3ac Netflix Recommender — Neural Collaborative Filtering")
st.caption(f"Model knows {len(checkpoint['user2idx']):,} users and {num_items:,} movies · device: {device.type}")

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
    "Customer ID", min_value=1, value=int(sample_warm_user), step=1,
    help="A warm user was seen in training; an unknown user gets the cold-start fallback.",
)
k = col_k.number_input("How many", min_value=1, max_value=50, value=k_default, step=1)

recs, segment = recommend_single_user(
    model, checkpoint, int(user_id), int(k), num_items, device, rating_scale
)

if segment == "warm":
    st.success("**Warm user** — personalized from the user's learned embedding.")
else:
    st.warning("**Cold user** — not in the model's vocabulary; popularity fallback (global mean + item bias).")

history = get_user_history(int(user_id))
if not history.empty:
    history["date"] = pd.to_datetime(history["date"]).dt.strftime("%d/%B/%Y")
history_caption = f"{len(history):,} ratings · showing highest-rated" if not history.empty else "no ratings found"

left, right = st.columns(2)
with left:
    st.markdown(f"**Top-{int(k)} recommendations**")
    st.caption(f"{int(k)} personalized results")
    st.dataframe(recs[["rank", "title", "score"]], hide_index=True, width="stretch")
with right:
    st.markdown("**User's rating history**")
    st.caption(history_caption)
    if history.empty:
        st.info("No rating history for this user in the processed dataset.")
    else:
        st.dataframe(history[["title", "rating", "date"]].head(15), hide_index=True, width="stretch")
