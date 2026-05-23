# Netflix Recommendation System — Technical & Business Documentation

## 1. Project Overview

This project implements an end-to-end movie recommendation pipeline built on the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).

The objective is to predict which movies users are likely to enjoy based on historical rating behavior.

The system supports two recommendation approaches:

1. Singular Value Decomposition (SVD)
2. Neural Collaborative Filtering (NCF)

The project was intentionally designed so the recommendation engine can be changed through configuration without modifying pipeline logic.

---

## 2. Business Problem

Streaming platforms contain thousands of titles.

Users rarely explore the entire catalog.

Recommendation systems help:

- Increase engagement
- Improve user retention
- Reduce discovery friction
- Personalize experience

This project predicts expected ratings and generates recommendations.

---

## 3. Why Two Models?

Two approaches were implemented to compare a classical recommender with a deep-learning-based recommender.

### SVD (Matrix Factorization)

SVD learns latent factors.

Instead of storing:

User → Movie → Rating

it learns:

User → Preferences vector  
Movie → Characteristics vector

The interaction estimates expected rating.

Advantages:
- Fast
- Interpretable
- Strong baseline
- Works well on sparse datasets

Limitations:
- Limited ability to model complex interactions

---

### Neural Collaborative Filtering (NCF)

NCF replaces matrix factorization interactions with neural networks.

Architecture:

User Embedding
↓
Movie Embedding
↓
MLP Layers
↓
Predicted Rating

Advantages:
- Captures nonlinear behavior
- Flexible
- Scales to richer features

Limitations:
- Higher computational cost
- Requires tuning

---

## 4. Data Preparation Philosophy

Raw recommendation datasets are extremely sparse.

Most users rate very few movies.

Most movies receive very few ratings.

Example:

1,000,000 users × 20,000 movies

Only a tiny fraction contains observations.

This sparsity creates:

### Sparse Matrix Problem

Too few interactions make learning unstable.

### Cold Start Problem

Users with almost no ratings:
- insufficient behavioral signal

Movies with very few ratings:
- insufficient exposure

To reduce this effect, preprocessing removes:

- Users below minimum rating threshold
- Movies below minimum rating threshold

Configured in:

settings.yaml

Example:

min_user_ratings: 10  
min_movie_ratings: 50

Tradeoff:

Pros:
- Better learning signal
- More stable metrics
- Faster training

Cons:
- Lower coverage

---

## 5. End-to-End Pipeline

### Stage 1 — Preprocessing

Input:
Raw Netflix text files

Process:

1. Load raw files
2. Parse Netflix format
3. Standardize columns
4. Convert IDs to categorical strings
5. Remove invalid ratings
6. Filter sparse users and movies
7. Save parquet dataset

Output:

data/processed/processed_data.parquet

Why parquet?

- Faster reads
- Compression
- Better analytical performance

---

### Stage 2 — Training

Controlled from:

config/settings.yaml

Training performs:

Load processed dataset
↓
Temporal train/test split
↓
Optional hyperparameter optimization
↓
Train selected model
↓
Evaluate
↓
Save artifacts

---

## 6. Why Temporal Split?

Random split introduces future information leakage.

Recommendation systems operate through time.

Temporal split trains on older interactions and evaluates on newer interactions.

This better approximates production.

---

## 7. Hyperparameter Optimization

SVD optionally uses Optuna.

Purpose:

Automatically search:

- learning rate
- regularization
- latent dimensions
- epochs

Goal:

Improve generalization.

---

## 8. Evaluation Metrics

Multiple metrics are used because no single metric fully captures recommendation quality.

### RMSE

Root Mean Squared Error.

Measures:

How close predicted ratings are to actual ratings.

Lower is better.

Business interpretation:

Lower RMSE means rating estimates are more accurate.

Limitations:

Good RMSE does not guarantee useful recommendations.

---

### Precision@K

Measures:

Among recommended movies,
how many were actually relevant.

Higher is better.

Business interpretation:

“How often are recommendations useful?”

---

### Recall@K

Measures:

Among relevant movies,
how many the system successfully surfaced.

Higher is better.

Business interpretation:

“How much of user interest did we capture?”

---

Why combine them?

RMSE:
Prediction accuracy

Precision:
Recommendation quality

Recall:
Recommendation coverage

Together they provide balanced evaluation.

---

## 9. Prediction Stage

Input:

qualifying.txt

Process:

Load model
↓
Load unseen interactions
↓
Remove unsupported users/items
↓
Generate predictions
↓
Attach metadata
↓
Export results

Output:

predictions.csv

Fields:

- user_id
- movie_id
- predicted_rating
- title
- release_year

---

## 10. Artifacts

Generated files:

processed_data.parquet  
valid_users.pkl  
valid_movies.pkl  
user2idx.pkl  
item2idx.pkl  
model files  
predictions.csv

Purpose:

Enable reproducibility.

---

## 11. Configuration System

Configuration exists to avoid code changes.

Switch model:

type: svd

or

type: ncf

Other settings:

sampling  
optimization  
thresholds

---

## 12. How to Interpret Results

Example:

Movie A → 4.7  
Movie B → 3.1

Interpretation:

Movie A is expected to be more relevant.

These are ranking signals.

Do not interpret ratings as guaranteed satisfaction.

---

## 13. Intended Design Principles

- Modular pipeline
- Reproducibility
- Experimentation
- Separation of concerns
- Config-driven execution
- Extensible architecture

---

## 14. Future Improvements

Possible improvements:

- Feature store
- Online inference
- MLflow
- Model registry
- Bias mitigation
- Explainability
