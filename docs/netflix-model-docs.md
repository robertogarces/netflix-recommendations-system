# Netflix Recommendation System Documentation

---

## 1. Project Overview

This project implements an end-to-end movie recommendation pipeline built on the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).

### Core Entities

| Entity | Description |
|--------|-------------|
| **Users** | Individuals who rated movies |
| **Movies** | The content being rated |
| **Ratings** | Explicit user feedback (1–5 scale) |

### Objective

Predict which movies a user is likely to enjoy based on their historical rating behavior. The business goal goes beyond prediction accuracy:

> Not only predict ratings correctly, but **rank relevant movies at the top** of each user's recommendation list.

This distinction is important and explains why multiple evaluation metrics are needed.

### Design Principles

- Modular pipeline
- Reproducibility
- Experimentation
- Separation of concerns
- Config-driven execution
- Extensible architecture

---

## 2. Business Problem

Streaming platforms contain thousands of titles. Users rarely explore the entire catalog. Recommendation systems help to:

- Increase engagement
- Improve user retention
- Reduce discovery friction
- Personalize the experience

---

## 3. Models

The system supports two recommendation approaches, both switchable through configuration without modifying pipeline logic.

### 3.1 SVD (Singular Value Decomposition)

SVD assumes that users and movies can be represented in a shared "preference space":

- Each **user** has hidden preferences (e.g., likes action, dislikes comedy)
- Each **movie** has hidden attributes (e.g., action-heavy, romantic, etc.)

The model learns these hidden patterns by decomposing the user–movie interaction matrix. A rating is predicted by measuring how well a user's preferences align with a movie's attributes.

**Advantages:** fast, interpretable, strong baseline, works well on sparse datasets  
**Limitations:** limited ability to model complex interactions

### 3.2 NCF (Neural Collaborative Filtering)

NCF follows a similar idea but uses a neural network instead of linear algebra:

1. Users and movies are converted into **embeddings** (dense vectors)
2. A neural network learns how to combine these embeddings
3. The model learns non-linear interactions between users and items

**Intuition:** instead of assuming simple relationships, the model learns complex patterns like: *"users who like X and Y together tend to dislike Z"*.

**Advantages:** captures non-linear behavior, flexible, scales to richer features  
**Limitations:** higher computational cost, requires tuning

---

## 4. Data Preparation

### 4.1 Filtering

To improve signal quality, the dataset is filtered before training:

- **Users** must have at least 10 ratings
- **Movies** must have been rated at least 50 times

This reduces noise and improves model stability and generalization.

**Configured in** `settings.yaml`:
```yaml
min_user_ratings: 10
min_movie_ratings: 50
```

**Trade-off:**

| Pros | Cons |
|------|------|
| Better learning signal | Lower coverage |
| More stable metrics | |
| Faster training | |

### 4.2 Sparse Matrix Problem

Rare movies do not have enough interactions to learn meaningful patterns. Too few interactions make learning unstable.

### 4.3 Cold Start Problem

- **Users** with almost no ratings: insufficient behavioral signal
- **Movies** with very few ratings: insufficient exposure

Preprocessing removes users and movies below the minimum thresholds to reduce this effect.

### 4.4 Temporal Split

Random splitting introduces future information leakage. Since recommendation systems operate through time:

- The model is **trained** on older interactions
- The model is **evaluated** on newer interactions

This better approximates production behavior.

---

## 5. End-to-End Pipeline

### Stage 1 — Preprocessing

**Input:** Raw Netflix text files

**Process:**
1. Load raw Netflix dataset
2. Filter users and movies based on minimum interaction thresholds
3. Encode users and movies into numerical indices
4. Split data into train / validation / test sets

**Output:** `data/processed/processed_data.parquet`

> **Why Parquet?** Faster reads, efficient compression, and better analytical performance.

---

### Stage 2 — Training

1. Train the recommendation model (SVD or NCF)
2. Optimize train and validation RMSE
3. Select the best model based on validation performance
4. Evaluate using RMSE, Precision@K, and Recall@K
5. Save artifacts

---

### Stage 3 — Prediction

**Input:** `qualifying.txt`

**Process:**
1. Load model
2. Load unseen interactions
3. Remove unsupported users/items
4. Generate predictions
5. Attach metadata
6. Export results

**Output:** `predictions.csv`

| Field | Description |
|-------|-------------|
| `user_id` | User identifier |
| `movie_id` | Movie identifier |
| `predicted_rating` | Estimated rating |
| `title` | Movie title |
| `release_year` | Release year |

**Interpretation example:**

| User | Movie | Predicted Rating |
|------|-------|-----------------|
| 1046323 | Dinosaur Planet | 3.47 |

> The model estimates this user would moderately enjoy this movie.

**How this output can be used:**
- Rank movies per user (descending by `pred_rating`)
- Generate Top-N recommendations
- Feed downstream recommendation APIs or products

---

## 6. Evaluation Metrics

Multiple metrics are used because no single metric fully captures recommendation quality.

### 6.1 RMSE (Root Mean Squared Error)

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Where: $\hat{y}_i$ = predicted rating, $y_i$ = actual rating, $n$ = number of samples.

**Measures:** on average, how far off predicted ratings are from real ratings. Penalizes large errors more than small ones.

**Business interpretation:** lower RMSE = more accurate rating estimates.

> ⚠️ **Key limitation:** low RMSE does not guarantee good recommendations. A model can predict ratings accurately but still rank items poorly.

**Example:**

| Movie | True Rating | Predicted |
|-------|-------------|-----------|
| A | 5.0 | 4.2 |
| B | 4.9 | 4.1 |
| C | 3.0 | 3.9 |

Even with an acceptable RMSE, the model might fail to put Movie A at the top or confuse the ordering between relevant items.

---

### 6.2 Precision@K

**Measures:** of the top K recommended movies, how many are actually relevant?

**Example (K=10):** model recommends 10 movies; user actually likes 4 → `Precision@10 = 0.4`

**Business interpretation:** higher Precision = more accurate top list.

---

### 6.3 Recall@K

**Measures:** of all relevant movies, how many were successfully recommended in the top K?

**Example:** user likes 20 movies; model recommends 15 of them in top K → `Recall@K = 0.75`

> **Insight:** high recall with low precision (e.g., ~0.999 recall) usually means the model is capturing almost everything relevant, but not filtering strongly enough.

**Business interpretation:** higher Recall = more of the user's interests captured.

---

### 6.4 Metrics Summary

| Metric | Measures | Better when |
|--------|----------|-------------|
| RMSE | Prediction accuracy | Lower |
| Precision@K | Recommendation quality | Higher |
| Recall@K | Recommendation coverage | Higher |

---

## 7. Hyperparameter Optimization

SVD optionally uses **Optuna** to automatically search for:

- Learning rate
- Regularization
- Latent dimensions
- Training epochs

**Goal:** improve model generalization.

---

## 8. Configuration System

Configuration exists to avoid code changes. To switch models:

```yaml
type: svd   # or 'ncf'
```

Other available settings: sampling, optimization, minimum interaction thresholds.

---

## 9. Artifacts

The following artifacts are saved during the pipeline:

| Artifact | Purpose |
|----------|---------|
| `user2idx.pkl` | Consistent user-to-index mapping |
| `movies2idx.pkl` | Consistent movie-to-index mapping |
| `valid_users.pkl` | Valid users set for evaluation |
| `valid_movies.pkl` | Valid movies set for evaluation |
| MLflow run logs | Experiment tracking |

> **Why do these matter?** They guarantee reproducibility — ensuring a model trained today behaves the same way tomorrow. Without them, consistency between training and inference cannot be guaranteed.

---

## 10. MLflow Tracking

The project uses **MLflow** to track experiments. It stores:

- Model parameters
- Training metrics
- Validation performance
- Model artifacts

**Benefits:**
- Experiment reproducibility
- Comparison between models (SVD vs NCF)
- Traceability of improvements over time
