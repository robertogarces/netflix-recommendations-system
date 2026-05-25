![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-pipeline-945DD6?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-container-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


# 🎬 Netflix Movie Recommendation System — End-to-End ML Pipeline

This repository contains a modular and reproducible machine learning pipeline built with [DVC](https://dvc.org/) and Docker. The goal is to streamline the end-to-end process of data preprocessing, model training, and inference in a clean and organized way.

📖 For full technical documentation, including model design, pipeline stages, evaluation metrics, and artifacts, see [docs/netflix-model-docs.md](docs/netflix-model-docs.md).

---

## 📦 Project Structure

```
.
├── artifacts/            # Output directory for processed data artifacts (e.g., valid users/movies lists)
│   ├── valid_movies.pkl  
│   └── valid_users.pkl   
├── config/               # Configuration files and settings for the project
│   ├── best_params.yaml  
│   ├── paths.py          
│   └── settings.yaml     
├── data/                 # Raw, processed, and final datasets
│   ├── final/            
│   ├── processed/        
│   └── raw/              
├── mlruns/               # MLflow tracking directory for experiment logs and metrics
├── models/               # Saved trained models
│   └── svd_model.pkl     
├── notebooks/            # Jupyter notebooks for EDA, modeling, prediction, and preprocessing
│   ├── eda.ipynb
│   ├── model.ipynb
│   ├── predict.ipynb
│   └── preprocessing.ipynb
├── src/                  # Source code: main scripts for preprocessing, training, and prediction
│   ├── ncf_model.py              
│   ├── predictions.py
│   ├── preprocessing.py
│   ├── svd_model.py
│   └── training.py
├── utils/                # Utility modules for config, data handling, splitting, file ops, and metrics
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_processing.py
│   ├── data_split.py
│   ├── files_management.py
│   ├── metrics.py
│   └── pytorch_utils.py
├── dvc.yaml              # DVC pipeline definition for reproducible data versioning and pipeline stages
├── Dockerfile            # Dockerfile to build containerized environment with dependencies and DVC setup
├── requirements.txt      # Python pip dependencies
└── environment.yaml      # Conda environment file (used for local development and Docker image)

```



---

## ⚙️ Pipeline Stages

The DVC pipeline is defined in `dvc.yaml` and includes the following stages:

1. **Preprocessing** – Cleans and transforms the input data.
2. **Training** – Trains a recommendation system model.
3. **Prediction** – Uses the trained model to generate predictions.

---

## 📊 Results

Both models were trained on a 5% sample of the Netflix Prize dataset and evaluated
on a temporal hold-out set (most recent 20% of interactions).

| Metric | NCF | SVD |
|--------|-----|-----|
| RMSE | 0.9877 | 1.0264 |
| Precision@10 | 0.1197 | 0.1460 |
| Recall@10 | 0.9998 | 0.9956 |

> **Threshold:** a movie is considered relevant if its true rating is ≥ 4.0.
> Experiments tracked with [MLflow](https://mlflow.org/).

### Interpretation

**RMSE** is acceptable for both models — less than 1 star of error on a 1–5 scale.
NCF edges out SVD slightly (0.99 vs 1.03), suggesting its neural architecture captures
user-movie interactions more precisely.

**Recall@10 is near-perfect (~1.0) for both models**, meaning almost every movie a user
would actually enjoy appears somewhere in the top 10. However, this comes at a cost:

**Precision@10 is low (0.12–0.15)**, meaning only 1 or 2 out of every 10 recommended
movies are actually relevant to the user. The models are being too generous — they
predict high ratings broadly rather than selectively.

This Recall/Precision imbalance is a known symptom of training on a **small data sample**.
With only 5% of the dataset, the models see too few interactions per user to learn
selective preferences, and tend to recommend popular or broadly-liked movies to everyone.

### Potential Improvements

- **Train on the full dataset** — the most impactful change. Using 100% of the data
  instead of 5% would give the models enough signal to learn user-specific preferences
  and improve Precision significantly.
- **Tune the recommendation threshold** — currently set at 4.0. Raising it to 4.5
  would make the definition of "relevant" stricter, likely improving Precision at the
  cost of some Recall.
- **Hyperparameter optimization for NCF** — SVD uses Optuna for tuning, but NCF
  hyperparameters are currently fixed. Applying the same optimization strategy to NCF
  embedding size, learning rate, and regularization could improve its performance.
- **Increase NCF model capacity** — the current MLP architecture (64→32→1) is relatively
  shallow. Deeper layers or larger embeddings could capture more complex patterns.


## 📋 Requirements
/
* Docker
* Git
* A Kaggle account to download the dataset
No need to install Python or DVC locally — everything runs inside Docker

---

## 📥 Data Setup
This project uses the Netflix Prize dataset from Kaggle. The raw data files are not included in this repository and must be downloaded manually.
Steps:

Go to the dataset page on Kaggle and download the files.
Place the following files inside data/raw/:

data/raw/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
├── movie_titles.csv
└── qualifying.txt

The pipeline is configured to use only combined_data_1.txt by default (num_raw_files: 1 in settings.yaml). Set it to 4 to use the full dataset.

---

## 🚀 Getting Started

### 1. Build the Docker Image

```bash
docker build -t netflix-pipeline .
```

> This creates a reproducible environment with Python, DVC, and all necessary dependencies.

### 2. Run the Pipeline

Run the full pipeline from the root of the project directory:

```bash
docker run --rm -v $(pwd):/app netflix-pipeline
```

This command executes `dvc repro` inside the container, automatically running the pipeline stages as needed.

> **Note:** The project must be a Git repository for DVC to work correctly.

---

## 🧪 Run Individual Scripts (Optional)

You can also manually run individual pipeline stages inside the container:

```bash
# Preprocessing
docker run --rm -v $(pwd):/app netflix-pipeline python -m src.preprocessing

# Training
docker run --rm -v $(pwd):/app netflix-pipeline python -m src.training

# Prediction
docker run --rm -v $(pwd):/app netflix-pipeline python -m src.predictions
```

---

## 📁 Volumes & Data Persistence

To preserve data and artifacts between runs, mount the entire project directory using:

```bash
-v $(pwd):/app
```

Alternatively, you can mount folders individually (less recommended):

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/artifacts:/app/artifacts \
  netflix-pipeline
```

---

## 📋 Requirements

- [Docker](https://www.docker.com/)
- [Git](https://git-scm.com/)
- No need to install Python or DVC locally — everything runs inside Docker

---

## ✅ Tips

- Use Git to version control your code and data.
- Use `dvc repro` to re-run the pipeline after modifying code or inputs.
- All stages and file dependencies are tracked in `dvc.yaml`.

---

## 📄 License

MIT License. See `LICENSE` file for details.
