# 🎬 Netflix ML Pipeline

This repository contains a modular and reproducible machine learning pipeline built with [DVC](https://dvc.org/) and Docker. The goal is to streamline the end-to-end process of data preprocessing, model training, and inference in a clean and organized way.

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
│   ├── preprocessing.py
│   ├── training.py
│   └── predictions.py
├── utils/                # Utility modules for config, data handling, splitting, file ops, and metrics
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_processing.py
│   ├── data_split.py
│   ├── files_management.py
│   └── metrics.py
├── dvc.yaml              # DVC pipeline definition for reproducible data versioning and pipeline stages
├── Dockerfile            # Dockerfile to build containerized environment with dependencies and DVC setup
├── requirements.txt      # Python pip dependencies
└── environment.yaml      # Conda environment file (used for local development and Docker image)

```



---

## ⚙️ Pipeline Stages

The DVC pipeline is defined in `dvc.yaml` and includes the following stages:

1. **Preprocessing** – Cleans and transforms the input data.
2. **Training** – Trains a classification model.
3. **Prediction** – Uses the trained model to generate predictions.

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
