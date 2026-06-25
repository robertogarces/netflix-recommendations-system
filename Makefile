# Netflix SVD recommender — common tasks. Run `make help` to list targets.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- Environment ---
.PHONY: install install-dev
install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install  ## Install dev deps + register pre-commit hooks
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

# --- Pipeline (run in this order) ---
.PHONY: preprocess train evaluate recommend pipeline
preprocess:  ## Parse raw data -> data/processed/processed_data.parquet
	$(PYTHON) -m src.preprocessing

train:  ## Train SVD -> models/svd_model.pkl
	$(PYTHON) -m src.train

evaluate:  ## Evaluate model -> outputs/metrics.json
	$(PYTHON) -m src.evaluate

recommend:  ## Batch recommendations -> outputs/recommendations.parquet
	$(PYTHON) -m src.recommend

pipeline: preprocess train evaluate recommend  ## Run the full pipeline end to end

# --- App ---
.PHONY: serve dashboard
serve:  ## Serve the recommendation API (FastAPI + uvicorn)
	uvicorn app.api:app --reload

dashboard:  ## Launch the Streamlit dashboard
	streamlit run app/dashboard.py

# --- Docker ---
.PHONY: up down
up:  ## Start the full stack (API + dashboard + MLflow) via docker compose
	docker compose up --build

down:  ## Stop and remove the docker compose stack
	docker compose down

# --- Quality ---
.PHONY: lint format test
lint:  ## Lint with ruff (no file changes)
	ruff check .

format:  ## Apply ruff formatter + autofix (black-style; collapses aligned '=')
	ruff format .
	ruff check --fix .

test:  ## Run the test suite
	pytest

# --- Housekeeping ---
.PHONY: clean
clean:  ## Remove caches and bytecode
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache
