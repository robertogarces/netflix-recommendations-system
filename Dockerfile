# Image for the Netflix SVD recommender. Used by both services in docker-compose:
# the API (uvicorn) and the Streamlit dashboard differ only by their command.
FROM python:3.10-slim

# build-essential: scikit-surprise compiles C/Cython extensions at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code only — the model, data and outputs are mounted at runtime (see compose),
# so the image stays small and code stays decoupled from the artifacts DVC manages.
COPY config/ config/
COPY src/ src/
COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
