# Use an ARM64-compatible base image
FROM python:3.10-slim

# Environment variables to avoid interactive package installation prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install DVC (add extras like [gs], [s3], etc. if needed)
RUN pip install --no-cache-dir dvc

# Copy the rest of the code
COPY . .

# Default command
CMD ["dvc", "repro"]
