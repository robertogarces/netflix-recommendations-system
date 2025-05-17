# Usa una imagen base compatible con ARM64
FROM python:3.10-slim

# Variables de entorno para no tener que interactuar en instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
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

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias con pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar DVC (agrega extras como [gs], [s3], etc. si los necesitas)
RUN pip install --no-cache-dir dvc

# Copiar el resto del código
COPY . .

# Comando por defecto
CMD ["dvc", "repro"]
