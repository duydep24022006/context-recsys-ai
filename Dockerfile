FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by LightGBM & XGBoost on Linux containers
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/

RUN mkdir -p /app/shared_data /app/mlflow_store /app/docs/results /app/docs/figures

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DATA_DIR=/app/shared_data
ENV FRONTEND_DIR=/app/frontend
