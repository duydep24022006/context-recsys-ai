FROM python:3.11-slim

WORKDIR /app

# libgomp1 bat buoc de LightGBM chay duoc tren Linux container
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY frontend/ ./frontend/

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/shared_data
ENV FRONTEND_DIR=/app/frontend
