# Context-Aware E-commerce Recommender (MLOps)

Hệ thống gợi ý sản phẩm theo ngữ cảnh (thiết bị, giờ, cuối tuần) — Đề tài 1, môn PTHTTM (INT14151).

## Kiến trúc
```
prepare_data.py -> shared_data/*.parquet
Prefect (flow_retrain.py, 24h) -> huấn luyện LightGBM -> MLflow Registry (@Production)
benchmark.py -> so sánh 4 mô hình (CBF, MF, LightGBM, NCF) -> MLflow
FastAPI (src/api.py) -> nạp model @Production -> /recommend, /metrics
Vue dashboard (frontend/index.html) -> gọi API
Prometheus -> scrape /metrics
```

## Chạy bằng Docker
```bash
python src/prepare_data.py                 # sinh dữ liệu (1 lần)
docker compose up -d mlflow-server
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 recsys-api python src/benchmark.py
docker compose up -d                       # api + prefect + prometheus
```
- Dashboard: http://localhost:8000
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5000 · Prometheus: http://localhost:9090

## Kết quả so sánh mô hình (CLO1)
Dán bảng in ra từ `benchmark.py` vào đây (Precision/Recall/NDCG@10, AUC, thời gian suy luận).

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`): lint, pytest, build Docker image.
