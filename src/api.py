"""FastAPI backend: phuc vu goi y theo ngu canh + dashboard + /metrics (Prometheus)."""
import os
import time
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

import recommender

DATA_DIR = os.getenv("DATA_DIR", "./shared_data")
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "./frontend")

app = FastAPI(title="Context-Aware Recommender API", version="1.0.0")

REQ_COUNT = Counter("recsys_requests_total", "So request /recommend", ["device"])
REQ_LATENCY = Histogram(
    "recsys_latency_seconds", "Do tre /recommend",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    device: str = Field("mobile", pattern="^(mobile|desktop)$")
    hour: int = Field(..., ge=0, le=23)
    top_k: int = Field(10, ge=1, le=50)


class Item(BaseModel):
    product_id: int
    title: str
    category: str
    price: float


class RecommendResponse(BaseModel):
    user_id: int
    device: str
    hour: int
    latency_ms: float
    items: List[Item]


def _products() -> pd.DataFrame:
    return pd.read_parquet(f"{DATA_DIR}/products.parquet").set_index("product_id")


@app.on_event("startup")
def _warmup() -> None:
    try:
        recommender.reload_model()
    except Exception as exc:  # API van len duoc khi chua co model Production
        print(f"[startup] Chua nap duoc model: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": recommender._service.model is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    try:
        pids = recommender.predict(req.user_id, req.device, req.hour, req.top_k)
    except Exception as exc:
        raise HTTPException(503, f"Model chua san sang: {exc}")
    dt = time.perf_counter() - t0
    REQ_COUNT.labels(req.device).inc()
    REQ_LATENCY.observe(dt)

    prods = _products()
    items = [
        Item(product_id=p, title=prods.loc[p, "title"], category=prods.loc[p, "category"],
             price=float(prods.loc[p, "price"]))
        for p in pids
    ]
    return RecommendResponse(user_id=req.user_id, device=req.device, hour=req.hour,
                             latency_ms=round(dt * 1000, 2), items=items)


@app.get("/interactions/recent")
def recent_interactions(limit: int = 20) -> list:
    df = pd.read_parquet(f"{DATA_DIR}/interactions.parquet").tail(max(1, min(limit, 200)))
    df = df.merge(_products()[["title", "category"]], left_on="product_id", right_index=True, how="left")
    df["timestamp"] = df["timestamp"].astype(str)
    return df.iloc[::-1].to_dict(orient="records")


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(f"{FRONTEND_DIR}/index.html")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
