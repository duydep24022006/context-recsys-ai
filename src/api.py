"""FastAPI backend: Context-Aware Recommender API + Prometheus metrics + Admin & Analytics."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

import recommender

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("recsys.api")

DATA_DIR = os.getenv("DATA_DIR", "./shared_data")
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "./frontend")

REQ_COUNT = Counter(
    "recsys_requests_total",
    "Total number of /recommend requests",
    ["device"],
)
REQ_LATENCY = Histogram(
    "recsys_latency_seconds",
    "Latency of /recommend endpoint in seconds",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
ERR_COUNT = Counter(
    "recsys_errors_total",
    "Total number of API errors",
    ["endpoint"],
)
MODEL_VERSION_GAUGE = Gauge(
    "recsys_model_version",
    "Currently loaded Production model version number",
)
REC_SCORE_HIST = Histogram(
    "recsys_recommendation_score",
    "Distribution of predicted recommendation scores",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0),
)
REC_SCORE_GAUGE = Gauge(
    "recsys_top1_score",
    "Most recent Top-1 recommendation score",
)


def _sync_version_gauge() -> None:
    ver = recommender._service.model_meta.get("version")
    if ver is not None and str(ver).isdigit():
        MODEL_VERSION_GAUGE.set(float(ver))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        recommender._service.refresh_feature_cache()
        recommender.reload_model()
        _sync_version_gauge()
    except Exception as exc:
        logger.warning("[startup] Model not yet loaded from Registry: %s", exc)
    yield


app = FastAPI(
    title="Context-Aware Recommender API",
    description="Real-time E-commerce Product Recommendation Service with MLOps & Context Awareness",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    device: str = Field("mobile", pattern="^(mobile|desktop)$")
    hour: int = Field(..., ge=0, le=23)
    top_k: int = Field(10, ge=1, le=50)
    is_weekend: Optional[int] = Field(None, ge=0, le=1)


class Item(BaseModel):
    product_id: int
    title: str
    category: str
    price: float
    score: float = 0.0


class RecommendResponse(BaseModel):
    user_id: int
    device: str
    hour: int
    latency_ms: float
    cold_start: bool = False
    model_version: Optional[str] = None
    items: List[Item]


def _products() -> pd.DataFrame:
    if recommender._service.products_df is not None and not recommender._service.products_df.empty:
        return recommender._service.products_df
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    prods = pd.read_parquet(f"{data_dir}/products.parquet").set_index("product_id", drop=False)
    recommender._service.products_df = prods
    return prods


@app.get("/health")
def health() -> dict:
    _sync_version_gauge()
    return {
        "status": "ok",
        "model_loaded": recommender._service.model is not None,
        "model_version": recommender._service.model_meta.get("version"),
        "model_name": recommender._service.model_meta.get("model_name_tag"),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    try:
        pids = recommender.predict(req.user_id, req.device, req.hour, req.top_k)
    except Exception as exc:
        ERR_COUNT.labels("recommend").inc()
        raise HTTPException(503, f"Model chua san sang: {exc}") from exc
    dt = time.perf_counter() - t0
    REQ_COUNT.labels(req.device).inc()
    REQ_LATENCY.observe(dt)
    _sync_version_gauge()

    prods = _products()
    scores_map = getattr(recommender._service, "last_scores", {})
    is_cold = bool(getattr(recommender._service, "last_cold_start", False))

    items: List[Item] = []
    for idx, p in enumerate(pids):
        sc = float(scores_map.get(p, round(max(0.05, 0.95 - idx * 0.04), 4)))
        REC_SCORE_HIST.observe(max(0.0, min(1.0, sc)))
        if idx == 0:
            REC_SCORE_GAUGE.set(sc)
        row = prods.loc[p]
        items.append(
            Item(
                product_id=int(p),
                title=str(row["title"]),
                category=str(row["category"]),
                price=float(row["price"]),
                score=sc,
            )
        )

    return RecommendResponse(
        user_id=req.user_id,
        device=req.device,
        hour=req.hour,
        latency_ms=round(dt * 1000.0, 2),
        cold_start=is_cold,
        model_version=recommender._service.model_meta.get("version"),
        items=items,
    )


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    info = dict(recommender.get_model_info())
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    retrain_file = Path(data_dir) / "last_retrain.json"
    if retrain_file.exists():
        try:
            info["last_retrain"] = json.loads(retrain_file.read_text(encoding="utf-8"))
        except Exception:
            info["last_retrain"] = None
    else:
        info["last_retrain"] = None
    info["model_loaded"] = recommender._service.model is not None
    return info


@app.get("/models/benchmark")
def models_benchmark() -> List[Dict[str, Any]]:
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    json_path = Path(data_dir) / "benchmark_results.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    csv_path = Path("docs/results/benchmark_results.csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path).to_dict(orient="records")
        except Exception:
            pass
    return []


@app.post("/admin/reload")
def admin_reload() -> Dict[str, Any]:
    try:
        meta = recommender.reload_model()
        _sync_version_gauge()
        return {"status": "reloaded", "model": meta}
    except Exception as exc:
        ERR_COUNT.labels("admin_reload").inc()
        recommender._service.refresh_feature_cache()
        return {
            "status": "cache_refreshed_only",
            "detail": str(exc),
            "model": recommender._service.model_meta,
        }


@app.get("/products")
def list_products(
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict[str, Any]]:
    prods = _products().copy()
    if category:
        prods = prods[prods["category"].str.lower() == category.lower()]
    return prods.head(limit)[["product_id", "title", "category", "price"]].to_dict(orient="records")


@app.get("/users/{user_id}/profile")
def user_profile(user_id: int) -> Dict[str, Any]:
    if user_id < 1:
        raise HTTPException(422, "user_id must be >= 1")
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    uc_path = Path(data_dir) / "feat_user_cat.parquet"
    inter_path = Path(data_dir) / "interactions.parquet"

    fav_cats: List[Dict[str, Any]] = []
    if uc_path.exists():
        uc_df = pd.read_parquet(uc_path)
        u_rows = uc_df[uc_df["user_id"] == user_id].sort_values("user_cat_ratio", ascending=False)
        fav_cats = [
            {"category": str(r.category), "ratio": round(float(r.user_cat_ratio), 4)}
            for r in u_rows.itertuples(index=False)
        ]

    total_events = 0
    device_breakdown: Dict[str, int] = {}
    if inter_path.exists():
        inter_df = pd.read_parquet(inter_path)
        u_inter = inter_df[inter_df["user_id"] == user_id]
        total_events = int(len(u_inter))
        device_breakdown = {str(k): int(v) for k, v in u_inter["device_type"].value_counts().items()}

    return {
        "user_id": user_id,
        "cold_start": len(fav_cats) == 0,
        "total_interactions": total_events,
        "favorite_categories": fav_cats,
        "device_breakdown": device_breakdown,
    }


@app.get("/interactions/recent")
def recent_interactions(
    limit: int = Query(20, ge=1, le=200),
    user_id: Optional[int] = Query(None),
    event_type: Optional[str] = Query(None),
    device_type: Optional[str] = Query(None),
) -> list:
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    df = pd.read_parquet(f"{data_dir}/interactions.parquet")
    if user_id is not None:
        df = df[df["user_id"] == user_id]
    if event_type:
        df = df[df["event_type"] == event_type]
    if device_type:
        df = df[df["device_type"] == device_type]
    df = df.tail(max(1, min(limit, 200)))
    prods = _products()[["title", "category", "price"]]
    df = df.merge(prods, left_on="product_id", right_index=True, how="left")
    df["timestamp"] = df["timestamp"].astype(str)
    return df.iloc[::-1].to_dict(orient="records")


@app.get("/analytics/summary")
def analytics_summary() -> Dict[str, Any]:
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    inter_path = Path(data_dir) / "interactions.parquet"
    if not inter_path.exists():
        return {"total_interactions": 0}

    inter = pd.read_parquet(inter_path)
    prods = _products()[["category", "price"]]
    df = inter.merge(prods, left_on="product_id", right_index=True, how="left")
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    by_device_cat = (
        df.groupby(["device_type", "category"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    by_hour_device = (
        df.groupby(["hour", "device_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )
    by_event = {str(k): int(v) for k, v in df["event_type"].value_counts().items()}

    return {
        "total_interactions": int(len(df)),
        "unique_users": int(df["user_id"].nunique()),
        "unique_products": int(len(_products())),
        "event_counts": by_event,
        "category_by_device": by_device_cat,
        "hourly_by_device": by_hour_device,
    }


@app.get("/metrics")
def metrics() -> Response:
    _sync_version_gauge()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def index() -> FileResponse:
    frontend_dir = os.getenv("FRONTEND_DIR", FRONTEND_DIR)
    dist_index = os.path.join(frontend_dir, "dist", "index.html")
    if os.path.exists(dist_index):
        return FileResponse(dist_index)
    return FileResponse(os.path.join(frontend_dir, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
