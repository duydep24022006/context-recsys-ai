"""Comprehensive unit & integration tests for FastAPI backend (Phase 4)."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_data"))
os.environ["DATA_DIR"] = DATA_DIR
os.environ["FRONTEND_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

from prepare_data import build_features_and_split, generate_synthetic_raw  # noqa: E402
import api  # noqa: E402
import recommender  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_shared_data():
    """Ensure shared_data/*.parquet exists before running API tests (even in clean CI)."""
    if not os.path.exists(os.path.join(DATA_DIR, "products.parquet")):
        prods, inter = generate_synthetic_raw(n_users=60, n_products=40, n_interactions=800, seed=42)
        build_features_and_split(inter, prods, data_dir=DATA_DIR, seed=42)
    recommender._service.refresh_feature_cache()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(recommender, "reload_model", lambda: {"version": "1", "name": "ContextAwareRecommender"})
    monkeypatch.setattr(recommender, "predict", lambda u, d, h, k: list(range(1, k + 1)))
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_recommend_ok(client):
    r = client.post("/recommend", json={"user_id": 1, "device": "mobile", "hour": 20, "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 5
    assert "score" in data["items"][0]


def test_recommend_validation(client):
    assert client.post("/recommend", json={"user_id": 1, "device": "tv", "hour": 20}).status_code == 422
    assert client.post("/recommend", json={"user_id": 1, "device": "mobile", "hour": 99}).status_code == 422
    assert client.post("/recommend", json={"user_id": 0, "device": "mobile", "hour": 12}).status_code == 422


def test_recommend_cold_start_real_service():
    """Verify cold-start user (user_id=99999) returns context-popular items without 500 error."""
    raw_client = TestClient(api.app)
    r = raw_client.post("/recommend", json={"user_id": 99999, "device": "mobile", "hour": 21, "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 5
    assert data["cold_start"] is True


def test_admin_reload_and_model_info(client):
    r_reload = client.post("/admin/reload")
    assert r_reload.status_code == 200
    assert r_reload.json()["status"] in ("reloaded", "cache_refreshed_only")

    r_info = client.get("/model/info")
    assert r_info.status_code == 200
    assert r_info.json()["name"] == "ContextAwareRecommender"


def test_products_and_user_profile(client):
    r_prod = client.get("/products?limit=10")
    assert r_prod.status_code == 200
    assert len(r_prod.json()) == 10

    r_prof = client.get("/users/1/profile")
    assert r_prof.status_code == 200
    assert r_prof.json()["user_id"] == 1

    r_cold = client.get("/users/88888/profile")
    assert r_cold.status_code == 200
    assert r_cold.json()["cold_start"] is True


def test_recent_interactions_and_analytics(client):
    r_rec = client.get("/interactions/recent?limit=5&device_type=mobile")
    assert r_rec.status_code == 200
    assert len(r_rec.json()) <= 5

    r_ana = client.get("/analytics/summary")
    assert r_ana.status_code == 200
    assert r_ana.json()["total_interactions"] > 0


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"recsys_requests_total" in r.content
