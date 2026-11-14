import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["DATA_DIR"] = os.path.join(os.path.dirname(__file__), "..", "shared_data")
os.environ["FRONTEND_DIR"] = os.path.join(os.path.dirname(__file__), "..", "frontend")

import api  # noqa: E402
import recommender  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(recommender, "reload_model", lambda: None)
    monkeypatch.setattr(recommender, "predict", lambda u, d, h, k: list(range(1, k + 1)))
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").status_code == 200


def test_recommend_ok(client):
    r = client.post("/recommend", json={"user_id": 1, "device": "mobile", "hour": 20, "top_k": 5})
    assert r.status_code == 200
    assert len(r.json()["items"]) == 5


def test_recommend_validation(client):
    assert client.post("/recommend", json={"user_id": 1, "device": "tv", "hour": 20}).status_code == 422
    assert client.post("/recommend", json={"user_id": 1, "device": "mobile", "hour": 99}).status_code == 422


def test_metrics_endpoint(client):
    assert b"recsys_requests_total" in client.get("/metrics").content or client.get("/metrics").status_code == 200
