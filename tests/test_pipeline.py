"""Unit tests for data preparation (contract & leak-free split), metrics, models, and drift."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drift import compute_psi, detect_data_drift  # noqa: E402
from metrics import evaluate_model, map_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from models import (  # noqa: E402
    train_content_based,
    train_lightgbm,
    train_matrix_factorization,
    train_ncf,
    train_popularity_baseline,
)
from prepare_data import build_features_and_split, generate_synthetic_raw  # noqa: E402


def test_ranking_metrics_math():
    ranked = [10, 20, 30, 40, 50]
    gt = {10, 30}
    assert abs(precision_at_k(ranked, gt, k=5) - 0.4) < 1e-6
    assert abs(recall_at_k(ranked, gt, k=5) - 1.0) < 1e-6
    assert 0.8 < ndcg_at_k(ranked, gt, k=5) <= 1.0
    # Hits at rank 1 (prec=1/1) and rank 3 (prec=2/3) -> AP = (1 + 2/3)/2 = 5/6
    assert abs(map_at_k(ranked, gt, k=5) - (5.0 / 6.0)) < 1e-6


def test_prepare_data_contract_and_no_leakage(tmp_path):
    data_dir = str(tmp_path / "data")
    prods, inter = generate_synthetic_raw(n_users=40, n_products=30, n_interactions=500, seed=42)

    # Inject a test-only user at the very end of the timeline to verify zero leakage into train features
    max_ts = pd.to_datetime(inter["timestamp"]).max()
    future_row = pd.DataFrame([{
        "user_id": 999,
        "product_id": 1,
        "event_type": "purchase",
        "device_type": "mobile",
        "timestamp": max_ts + pd.Timedelta(days=2),
    }])
    inter_all = pd.concat([inter, future_row], ignore_index=True)

    build_features_and_split(inter_all, prods, data_dir=data_dir, seed=42)

    saved_inter = pd.read_parquet(f"{data_dir}/interactions.parquet")
    assert list(saved_inter.columns) == ["user_id", "product_id", "event_type", "device_type", "timestamp"]

    feat_uc = pd.read_parquet(f"{data_dir}/feat_user_cat.parquet")
    # User 999 only exists in the future 20% test split -> must NOT appear in feat_user_cat computed on train!
    assert 999 not in set(feat_uc["user_id"].tolist()), "Data leakage detected: future test user leaked into feat_user_cat"


def test_models_and_evaluate(tmp_path):
    data_dir = str(tmp_path / "data")
    prods, inter = generate_synthetic_raw(n_users=30, n_products=25, n_interactions=400, seed=42)
    build_features_and_split(inter, prods, data_dir=data_dir, seed=42)

    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    feat_uc = pd.read_parquet(f"{data_dir}/feat_user_cat.parquet")
    feat_ci = pd.read_parquet(f"{data_dir}/feat_ctx_item.parquet")

    pop_m = train_popularity_baseline(train_df, prods)
    cb_m = train_content_based(train_df, prods, max_features=16)
    mf_m = train_matrix_factorization(train_df, prods, n_components=8)
    lgb_m = train_lightgbm(train_df, n_estimators=20)
    ncf_m = train_ncf(train_df, prods, emb_dim=8, epochs=2)

    for m in (pop_m, cb_m, mf_m, lgb_m, ncf_m):
        preds = m.predict(None, test_df.head(10))
        assert len(preds) == 10

    metrics = evaluate_model(
        predict_fn=lambda df_in: lgb_m.predict(None, df_in),
        test_df=test_df,
        products_df=prods,
        feat_user_cat=feat_uc,
        feat_ctx_item=feat_ci,
        k=10,
        max_queries=15,
    )
    assert "ndcg_at_10" in metrics and "map_at_10" in metrics and "auc" in metrics

    # Drift check
    rep_file = str(tmp_path / "drift.html")
    drift_res = detect_data_drift(data_dir=data_dir, report_path=rep_file)
    assert os.path.exists(rep_file)
    assert "overall_drift" in drift_res
    assert compute_psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) < 1e-5
