"""Prefect MLOps Retraining Pipeline for Context-Aware Recommender (Phase 3).

Pipeline stages:
1. ingest_and_validate_data: Verify 5-column data contract & value domains
2. feature_engineering_task: Recompute leak-free temporal split & features
3. check_drift_task: Compute PSI/KS data drift report (docs/results/drift_report.html)
4. train_and_evaluate_task: Train LightGBM context-aware model & log to MLflow with code_paths
5. compare_and_promote_task: Promote to @Production if NDCG@10 >= 95% of current Production
   and notify FastAPI `/admin/reload`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
from prefect import flow, get_run_logger, task

from drift import detect_data_drift
from metrics import evaluate_model
from models import MODELS_CODE_PATH, train_lightgbm
from prepare_data import build_features_and_split

DATA_DIR = os.getenv("DATA_DIR", "./shared_data")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
API_BASE_URL = os.getenv("API_BASE_URL", "http://recsys-api:8000")
EXPERIMENT_NAME = os.getenv("RETRAIN_EXPERIMENT_NAME", "Context_Aware_Recommender_Retrain")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "ContextAwareRecommender")
PROMOTION_RATIO = float(os.getenv("PROMOTION_RATIO", "0.95"))

REQUIRED_INTERACTION_COLS = {"user_id", "product_id", "event_type", "device_type", "timestamp"}
VALID_EVENTS = {"view", "cart", "purchase"}
VALID_DEVICES = {"mobile", "desktop"}


def _log(msg: str) -> None:
    try:
        get_run_logger().info(msg)
    except Exception:
        print(f"[prefect] {msg}")


@task(name="ingest_and_validate_data", retries=2, retry_delay_seconds=3)
def ingest_and_validate_data(data_dir: str) -> Dict[str, int]:
    """Validate interactions.parquet and products.parquet against the data contract."""
    inter_path = os.path.join(data_dir, "interactions.parquet")
    prod_path = os.path.join(data_dir, "products.parquet")
    if not os.path.exists(inter_path) or not os.path.exists(prod_path):
        raise FileNotFoundError(f"Missing parquet files in {data_dir}. Run prepare_data.py first.")

    inter = pd.read_parquet(inter_path)
    prods = pd.read_parquet(prod_path)

    missing_cols = REQUIRED_INTERACTION_COLS - set(inter.columns)
    if missing_cols:
        raise ValueError(f"interactions.parquet violated contract; missing columns: {missing_cols}")

    invalid_events = set(inter["event_type"].unique()) - VALID_EVENTS
    if invalid_events:
        raise ValueError(f"Invalid event_type values found: {invalid_events}")

    invalid_devices = set(inter["device_type"].unique()) - VALID_DEVICES
    if invalid_devices:
        raise ValueError(f"Invalid device_type values found: {invalid_devices}")

    stats = {
        "n_interactions": int(len(inter)),
        "n_users": int(inter["user_id"].nunique()),
        "n_products": int(len(prods)),
    }
    _log(f"Validated data contract successfully: {stats}")
    return stats


@task(name="feature_engineering", retries=2, retry_delay_seconds=3)
def feature_engineering_task(data_dir: str) -> Dict[str, Any]:
    """Rebuild temporal train/test split and leak-free aggregate features."""
    inter = pd.read_parquet(os.path.join(data_dir, "interactions.parquet"))
    prods = pd.read_parquet(os.path.join(data_dir, "products.parquet"))
    split_stats = build_features_and_split(inter, prods, data_dir=data_dir)
    _log(f"Feature engineering complete: {split_stats}")
    return split_stats


@task(name="check_data_drift", retries=1, retry_delay_seconds=2)
def check_drift_task(data_dir: str) -> Dict[str, Any]:
    """Run PSI + KS-test drift detection and write HTML report."""
    drift_summary = detect_data_drift(data_dir=data_dir)
    _log(
        f"Drift check finished: overall_drift={drift_summary['overall_drift']} "
        f"max_psi={drift_summary['max_psi']}"
    )
    return drift_summary


@task(name="train_and_evaluate", retries=2, retry_delay_seconds=5)
def train_and_evaluate_task(data_dir: str, drift_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Train candidate LightGBM model, evaluate on test split, and log to MLflow."""
    products_df = pd.read_parquet(f"{data_dir}/products.parquet")
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    feat_user_cat = pd.read_parquet(f"{data_dir}/feat_user_cat.parquet")
    feat_ctx_item = pd.read_parquet(f"{data_dir}/feat_ctx_item.parquet")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"Retrain_LightGBM_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "algo": "LGBMClassifier",
            "n_estimators": 200,
            "learning_rate": 0.08,
            "num_leaves": 63,
            "context_aware": True,
            "drift_triggered": drift_summary.get("overall_drift", False),
            "max_psi": drift_summary.get("max_psi", 0.0),
        }
        mlflow.log_params(params)

        t0 = time.perf_counter()
        wrapper = train_lightgbm(train_df, n_estimators=200, learning_rate=0.08, num_leaves=63)
        train_sec = round(time.perf_counter() - t0, 3)

        metrics = evaluate_model(
            predict_fn=lambda df_in: wrapper.predict(None, df_in),
            test_df=test_df,
            products_df=products_df,
            feat_user_cat=feat_user_cat,
            feat_ctx_item=feat_ctx_item,
            k=10,
        )
        metrics["train_time_sec"] = train_sec
        mlflow.log_metrics(metrics)

        sample_in = test_df.drop(columns=["label"], errors="ignore").head(5)
        sample_out = wrapper.predict(None, sample_in)
        sig = infer_signature(sample_in, sample_out)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            code_paths=[MODELS_CODE_PATH],
            signature=sig,
            input_example=sample_in,
        )

        run_id = run.info.run_id
        _log(f"Logged candidate run {run_id} with NDCG@10={metrics['ndcg_at_10']}")
        return {"run_id": run_id, "metrics": metrics}


@task(name="compare_and_promote", retries=2, retry_delay_seconds=3)
def compare_and_promote_task(candidate: Dict[str, Any], drift_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare candidate NDCG@10 against current @Production and promote if >= 95%."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    new_run_id = candidate["run_id"]
    new_ndcg = float(candidate["metrics"]["ndcg_at_10"])

    old_ndcg = None
    old_version = None
    try:
        prod_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Production")
        old_version = prod_mv.version
        old_run = client.get_run(prod_mv.run_id)
        old_ndcg = float(old_run.data.metrics.get("ndcg_at_10", 0.0))
    except Exception:
        pass

    threshold = round(old_ndcg * PROMOTION_RATIO, 4) if old_ndcg is not None else 0.0
    should_promote = (old_ndcg is None) or (new_ndcg >= threshold)

    mv = mlflow.register_model(model_uri=f"runs:/{new_run_id}/model", name=REGISTERED_MODEL_NAME)

    if should_promote:
        reason = (
            f"Promoted: new NDCG@10 ({new_ndcg:.4f}) >= {PROMOTION_RATIO*100:.0f}% "
            f"of previous Production v{old_version} ({old_ndcg if old_ndcg is not None else 'N/A'})"
        )
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Production", mv.version)
        client.set_tag(new_run_id, "promotion_decision", "promoted")
        client.set_tag(new_run_id, "promotion_reason", reason)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "promotion_reason", reason)
        _notify_api_reload()
    else:
        reason = (
            f"Skipped: new NDCG@10 ({new_ndcg:.4f}) < {PROMOTION_RATIO*100:.0f}% "
            f"of Production v{old_version} ({old_ndcg:.4f}, threshold={threshold:.4f})"
        )
        client.set_tag(new_run_id, "promotion_decision", "skipped")
        client.set_tag(new_run_id, "promotion_reason", reason)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "promotion_reason", reason)

    result = {
        "candidate_version": str(mv.version),
        "previous_version": str(old_version) if old_version else None,
        "new_ndcg_at_10": new_ndcg,
        "previous_ndcg_at_10": old_ndcg,
        "promoted": should_promote,
        "reason": reason,
        "drift_detected": drift_summary.get("overall_drift", False),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _log(f"Promotion decision: {result}")

    # Persist latest retrain status so API & Frontend can display it
    status_path = Path(DATA_DIR) / "last_retrain.json"
    try:
        status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass
    return result


def _notify_api_reload() -> None:
    """Trigger POST /admin/reload on the FastAPI service so it swaps models immediately."""
    for base in [API_BASE_URL, "http://127.0.0.1:8000"]:
        url = f"{base.rstrip('/')}/admin/reload"
        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    _log(f"Notified API reload at {url}")
                    return
        except Exception:
            continue


@flow(name="context-recsys-retrain-flow", log_prints=True)
def retrain_flow(data_dir: str = DATA_DIR, force: bool = True) -> Dict[str, Any]:
    """End-to-end Prefect retraining flow."""
    val_stats = ingest_and_validate_data(data_dir)
    feat_stats = feature_engineering_task(data_dir)
    drift_summary = check_drift_task(data_dir)

    if not force and not drift_summary.get("overall_drift", False):
        _log("No significant drift detected and force=False; skipping training.")
        return {"skipped": True, "drift": drift_summary}

    candidate = train_and_evaluate_task(data_dir, drift_summary)
    outcome = compare_and_promote_task(candidate, drift_summary)
    return {
        "validation": val_stats,
        "features": feat_stats,
        "drift": drift_summary,
        "promotion": outcome,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or serve Prefect retraining flow")
    parser.add_argument("--serve", action="store_true", help="Run continuously every 24h")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    args = parser.parse_args()

    if not args.serve:
        retrain_flow(data_dir=DATA_DIR, force=True)
        return

    print(f"[flow_retrain] Starting scheduled retraining service (interval={args.interval_hours}h)...")
    try:
        retrain_flow.serve(
            name="context-recsys-daily-retrain",
            interval=datetime.timedelta(hours=args.interval_hours),
        )
    except Exception as exc:
        print(f"[flow_retrain] Prefect server unavailable ({exc}); running built-in 24h scheduler loop.")
        while True:
            try:
                if os.path.exists(os.path.join(DATA_DIR, "interactions.parquet")):
                    retrain_flow(data_dir=DATA_DIR, force=True)
            except Exception as run_exc:
                print(f"[flow_retrain] Scheduled run error: {run_exc}")
            time.sleep(int(args.interval_hours * 3600))


if __name__ == "__main__":
    main()
