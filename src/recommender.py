"""Real-time Context-Aware Recommendation Service.

Loads `models:/ContextAwareRecommender@Production` from the MLflow Model Registry,
caches product and context feature matrices in memory (zero per-request DataFrame
merges), handles cold-start users gracefully, and auto-checks for newly promoted
Production model versions.
"""
from __future__ import annotations

import datetime
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

import models  # noqa: F401 - ensure wrapper classes are importable for unpickling
from models import CAT_MAP, time_slot_from_hour

logger = logging.getLogger("recsys.recommender")

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "ContextAwareRecommender")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")
VERSION_CHECK_INTERVAL_SEC = float(os.getenv("VERSION_CHECK_INTERVAL_SEC", "30"))


class RecommenderService:
    """In-memory cached recommender engine backed by MLflow Registry."""

    def __init__(self) -> None:
        self.model: Optional[mlflow.pyfunc.PyFuncModel] = None
        self.model_meta: Dict[str, Any] = {
            "name": REGISTERED_MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": None,
            "run_id": None,
            "model_name_tag": None,
            "metrics": {},
            "loaded_at": None,
        }
        self._lock = threading.Lock()
        self._last_version_check: float = 0.0

        # Precomputed in-memory catalog & feature arrays
        self.products_df: Optional[pd.DataFrame] = None
        self.all_pids: np.ndarray = np.array([], dtype=np.int32)
        self.all_cats: np.ndarray = np.array([], dtype=object)
        self.all_cat_codes: np.ndarray = np.array([], dtype=np.int32)
        self.all_prices: np.ndarray = np.array([], dtype=np.float32)
        self.known_users: Set[int] = set()
        self.user_cat_vectors: Dict[int, np.ndarray] = {}
        self.ctx_item_vectors: Dict[Tuple[int, int], np.ndarray] = {}
        self.default_user_cat_vec: np.ndarray = np.array([], dtype=np.float32)
        self.last_scores: Dict[int, float] = {}
        self.last_cold_start: bool = False

    @property
    def data_dir(self) -> str:
        return os.getenv("DATA_DIR", "./shared_data")

    @property
    def tracking_uri(self) -> str:
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        if uri.startswith("http"):
            parsed = urlparse(uri)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 80
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return uri
            except OSError:
                local_db = os.path.abspath("mlflow_store/mlflow.db")
                if os.path.exists(local_db):
                    return f"sqlite:///{local_db.replace(chr(92), '/')}"
                raise RuntimeError(f"MLflow server at {uri} is unreachable")
        return uri

    def refresh_feature_cache(self) -> None:
        """Precompute product and context feature vectors in RAM for sub-10ms inference."""
        data_dir = self.data_dir
        prod_path = os.path.join(data_dir, "products.parquet")
        if not os.path.exists(prod_path):
            return

        prods = pd.read_parquet(prod_path).sort_values("product_id").reset_index(drop=True)
        self.products_df = prods.set_index("product_id", drop=False)
        self.all_pids = prods["product_id"].to_numpy(dtype=np.int32)
        self.all_cats = prods["category"].astype(str).to_numpy()
        self.all_cat_codes = prods["category"].map(CAT_MAP).fillna(0).to_numpy(dtype=np.int32)
        self.all_prices = prods["price"].to_numpy(dtype=np.float32)
        n_items = len(prods)
        self.default_user_cat_vec = np.full(n_items, 0.2, dtype=np.float32)

        pid_to_idx = {int(pid): idx for idx, pid in enumerate(self.all_pids)}

        # Precompute user -> item-aligned user_cat_ratio vector
        uc_path = os.path.join(data_dir, "feat_user_cat.parquet")
        user_vecs: Dict[int, np.ndarray] = {}
        known_u: Set[int] = set()
        if os.path.exists(uc_path):
            uc_df = pd.read_parquet(uc_path)
            known_u = set(uc_df["user_id"].astype(int).unique().tolist())
            for uid, grp in uc_df.groupby("user_id"):
                cat_ratio = dict(zip(grp["category"].astype(str), grp["user_cat_ratio"].astype(float)))
                vec = np.array([cat_ratio.get(c, 0.05) for c in self.all_cats], dtype=np.float32)
                user_vecs[int(uid)] = vec
        self.user_cat_vectors = user_vecs
        self.known_users = known_u

        # Precompute (is_mobile, time_slot) -> item-aligned ctx_item_pop vector
        ci_path = os.path.join(data_dir, "feat_ctx_item.parquet")
        ctx_vecs: Dict[Tuple[int, int], np.ndarray] = {}
        for is_mob in (0, 1):
            for slot in (0, 1, 2, 3):
                ctx_vecs[(is_mob, slot)] = np.full(n_items, 0.05, dtype=np.float32)
        if os.path.exists(ci_path):
            ci_df = pd.read_parquet(ci_path)
            for (is_mob, slot), grp in ci_df.groupby(["is_mobile", "time_slot"]):
                vec = np.full(n_items, 0.02, dtype=np.float32)
                for r in grp.itertuples(index=False):
                    idx = pid_to_idx.get(int(r.product_id))
                    if idx is not None:
                        vec[idx] = float(r.ctx_item_pop)
                ctx_vecs[(int(is_mob), int(slot))] = vec
        self.ctx_item_vectors = ctx_vecs

    def reload_model(self) -> Dict[str, Any]:
        """Reload the `@Production` model from MLflow Registry and refresh feature cache."""
        with self._lock:
            self.refresh_feature_cache()
            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient(tracking_uri=self.tracking_uri)

            try:
                mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
            except Exception:
                # Fallback if alias is not supported or only stage exists
                versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
                if not versions:
                    raise RuntimeError(f"No model versions registered for {REGISTERED_MODEL_NAME}")
                mv = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

            model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
            try:
                loaded = mlflow.pyfunc.load_model(model_uri)
            except Exception:
                loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{mv.version}")

            run_metrics: Dict[str, float] = {}
            model_name_tag = mv.name
            try:
                run = client.get_run(mv.run_id)
                run_metrics = {k: round(float(v), 4) for k, v in run.data.metrics.items()}
                model_name_tag = run.data.tags.get("mlflow.runName", mv.name)
            except Exception:
                pass

            self.model = loaded
            self.model_meta = {
                "name": REGISTERED_MODEL_NAME,
                "alias": MODEL_ALIAS,
                "version": str(mv.version),
                "run_id": mv.run_id,
                "model_name_tag": model_name_tag,
                "metrics": run_metrics,
                "loaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self._last_version_check = time.monotonic()
            logger.info("Loaded model %s v%s (run=%s)", REGISTERED_MODEL_NAME, mv.version, mv.run_id)
            return self.model_meta

    def maybe_auto_reload(self) -> None:
        """Periodically check if a newer version has been promoted to @Production."""
        now = time.monotonic()
        if self.model is not None and (now - self._last_version_check) < VERSION_CHECK_INTERVAL_SEC:
            return
        self._last_version_check = now
        try:
            client = MlflowClient(tracking_uri=self.tracking_uri)
            mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
            if self.model is None or str(mv.version) != str(self.model_meta.get("version")):
                self.reload_model()
        except Exception:
            pass

    def predict_with_scores(
        self,
        user_id: int,
        device: str,
        hour: int,
        top_k: int = 10,
        is_weekend: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], bool]:
        """Return [(product_id, score), ...] and whether user_id is cold-start."""
        if len(self.all_pids) == 0:
            self.refresh_feature_cache()
        self.maybe_auto_reload()

        if len(self.all_pids) == 0:
            raise RuntimeError("Product catalog is not available in DATA_DIR")

        is_mob = 1 if str(device).lower() == "mobile" else 0
        t_slot = int(time_slot_from_hour(hour))
        if is_weekend is None:
            is_wknd = 1 if datetime.datetime.now().weekday() >= 5 else 0
        else:
            is_wknd = int(is_weekend)

        is_cold = int(user_id) not in self.known_users
        ctx_vec = self.ctx_item_vectors.get((is_mob, t_slot), self.default_user_cat_vec)

        n_items = len(self.all_pids)
        u_vec = self.user_cat_vectors.get(int(user_id), self.default_user_cat_vec)

        if self.model is None:
            # Fallback to context-aware popularity + user category affinity if model not loaded
            scores = 0.7 * ctx_vec.astype(np.float64) + 0.3 * u_vec.astype(np.float64)
        else:
            cand_df = pd.DataFrame({
                "user_id": np.full(n_items, int(user_id), dtype=np.int32),
                "product_id": self.all_pids,
                "hour": np.full(n_items, int(hour), dtype=np.int32),
                "is_weekend": np.full(n_items, is_wknd, dtype=np.int32),
                "is_mobile": np.full(n_items, is_mob, dtype=np.int32),
                "time_slot": np.full(n_items, t_slot, dtype=np.int32),
                "category": self.all_cats,
                "cat_code": self.all_cat_codes,
                "price": self.all_prices,
                "user_cat_ratio": u_vec,
                "ctx_item_pop": ctx_vec,
            })

            raw_scores = np.asarray(self.model.predict(cand_df), dtype=np.float64)
            if is_cold:
                # Blend model contextual score with context popularity for cold-start robustness
                scores = 0.6 * raw_scores + 0.4 * ctx_vec.astype(np.float64)
            else:
                scores = raw_scores

        top_idx = np.argsort(-scores)[:top_k]
        ranked = [(int(self.all_pids[i]), round(float(scores[i]), 4)) for i in top_idx]
        self.last_scores = {pid: sc for pid, sc in ranked}
        self.last_cold_start = is_cold
        return ranked, is_cold


_service = RecommenderService()


def reload_model() -> Dict[str, Any]:
    return _service.reload_model()


def get_model_info() -> Dict[str, Any]:
    _service.maybe_auto_reload()
    return _service.model_meta


def predict(user_id: int, device: str, hour: int, top_k: int = 10) -> List[int]:
    """Return top_k product_ids for (user_id, device, hour). Kept compatible with existing tests."""
    ranked, _ = _service.predict_with_scores(user_id=user_id, device=device, hour=hour, top_k=top_k)
    return [pid for pid, _score in ranked]


def predict_detailed(
    user_id: int,
    device: str,
    hour: int,
    top_k: int = 10,
    is_weekend: Optional[int] = None,
) -> Tuple[List[Tuple[int, float]], bool]:
    """Return ([(product_id, score), ...], is_cold_start)."""
    return _service.predict_with_scores(
        user_id=user_id, device=device, hour=hour, top_k=top_k, is_weekend=is_weekend
    )
