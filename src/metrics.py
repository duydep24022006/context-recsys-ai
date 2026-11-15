"""Evaluation metrics for Top-K recommendation and binary classification (CLO1)."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models import CAT_MAP


def precision_at_k(ranked: List[int], ground_truth: Set[int], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for pid in top_k if pid in ground_truth)
    return float(hits) / float(k)


def recall_at_k(ranked: List[int], ground_truth: Set[int], k: int = 10) -> float:
    if not ground_truth or k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for pid in top_k if pid in ground_truth)
    return float(hits) / float(len(ground_truth))


def ndcg_at_k(ranked: List[int], ground_truth: Set[int], k: int = 10) -> float:
    if not ground_truth or k <= 0:
        return 0.0
    top_k = ranked[:k]
    dcg = sum(
        (1.0 / np.log2(idx + 2))
        for idx, pid in enumerate(top_k)
        if pid in ground_truth
    )
    ideal_len = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_len))
    return float(dcg / idcg) if idcg > 0 else 0.0


def map_at_k(ranked: List[int], ground_truth: Set[int], k: int = 10) -> float:
    if not ground_truth or k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = 0.0
    sum_precisions = 0.0
    for idx, pid in enumerate(top_k):
        if pid in ground_truth:
            hits += 1.0
            sum_precisions += hits / float(idx + 1)
    denom = float(min(k, len(ground_truth)))
    return float(sum_precisions / denom) if denom > 0 else 0.0


def evaluate_model(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    test_df: pd.DataFrame,
    products_df: pd.DataFrame,
    feat_user_cat: pd.DataFrame,
    feat_ctx_item: pd.DataFrame,
    k: int = 10,
    max_queries: int = 200,
) -> Dict[str, float]:
    """Evaluate a recommendation model on AUC and Top-K ranking metrics."""
    # 1. Pointwise ROC-AUC on test_df (includes positive + negative samples)
    scores_all = np.asarray(predict_fn(test_df), dtype=np.float64)
    y_true = test_df["label"].to_numpy(dtype=np.int32)
    if len(np.unique(y_true)) > 1:
        auc_val = float(roc_auc_score(y_true, scores_all))
    else:
        auc_val = 0.5

    # 2. Top-K ranking evaluation across user-context queries
    pos_test = test_df[test_df["label"] == 1]
    if pos_test.empty:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"ndcg_at_{k}": 0.0,
            f"map_at_{k}": 0.0,
            "auc": round(auc_val, 4),
            "inference_ms_per_query": 0.0,
        }

    # Pre-index product catalog and lookups for fast candidate construction
    prods = products_df.sort_values("product_id").reset_index(drop=True)
    all_pids = prods["product_id"].to_numpy(dtype=np.int32)
    all_cats = prods["category"].astype(str).to_numpy()
    all_cat_codes = prods["category"].map(CAT_MAP).fillna(0).to_numpy(dtype=np.int32)
    all_prices = prods["price"].to_numpy(dtype=np.float32)
    n_items = len(prods)

    user_cat_map: Dict[tuple, float] = {}
    for row in feat_user_cat.itertuples(index=False):
        user_cat_map[(int(row.user_id), str(row.category))] = float(row.user_cat_ratio)

    ctx_item_map: Dict[tuple, float] = {}
    for row in feat_ctx_item.itertuples(index=False):
        ctx_item_map[(int(row.is_mobile), int(row.time_slot), int(row.product_id))] = float(row.ctx_item_pop)

    grouped = list(pos_test.groupby(["user_id", "is_mobile", "time_slot"]))
    if len(grouped) > max_queries:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(grouped), size=max_queries, replace=False)
        grouped = [grouped[i] for i in sorted(indices)]

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    maps: List[float] = []
    latencies_ms: List[float] = []

    for (uid, is_mob, t_slot), grp in grouped:
        gt_pids: Set[int] = set(grp["product_id"].astype(int).tolist())
        sample_row = grp.iloc[0]
        hour_val = int(sample_row["hour"])
        wknd_val = int(sample_row["is_weekend"])

        u_cat_arr = np.array(
            [user_cat_map.get((int(uid), c), 0.2) for c in all_cats],
            dtype=np.float32,
        )
        ctx_pop_arr = np.array(
            [ctx_item_map.get((int(is_mob), int(t_slot), int(pid)), 0.05) for pid in all_pids],
            dtype=np.float32,
        )

        cand_df = pd.DataFrame({
            "user_id": np.full(n_items, int(uid), dtype=np.int32),
            "product_id": all_pids,
            "hour": np.full(n_items, hour_val, dtype=np.int32),
            "is_weekend": np.full(n_items, wknd_val, dtype=np.int32),
            "is_mobile": np.full(n_items, int(is_mob), dtype=np.int32),
            "time_slot": np.full(n_items, int(t_slot), dtype=np.int32),
            "category": all_cats,
            "cat_code": all_cat_codes,
            "price": all_prices,
            "user_cat_ratio": u_cat_arr,
            "ctx_item_pop": ctx_pop_arr,
        })

        t0 = time.perf_counter()
        scores = np.asarray(predict_fn(cand_df), dtype=np.float64)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        top_idx = np.argsort(-scores)[:k]
        ranked_pids = all_pids[top_idx].tolist()

        precisions.append(precision_at_k(ranked_pids, gt_pids, k))
        recalls.append(recall_at_k(ranked_pids, gt_pids, k))
        ndcgs.append(ndcg_at_k(ranked_pids, gt_pids, k))
        maps.append(map_at_k(ranked_pids, gt_pids, k))

    return {
        f"precision_at_{k}": round(float(np.mean(precisions)), 4),
        f"recall_at_{k}": round(float(np.mean(recalls)), 4),
        f"ndcg_at_{k}": round(float(np.mean(ndcgs)), 4),
        f"map_at_{k}": round(float(np.mean(maps)), 4),
        "auc": round(auc_val, 4),
        "inference_ms_per_query": round(float(np.mean(latencies_ms)), 3),
    }
