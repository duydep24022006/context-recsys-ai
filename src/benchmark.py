"""Multi-seed Model Benchmark & Ablation Study (CLO1).

Compares 6 models + 1 ablation across 3 random seeds:
1. Popularity_Baseline
2. Content_Based_Filtering (TF-IDF + Category + Price)
3. Collaborative_Filtering_MF (TruncatedSVD, tuned rank)
4. LightGBM_No_Context (Ablation: without context features)
5. LightGBM_Context_Aware (Tuned GBDT with context features)
6. XGBoost_Context_Aware (Tuned Hist-GBDT with context features)
7. NCF_PyTorch_CPU (Context-Aware Neural Collaborative Filtering with Early Stopping)

Logs parameters, metrics, model signatures, feature importances, and artifacts
to MLflow, promotes the best model (by NDCG@10) to `ContextAwareRecommender@Production`,
and exports tables/figures to `docs/results/` and `docs/figures/`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from metrics import evaluate_model
from models import (
    LGB_FEATURES,
    MODELS_CODE_PATH,
    NCFWrapper,
    train_content_based,
    train_lightgbm,
    train_matrix_factorization,
    train_ncf,
    train_popularity_baseline,
    train_xgboost,
)

DATA_DIR = os.getenv("DATA_DIR", "./shared_data")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Context_Aware_Recommender_CLO1")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "ContextAwareRecommender")
DEFAULT_SEEDS = [42, 100, 2026]


def _model_size_kb(pyfunc_model: Any) -> float:
    try:
        raw = pickle.dumps(pyfunc_model)
        return round(len(raw) / 1024.0, 2)
    except Exception:
        return 0.0


def tune_hyperparameters(train_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Run a fast validation grid search on the last 15% of train_df to select hyperparameters."""
    split_idx = int(len(train_df) * 0.85)
    tr_sub = train_df.iloc[:split_idx]
    val_sub = train_df.iloc[split_idx:]
    y_val = val_sub["label"].to_numpy(dtype=np.int32)

    from sklearn.metrics import roc_auc_score

    # 1. Tune MF n_components
    best_mf_k, best_mf_auc = 24, -1.0
    for k_cand in [16, 24, 32]:
        m = train_matrix_factorization(tr_sub, products_df, n_components=k_cand, seed=42)
        auc = float(roc_auc_score(y_val, m.predict(None, val_sub)))
        if auc > best_mf_auc:
            best_mf_auc, best_mf_k = auc, k_cand

    # 2. Tune LightGBM
    best_lgb, best_lgb_auc = {"n_estimators": 180, "learning_rate": 0.05, "num_leaves": 31}, -1.0
    for n_est, lr, leaves in [(150, 0.05, 31), (220, 0.05, 31), (200, 0.08, 63)]:
        m = train_lightgbm(tr_sub, n_estimators=n_est, learning_rate=lr, num_leaves=leaves, seed=42)
        auc = float(roc_auc_score(y_val, m.predict(None, val_sub)))
        if auc > best_lgb_auc:
            best_lgb_auc = auc
            best_lgb = {"n_estimators": n_est, "learning_rate": lr, "num_leaves": leaves}

    # 3. Tune XGBoost
    best_xgb, best_xgb_auc = {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 6}, -1.0
    for n_est, lr, depth in [(160, 0.05, 5), (200, 0.05, 6), (220, 0.08, 6)]:
        m = train_xgboost(tr_sub, n_estimators=n_est, learning_rate=lr, max_depth=depth, seed=42)
        auc = float(roc_auc_score(y_val, m.predict(None, val_sub)))
        if auc > best_xgb_auc:
            best_xgb_auc = auc
            best_xgb = {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth}

    return {
        "mf": {"n_components": best_mf_k},
        "lgb": best_lgb,
        "xgb": best_xgb,
        "ncf": {"emb_dim": 16, "epochs": 10, "lr": 0.004, "patience": 3},
    }


def _save_plots_and_tables(
    agg_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ncf_history: Dict[str, List[float]],
    feat_importances: Dict[str, float],
    out_results_dir: str = "docs/results",
    out_figures_dir: str = "docs/figures",
) -> None:
    Path(out_results_dir).mkdir(parents=True, exist_ok=True)
    Path(out_figures_dir).mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    agg_df.to_csv(f"{out_results_dir}/benchmark_results.csv", index=False)
    raw_df.to_csv(f"{out_results_dir}/benchmark_runs_detailed.csv", index=False)

    # Also save JSON to shared_data for Frontend Model Tab
    try:
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(f"{DATA_DIR}/benchmark_results.json").write_text(
            agg_df.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # 2. Save Markdown report with mean ± std and Ablation analysis
    md_lines = [
        "# Kết quả Đánh giá & So sánh Mô hình (CLO1)",
        "",
        "Bảng dưới đây tổng hợp kết quả thực nghiệm trên tập kiểm thử theo thời gian (Temporal Split 80/20, không rò rỉ dữ liệu) qua **3 random seeds (`42, 100, 2026`)**, báo cáo dưới dạng `mean ± std`.",
        "",
        "| Mô hình | Context-Aware | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | ROC-AUC | Latency (ms/query) | Train Time (s) | Size (KB) |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_df.itertuples(index=False):
        ctx_str = "Có" if r.context_aware else "Không"
        md_lines.append(
            f"| **{r.Model}** | {ctx_str} "
            f"| {r.precision_at_10_mean:.4f} ± {r.precision_at_10_std:.4f} "
            f"| {r.recall_at_10_mean:.4f} ± {r.recall_at_10_std:.4f} "
            f"| **{r.ndcg_at_10_mean:.4f} ± {r.ndcg_at_10_std:.4f}** "
            f"| {r.map_at_10_mean:.4f} ± {r.map_at_10_std:.4f} "
            f"| {r.auc_mean:.4f} ± {r.auc_std:.4f} "
            f"| {r.inference_ms_per_query_mean:.2f} "
            f"| {r.train_time_sec_mean:.2f} "
            f"| {r.model_size_kb_mean:.1f} |"
        )

    # Compute ablation delta
    try:
        lgb_ctx = float(agg_df.loc[agg_df["Model"] == "LightGBM_Context_Aware", "ndcg_at_10_mean"].iloc[0])
        lgb_no = float(agg_df.loc[agg_df["Model"] == "LightGBM_No_Context", "ndcg_at_10_mean"].iloc[0])
        rel_gain = ((lgb_ctx - lgb_no) / max(lgb_no, 1e-6)) * 100.0
        md_lines.extend([
            "",
            "## Nghiên cứu Ablation: Giá trị của Ngữ cảnh (Context Features)",
            "",
            f"- **LightGBM (Có Context - `hour, is_weekend, is_mobile, time_slot, ctx_item_pop`)**: NDCG@10 = `{lgb_ctx:.4f}`",
            f"- **LightGBM (Không Context - chỉ `user_id, product_id, category, price, user_cat_ratio`)**: NDCG@10 = `{lgb_no:.4f}`",
            f"- **Mức cải thiện tương đối**: **+{rel_gain:.2f}%** trên chỉ số NDCG@10 khi tích hợp tín hiệu ngữ cảnh thời gian thực.",
        ])
    except Exception:
        pass

    Path(f"{out_results_dir}/benchmark_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 3. Plot Model Comparison Bar Chart (docs/figures/model_comparison.png)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    models = agg_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    axes[0].bar(x - width, agg_df["ndcg_at_10_mean"], width, yerr=agg_df["ndcg_at_10_std"], label="NDCG@10", color="#0b6e6e", capsize=3)
    axes[0].bar(x, agg_df["map_at_10_mean"], width, yerr=agg_df["map_at_10_std"], label="MAP@10", color="#2b8cbe", capsize=3)
    axes[0].bar(x + width, agg_df["recall_at_10_mean"], width, yerr=agg_df["recall_at_10_std"], label="Recall@10", color="#7bccc4", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Top-10 Ranking Metrics across Models (Mean ± Std)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, agg_df["auc_mean"], width, yerr=agg_df["auc_std"], label="ROC-AUC", color="#0b6e6e", capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_title("Pointwise ROC-AUC & Query Latency")
    axes[1].grid(axis="y", alpha=0.3)

    ax2 = axes[1].twinx()
    ax2.plot(x, agg_df["inference_ms_per_query_mean"], color="#e34a33", marker="o", lw=2, label="Latency (ms/query)")
    ax2.set_ylabel("Inference Latency (ms/query)", color="#e34a33")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{out_figures_dir}/model_comparison.png", dpi=150)
    plt.close(fig)

    # 4. Plot NCF Learning Curve (docs/figures/ncf_learning_curve.png)
    if ncf_history and ncf_history.get("train_loss"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ep = range(1, len(ncf_history["train_loss"]) + 1)
        ax.plot(ep, ncf_history["train_loss"], marker="o", lw=2, color="#0b6e6e", label="Train BCE Loss")
        if ncf_history.get("val_loss"):
            ax.plot(ep, ncf_history["val_loss"], marker="s", lw=2, color="#e34a33", label="Validation BCE Loss")
        ax.set_title("NCF (PyTorch CPU) Learning Curve with Early Stopping")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_figures_dir}/ncf_learning_curve.png", dpi=150)
        plt.close(fig)

    # 5. Plot Feature Importance for LightGBM (docs/figures/feature_importance.png)
    if feat_importances:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sorted_items = sorted(feat_importances.items(), key=lambda kv: kv[1])
        f_names = [k for k, _ in sorted_items]
        f_vals = [v for _, v in sorted_items]
        ax.barh(f_names, f_vals, color="#0b6e6e")
        ax.set_title("LightGBM Context-Aware Feature Importance (Splits)")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_figures_dir}/feature_importance.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed CLO1 model benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    seeds: List[int] = args.seeds

    print(f"1. Reading dataset from {DATA_DIR} ...")
    products_df = pd.read_parquet(f"{DATA_DIR}/products.parquet")
    train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    test_df = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    feat_user_cat = pd.read_parquet(f"{DATA_DIR}/feat_user_cat.parquet")
    feat_ctx_item = pd.read_parquet(f"{DATA_DIR}/feat_ctx_item.parquet")

    print("2. Tuning hyperparameters on validation split ...")
    tuned = tune_hyperparameters(train_df, products_df)
    print(f"   Tuned params: {tuned}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    sample_input = test_df.drop(columns=["label"], errors="ignore").head(5)

    def _build_specs(seed: int) -> List[Tuple[str, Dict[str, Any], Callable[[], Any]]]:
        return [
            (
                "Popularity_Baseline",
                {"algo": "GlobalItemPopularity", "context_aware": False, "seed": seed},
                lambda: train_popularity_baseline(train_df, products_df),
            ),
            (
                "Content_Based_Filtering",
                {"algo": "TFIDF_Category_Cosine", "max_features": 64, "context_aware": False, "seed": seed},
                lambda: train_content_based(train_df, products_df, max_features=64),
            ),
            (
                "Collaborative_Filtering_MF",
                {"algo": "TruncatedSVD", **tuned["mf"], "context_aware": False, "seed": seed},
                lambda: train_matrix_factorization(train_df, products_df, seed=seed, **tuned["mf"]),
            ),
            (
                "LightGBM_No_Context",
                {"algo": "LGBMClassifier_NoContext", **tuned["lgb"], "context_aware": False, "seed": seed},
                lambda: train_lightgbm(train_df, seed=seed, include_context=False, **tuned["lgb"]),
            ),
            (
                "LightGBM_Context_Aware",
                {"algo": "LGBMClassifier", **tuned["lgb"], "context_aware": True, "seed": seed},
                lambda: train_lightgbm(train_df, seed=seed, include_context=True, **tuned["lgb"]),
            ),
            (
                "XGBoost_Context_Aware",
                {"algo": "XGBClassifier", **tuned["xgb"], "context_aware": True, "seed": seed},
                lambda: train_xgboost(train_df, seed=seed, include_context=True, **tuned["xgb"]),
            ),
            (
                "NCF_PyTorch_CPU",
                {"algo": "ContextNCF_MLP", **tuned["ncf"], "context_aware": True, "seed": seed},
                lambda: train_ncf(train_df, products_df, seed=seed, **tuned["ncf"]),
            ),
        ]

    all_runs: List[Dict[str, Any]] = []
    best_model_name, best_ndcg, best_run_id = None, -1.0, None
    last_ncf_history: Dict[str, List[float]] = {}
    lgb_feat_imp: Dict[str, float] = {}

    print(f"3. Running benchmark across {len(seeds)} seeds: {seeds} ...\n")
    for seed_idx, seed in enumerate(seeds):
        is_primary_seed = (seed_idx == 0)
        for model_name, params, train_fn in _build_specs(seed):
            run_label = f"{model_name}_seed{seed}" if len(seeds) > 1 else model_name
            print(f"--> [{run_label}] training & evaluating ...")
            with mlflow.start_run(run_name=run_label) as run:
                mlflow.set_tag("model_family", model_name)
                mlflow.set_tag("seed", str(seed))
                mlflow.log_params(params)

                t0 = time.perf_counter()
                pyfunc_model = train_fn()
                train_time_sec = round(time.perf_counter() - t0, 3)
                size_kb = _model_size_kb(pyfunc_model)

                metrics = evaluate_model(
                    predict_fn=lambda df_in, m=pyfunc_model: m.predict(None, df_in),
                    test_df=test_df,
                    products_df=products_df,
                    feat_user_cat=feat_user_cat,
                    feat_ctx_item=feat_ctx_item,
                    k=10,
                )
                metrics["train_time_sec"] = train_time_sec
                metrics["model_size_kb"] = size_kb
                mlflow.log_metrics(metrics)

                if isinstance(pyfunc_model, NCFWrapper) and pyfunc_model.history:
                    last_ncf_history = pyfunc_model.history
                    for ep_i, (tr_l, va_l) in enumerate(
                        zip(pyfunc_model.history["train_loss"], pyfunc_model.history["val_loss"]), start=1
                    ):
                        mlflow.log_metric("epoch_train_loss", tr_l, step=ep_i)
                        mlflow.log_metric("epoch_val_loss", va_l, step=ep_i)

                if model_name == "LightGBM_Context_Aware" and hasattr(pyfunc_model, "booster"):
                    importances = pyfunc_model.booster.feature_importances_
                    lgb_feat_imp = {
                        fname: float(val) for fname, val in zip(LGB_FEATURES, importances)
                    }
                    mlflow.log_dict(lgb_feat_imp, "feature_importance.json")

                # Log model artifact on primary seed (or if it achieves a new global best NDCG@10)
                is_new_best = metrics["ndcg_at_10"] > best_ndcg
                if is_primary_seed or is_new_best:
                    sample_out = pyfunc_model.predict(None, sample_input)
                    sig = infer_signature(sample_input, sample_out)
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=pyfunc_model,
                        code_paths=[MODELS_CODE_PATH],
                        signature=sig,
                        input_example=sample_input,
                    )
                    if is_new_best:
                        best_ndcg = metrics["ndcg_at_10"]
                        best_model_name = model_name
                        best_run_id = run.info.run_id

                all_runs.append({
                    "Model": model_name,
                    "seed": seed,
                    "context_aware": bool(params["context_aware"]),
                    **metrics,
                    "run_id": run.info.run_id,
                })

    raw_df = pd.DataFrame(all_runs)
    metric_cols = [
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "map_at_10",
        "auc",
        "inference_ms_per_query",
        "train_time_sec",
        "model_size_kb",
    ]
    agg_rows = []
    for model_name, grp in raw_df.groupby("Model", sort=False):
        row_dict: Dict[str, Any] = {
            "Model": model_name,
            "context_aware": bool(grp["context_aware"].iloc[0]),
            "n_seeds": int(len(grp)),
        }
        for col in metric_cols:
            row_dict[f"{col}_mean"] = round(float(grp[col].mean()), 4)
            row_dict[f"{col}_std"] = round(float(grp[col].std(ddof=0)), 4)
            row_dict[col] = row_dict[f"{col}_mean"]
        agg_rows.append(row_dict)

    agg_df = pd.DataFrame(agg_rows).sort_values("ndcg_at_10_mean", ascending=False).reset_index(drop=True)
    _save_plots_and_tables(agg_df, raw_df, last_ncf_history, lgb_feat_imp)

    display_cols = [
        "Model", "context_aware", "precision_at_10", "recall_at_10",
        "ndcg_at_10", "map_at_10", "auc", "inference_ms_per_query", "train_time_sec",
    ]
    print("\n==================== BANG SO SANH MO HINH (CLO1 - MEAN ACROSS SEEDS) ====================")
    print(agg_df[display_cols].to_string(index=False))

    print(f"\n4. Registering best model [{best_model_name}] (NDCG@10={best_ndcg}) to MLflow Registry...")
    mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Production", mv.version)
    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception:
        pass

    print(f"DONE! Model '{REGISTERED_MODEL_NAME}' version {mv.version} is now aliased as 'Production'.")


if __name__ == "__main__":
    main()
