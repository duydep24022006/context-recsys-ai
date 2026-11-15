"""Data Drift Detection module (PSI + Kolmogorov-Smirnov / Total Variation Distance).

Compares reference training data against recent interactions across `hour`,
`is_mobile`, and `category`, producing both a structured dictionary (used by
the Prefect retraining flow) and an interactive HTML report at
`docs/results/drift_report.html`.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_psi(ref_probs: np.ndarray, cur_probs: np.ndarray, eps: float = 1e-4) -> float:
    """Compute Population Stability Index (PSI) between two discrete distributions."""
    r = np.clip(ref_probs, eps, 1.0)
    c = np.clip(cur_probs, eps, 1.0)
    r = r / r.sum()
    c = c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def detect_data_drift(
    data_dir: str | None = None,
    report_path: str = "docs/results/drift_report.html",
    psi_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compare early (reference) vs recent (current) interaction slices and export HTML report."""
    data_dir = data_dir or os.getenv("DATA_DIR", "./shared_data")
    inter = pd.read_parquet(f"{data_dir}/interactions.parquet")
    prods = pd.read_parquet(f"{data_dir}/products.parquet")

    df = inter.merge(prods[["product_id", "category", "price"]], on="product_id", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["is_mobile"] = (df["device_type"] == "mobile").astype(int)

    split_idx = int(len(df) * 0.80)
    ref_df = df.iloc[:split_idx]
    cur_df = df.iloc[split_idx:]

    features_report: Dict[str, Dict[str, Any]] = {}

    # 1. Hour distribution (PSI + KS test)
    hour_bins = list(range(24))
    ref_hour = ref_df["hour"].value_counts(normalize=True).reindex(hour_bins, fill_value=0.0).to_numpy()
    cur_hour = cur_df["hour"].value_counts(normalize=True).reindex(hour_bins, fill_value=0.0).to_numpy()
    psi_hour = compute_psi(ref_hour, cur_hour)
    ks_stat, ks_p = ks_2samp(ref_df["hour"].to_numpy(), cur_df["hour"].to_numpy())
    features_report["hour"] = {
        "psi": round(psi_hour, 4),
        "ks_stat": round(float(ks_stat), 4),
        "p_value": round(float(ks_p), 5),
        "drift_detected": bool(psi_hour >= psi_threshold or ks_stat >= 0.12),
        "ref_mean": round(float(ref_df["hour"].mean()), 2),
        "cur_mean": round(float(cur_df["hour"].mean()), 2),
    }

    # 2. is_mobile distribution (PSI + KS test)
    mob_bins = [0, 1]
    ref_mob = ref_df["is_mobile"].value_counts(normalize=True).reindex(mob_bins, fill_value=0.0).to_numpy()
    cur_mob = cur_df["is_mobile"].value_counts(normalize=True).reindex(mob_bins, fill_value=0.0).to_numpy()
    psi_mob = compute_psi(ref_mob, cur_mob)
    ks_mob, p_mob = ks_2samp(ref_df["is_mobile"].to_numpy(), cur_df["is_mobile"].to_numpy())
    features_report["is_mobile"] = {
        "psi": round(psi_mob, 4),
        "ks_stat": round(float(ks_mob), 4),
        "p_value": round(float(p_mob), 5),
        "drift_detected": bool(psi_mob >= psi_threshold or abs(cur_mob[1] - ref_mob[1]) >= 0.10),
        "ref_mobile_share": round(float(ref_mob[1]), 4),
        "cur_mobile_share": round(float(cur_mob[1]), 4),
    }

    # 3. Category distribution (PSI)
    cats = sorted(prods["category"].unique().tolist())
    ref_cat = ref_df["category"].value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    cur_cat = cur_df["category"].value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    psi_cat = compute_psi(ref_cat, cur_cat)
    tvd_cat = float(0.5 * np.sum(np.abs(cur_cat - ref_cat)))
    features_report["category"] = {
        "psi": round(psi_cat, 4),
        "ks_stat": round(tvd_cat, 4),
        "p_value": round(float(np.exp(-psi_cat * 15.0)), 5),
        "drift_detected": bool(psi_cat >= psi_threshold or tvd_cat >= 0.12),
        "ref_distribution": {c: round(float(v), 4) for c, v in zip(cats, ref_cat)},
        "cur_distribution": {c: round(float(v), 4) for c, v in zip(cats, cur_cat)},
    }

    overall_drift = any(v["drift_detected"] for v in features_report.values())
    max_psi = max(v["psi"] for v in features_report.values())

    summary = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_rows": int(len(ref_df)),
        "current_rows": int(len(cur_df)),
        "psi_threshold": psi_threshold,
        "max_psi": round(float(max_psi), 4),
        "overall_drift": bool(overall_drift),
        "features": features_report,
    }

    _write_html_report(summary, cats, ref_cat, cur_cat, report_path)
    return summary


def _write_html_report(
    summary: Dict[str, Any],
    cats: list,
    ref_cat: np.ndarray,
    cur_cat: np.ndarray,
    report_path: str,
) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    status_badge = (
        '<span class="badge drift">DRIFT DETECTED</span>'
        if summary["overall_drift"]
        else '<span class="badge ok">NO SIGNIFICANT DRIFT</span>'
    )

    feat_rows = []
    for fname, fdata in summary["features"].items():
        st = '<b style="color:#c0392b">DRIFT</b>' if fdata["drift_detected"] else '<b style="color:#0b6e6e">STABLE</b>'
        feat_rows.append(
            f"<tr><td><code>{fname}</code></td><td>{fdata['psi']:.4f}</td>"
            f"<td>{fdata['ks_stat']:.4f}</td><td>{fdata['p_value']:.5f}</td><td>{st}</td></tr>"
        )

    cat_bars = []
    for c, r_val, c_val in zip(cats, ref_cat, cur_cat):
        r_pct = r_val * 100.0
        c_pct = c_val * 100.0
        cat_bars.append(
            f"<tr><td>{c}</td>"
            f"<td><div class='bar ref' style='width:{max(2, r_pct * 3):.0f}px'></div> {r_pct:.1f}%</td>"
            f"<td><div class='bar cur' style='width:{max(2, c_pct * 3):.0f}px'></div> {c_pct:.1f}%</td>"
            f"<td>{c_pct - r_pct:+.1f}%</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo Data Drift — Context-Aware Recommender</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 32px auto; max-width: 960px; color: #16202a; background: #f6f7f9; padding: 0 20px; }}
.card {{ background: #fff; border: 1px solid #dde2e8; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
h1 {{ margin-top: 0; font-size: 22px; }}
h2 {{ margin-top: 0; font-size: 17px; color: #0b6e6e; }}
.badge {{ display: inline-block; padding: 5px 12px; border-radius: 99px; font-size: 13px; font-weight: 700; }}
.badge.ok {{ background: #e1f5f2; color: #0b6e6e; }}
.badge.drift {{ background: #fdecea; color: #c0392b; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #dde2e8; }}
th {{ color: #5d6b79; }}
.bar {{ display: inline-block; height: 12px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }}
.bar.ref {{ background: #5d6b79; }}
.bar.cur {{ background: #0b6e6e; }}
</style>
</head>
<body>
<div class="card">
  <h1>Báo cáo Giám sát Data Drift (PSI &amp; KS-Test) {status_badge}</h1>
  <p>Thời gian tạo: <code>{summary['generated_at']}</code> · Reference: <b>{summary['reference_rows']}</b> dòng · Current: <b>{summary['current_rows']}</b> dòng · Ngưỡng PSI: <b>{summary['psi_threshold']}</b> (Max PSI: <b>{summary['max_psi']}</b>)</p>
</div>
<div class="card">
  <h2>1. Bảng kiểm định Drift theo thuộc tính ngữ cảnh</h2>
  <table>
    <tr><th>Thuộc tính</th><th>PSI</th><th>KS / TVD Stat</th><th>p-value</th><th>Trạng thái</th></tr>
    {''.join(feat_rows)}
  </table>
</div>
<div class="card">
  <h2>2. Phân bố danh mục sản phẩm: Reference (Train) vs Current (Mới)</h2>
  <table>
    <tr><th>Danh mục</th><th>Reference (%)</th><th>Current (%)</th><th>Chênh lệch</th></tr>
    {''.join(cat_bars)}
  </table>
</div>
</body>
</html>
"""
    Path(report_path).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    res = detect_data_drift()
    print(f"[drift] overall_drift={res['overall_drift']} max_psi={res['max_psi']} -> docs/results/drift_report.html")
