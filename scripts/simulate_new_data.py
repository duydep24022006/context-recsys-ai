"""Simulate incoming user interactions (with optional concept/data drift) for retraining demos."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Append simulated new interactions to interactions.parquet")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./shared_data"))
    parser.add_argument("--rows", type=int, default=2500, help="Number of new interactions to append")
    parser.add_argument("--drift", action="store_true", help="Inject strong mobile late-night Electronics/Books drift")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    inter_path = os.path.join(args.data_dir, "interactions.parquet")
    prod_path = os.path.join(args.data_dir, "products.parquet")
    inter = pd.read_parquet(inter_path)
    prods = pd.read_parquet(prod_path)

    rng = np.random.default_rng(args.seed)
    ts_series = pd.to_datetime(inter["timestamp"]).sort_values().reset_index(drop=True)
    split_ts = ts_series.iloc[int(len(ts_series) * 0.78)]
    max_ts = ts_series.max()
    uids = inter["user_id"].unique()

    if args.drift:
        # Shift toward Mobile evening (18-23h) Fashion & Accessories items matching context bucket
        sale_pids = prods.loc[prods["category"].isin(["Fashion", "Accessories"]), "product_id"].to_numpy()
        # Favor context bucket (is_mob=1, t_slot=3 -> bucket=(2+1)%4=3)
        sale_pids = sale_pids[np.arange(len(sale_pids)) % 4 == 3]
        other_pids = prods["product_id"].to_numpy()
    else:
        sale_pids = prods["product_id"].to_numpy()
        other_pids = sale_pids

    new_records = []
    for i in range(args.rows):
        # Place 75% of new interactions in the recent training window and 25% in the test window
        if i < int(args.rows * 0.75):
            base_t = split_ts - pd.Timedelta(hours=int(rng.integers(1, 72)))
        else:
            base_t = max_ts + pd.Timedelta(seconds=int((i + 1) * rng.integers(10, 45)))

        if args.drift:
            drifted_hour = int(rng.choice([19, 20, 21, 22, 23], p=[0.15, 0.25, 0.25, 0.20, 0.15]))
            ts = base_t.replace(hour=drifted_hour)
            device = str(rng.choice(["mobile", "desktop"], p=[0.90, 0.10]))
            pid = int(rng.choice(sale_pids if rng.random() < 0.82 else other_pids))
            ev = str(rng.choice(["view", "cart", "purchase"], p=[0.55, 0.28, 0.17]))
        else:
            ts = base_t
            device = str(rng.choice(["mobile", "desktop"], p=[0.62, 0.38]))
            pid = int(rng.choice(other_pids))
            ev = str(rng.choice(["view", "cart", "purchase"], p=[0.70, 0.20, 0.10]))

        new_records.append({
            "user_id": int(rng.choice(uids)),
            "product_id": pid,
            "event_type": ev,
            "device_type": device,
            "timestamp": ts,
        })

    new_df = pd.DataFrame(new_records)
    updated = pd.concat([inter, new_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    updated.to_parquet(inter_path, index=False)
    print(
        f"[simulate_new_data] Appended {len(new_df)} interactions (drift={args.drift}). "
        f"Total interactions: {len(updated)}"
    )


if __name__ == "__main__":
    main()
