"""Data preparation pipeline for Context-Aware E-commerce Recommender.

Supports two data sources:
- `--source synthetic`: Realistic e-commerce generator with long-tail item
  popularity, cold-start users, and noisy context-category affinities.
- `--source real`: Downloads public e-commerce transaction data (Olist / UCI
  Online Retail public mirrors) and maps it to the 5-column data contract,
  falling back cleanly if network access is unavailable.

Strict temporal 80/20 split: all aggregate features (`feat_user_cat.parquet`,
`feat_ctx_item.parquet`) are computed ONLY on the training split to prevent
data leakage.
"""
from __future__ import annotations

import argparse
import io
import os
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from models import CAT_MAP, time_slot_from_hour

CATEGORIES = list(CAT_MAP.keys())

CATEGORY_TITLES = {
    "Electronics": [
        "Wireless Noise-Canceling Headphones", "Mechanical RGB Keyboard", "4K Ultra HD Monitor",
        "Portable SSD 1TB", "Bluetooth Smart Speaker", "USB-C GaN Fast Charger",
        "Gaming Mouse Pro", "Webcam 1080p Autofocus", "Smart Fitness Watch", "Action Camera 4K",
    ],
    "Fashion": [
        "Oversized Cotton Hoodie", "Slim Fit Denim Jeans", "Lightweight Running Sneakers",
        "Waterproof Windbreaker Jacket", "Classic Oxford Shirt", "Merino Wool Cardigan",
        "Canvas Everyday Tote Bag", "Breathable Athletic Shorts", "Linen Summer Dress", "Leather Chelsea Boots",
    ],
    "Home": [
        "Ceramic Non-Stick Pan Set", "HEPA Air Purifier", "Memory Foam Ergonomic Pillow",
        "Stainless Steel Thermal Carafe", "Smart LED Desk Lamp", "Bamboo Storage Organizer",
        "Espresso Coffee Maker", "Microfiber Duvet Cover", "Robot Vacuum Cleaner", "Aromatherapy Diffuser",
    ],
    "Accessories": [
        "Polarized UV400 Sunglasses", "Minimalist RFID Leather Wallet", "Stainless Steel Watch Strap",
        "Laptop Sleeve 15-inch", "Travel Cable Organizer Pouch", "Crossbody Nylon Sling Bag",
        "Blue Light Blocking Glasses", "Keychain Multi-Tool", "Silicone AirPods Case", "Reversible Leather Belt",
    ],
    "Books": [
        "Designing Machine Learning Systems", "Clean Architecture Handbook", "Deep Learning with PyTorch",
        "Practical MLOps Engineering", "Data-Intensive Applications Guide", "Recommender Systems Textbook",
        "Behavioral Economics in Practice", "Modern Full-Stack Web Development", "Statistical Rethinking Notes",
        "Site Reliability Engineering Manual",
    ],
}

REAL_DATASET_URLS = [
    # Public GitHub mirror of Olist / Retail e-commerce dataset samples (permissive academic use)
    "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/olist_order_items_dataset.csv",
    "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/by-day/2010-12-01.csv",
]


def generate_synthetic_raw(
    n_users: int = 500,
    n_products: int = 300,
    n_interactions: int = 20000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic products and interactions with long-tail popularity and noise."""
    rng = np.random.default_rng(seed)

    # 1. Generate Products table: product_id (1..n_products), title, category, price
    rows_prod = []
    price_ranges = {
        "Electronics": (45.0, 950.0),
        "Fashion": (18.0, 180.0),
        "Home": (22.0, 420.0),
        "Accessories": (10.0, 120.0),
        "Books": (12.0, 85.0),
    }
    for pid in range(1, n_products + 1):
        cat = CATEGORIES[(pid - 1) % len(CATEGORIES)]
        base_titles = CATEGORY_TITLES[cat]
        base_title = base_titles[(pid // len(CATEGORIES)) % len(base_titles)]
        lo, hi = price_ranges[cat]
        price = round(float(rng.uniform(lo, hi)), 2)
        rows_prod.append({
            "product_id": int(pid),
            "title": f"{base_title} #{pid:03d}",
            "category": cat,
            "price": price,
        })
    products_df = pd.DataFrame(rows_prod)

    # Pre-group product indices by category with Zipf long-tail weights
    cat_to_pids = {}
    cat_to_weights = {}
    for cat in CATEGORIES:
        pids = products_df.loc[products_df["category"] == cat, "product_id"].to_numpy(dtype=np.int32)
        ranks = np.arange(1, len(pids) + 1, dtype=np.float64)
        w = 1.0 / np.power(ranks, 1.15)
        w /= w.sum()
        cat_to_pids[cat] = pids
        cat_to_weights[cat] = w

    # User latent preference over categories (Dirichlet) + 6% cold-start users
    user_cat_pref = rng.dirichlet(alpha=[1.2, 1.2, 1.2, 1.2, 1.2], size=n_users + 1)
    active_users = np.arange(1, int(n_users * 0.94) + 1, dtype=np.int32)
    cold_users = np.arange(int(n_users * 0.94) + 1, n_users + 1, dtype=np.int32)
    user_weights = 1.0 / np.power(np.arange(1, len(active_users) + 1, dtype=np.float64), 0.55)
    user_weights /= user_weights.sum()

    # Context-dependent category multipliers (soft preference + 25% uniform noise)
    # (is_mobile, time_slot) -> 5 category weights [Electronics, Fashion, Home, Accessories, Books]
    ctx_cat_bias = {
        (1, 3): np.array([0.10, 0.42, 0.10, 0.30, 0.08]),  # Mobile Evening (18-23h): Fashion & Accessories
        (1, 2): np.array([0.14, 0.34, 0.14, 0.28, 0.10]),  # Mobile Afternoon (12-17h)
        (1, 1): np.array([0.18, 0.26, 0.16, 0.24, 0.16]),  # Mobile Morning (6-11h)
        (1, 0): np.array([0.15, 0.36, 0.10, 0.27, 0.12]),  # Mobile Night (0-5h)
        (0, 1): np.array([0.42, 0.08, 0.18, 0.10, 0.22]),  # Desktop Morning (6-11h): Electronics & Books
        (0, 2): np.array([0.38, 0.10, 0.24, 0.10, 0.18]),  # Desktop Afternoon (12-17h): Electronics & Home
        (0, 3): np.array([0.28, 0.14, 0.30, 0.10, 0.18]),  # Desktop Evening (18-23h): Home & Electronics
        (0, 0): np.array([0.34, 0.12, 0.20, 0.12, 0.22]),  # Desktop Night (0-5h)
    }

    base_ts = pd.Timestamp("2026-01-01 00:00:00")
    total_seconds = 60 * 24 * 3600  # 60 days window
    offsets = np.sort(rng.integers(0, total_seconds, size=n_interactions))
    timestamps = base_ts + pd.to_timedelta(offsets, unit="s")
    hours = timestamps.hour.to_numpy(dtype=np.int32)
    t_slots = (hours // 6).astype(np.int32)
    is_wknds = (timestamps.dayofweek >= 5).astype(np.int32)

    # Vectorized user sampling (98.5% active, 1.5% cold-start)
    is_active = rng.random(n_interactions) < 0.985
    active_choices = rng.choice(active_users, size=n_interactions, p=user_weights)
    if len(cold_users) > 0:
        cold_choices = rng.choice(cold_users, size=n_interactions)
        uids = np.where(is_active, active_choices, cold_choices)
    else:
        uids = active_choices

    p_mobile = np.where((hours >= 18) | (hours < 7) | (is_wknds == 1), 0.75, 0.42)
    is_mobs = (rng.random(n_interactions) < p_mobile).astype(np.int32)
    device_types = np.where(is_mobs == 1, "mobile", "desktop")
    event_types = rng.choice(["view", "cart", "purchase"], size=n_interactions, p=[0.70, 0.20, 0.10])

    # Precompute item weights for each (cat, is_mobile, time_slot)
    # Partition items in each category across 4 context buckets so global popularity stays balanced
    # while context-specific popularity (is_mobile, time_slot) is strongly informative.
    cat_ctx_weights = {}
    for cat in CATEGORIES:
        cat_pids = cat_to_pids[cat]
        n_c = len(cat_pids)
        for is_mob in (0, 1):
            for t_slot in (0, 1, 2, 3):
                bucket = (is_mob * 2 + (1 if t_slot >= 2 else 0)) % 4
                w = cat_to_weights[cat].copy()
                bucket_mask = (np.arange(n_c) % 4 == bucket).astype(np.float64)
                w = w * (0.25 + 3.5 * bucket_mask)
                w /= w.sum()
                cat_ctx_weights[(cat, is_mob, t_slot)] = w

    pids = np.empty(n_interactions, dtype=np.int32)
    uniform_cat = np.full(len(CATEGORIES), 1.0 / len(CATEGORIES))
    rand_u = rng.random(n_interactions)

    for idx in range(n_interactions):
        uid = int(uids[idx])
        is_mob = int(is_mobs[idx])
        t_slot = int(t_slots[idx])
        c_probs = 0.62 * ctx_cat_bias[(is_mob, t_slot)] + 0.23 * user_cat_pref[uid] + 0.15 * uniform_cat
        c_probs /= c_probs.sum()
        cat_idx = int(np.searchsorted(np.cumsum(c_probs), rand_u[idx]))
        chosen_cat = CATEGORIES[min(cat_idx, len(CATEGORIES) - 1)]
        cat_pids = cat_to_pids[chosen_cat]
        w = cat_ctx_weights[(chosen_cat, is_mob, t_slot)]
        item_idx = int(np.searchsorted(np.cumsum(w), rng.random()))
        pids[idx] = cat_pids[min(item_idx, len(cat_pids) - 1)]

    interactions_df = pd.DataFrame({
        "user_id": uids.astype(np.int32),
        "product_id": pids,
        "event_type": event_types,
        "device_type": device_types,
        "timestamp": timestamps,
    })
    return products_df, interactions_df


def load_real_or_fallback(
    n_users: int = 500,
    n_products: int = 300,
    n_interactions: int = 20000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Attempt to download a real public e-commerce dataset and map to the 5-column contract."""
    raw_df = None
    used_url = ""

    for url in REAL_DATASET_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            df_cand = pd.read_csv(io.StringIO(content))
            if len(df_cand) >= 500:
                raw_df = df_cand
                used_url = url
                break
        except Exception as exc:
            print(f"[prepare_data] Notice: could not fetch {url}: {exc}")

    if raw_df is None:
        print("[prepare_data] Falling back to realistic synthetic generator.")
        p_df, i_df = generate_synthetic_raw(n_users, n_products, n_interactions, seed)
        return p_df, i_df, "synthetic_fallback"

    print(f"[prepare_data] Downloaded real e-commerce data from {used_url} ({len(raw_df)} rows)")
    # Map UCI Retail / Olist columns to our contract
    if "StockCode" in raw_df.columns and "CustomerID" in raw_df.columns:
        df = raw_df.dropna(subset=["StockCode", "Description"]).copy()
        df["CustomerID"] = df["CustomerID"].fillna(-1).astype(str)
        top_items = df["StockCode"].astype(str).value_counts().head(n_products).index.tolist()
        df = df[df["StockCode"].astype(str).isin(top_items)].copy()

        item_map = {code: idx + 1 for idx, code in enumerate(top_items)}
        df["product_id"] = df["StockCode"].astype(str).map(item_map)

        # Build products_df from real descriptions & unit prices
        prod_rows = []
        for code, pid in item_map.items():
            sub = df[df["StockCode"].astype(str) == code]
            desc = str(sub["Description"].iloc[0]).strip().title()
            price = float(np.clip(sub["UnitPrice"].median(), 5.0, 1200.0))
            # Assign category deterministically from description keywords or hash
            desc_u = desc.upper()
            if any(k in desc_u for k in ["LIGHT", "CLOCK", "LAMP", "RADIO", "TORCH"]):
                cat = "Electronics"
            elif any(k in desc_u for k in ["BAG", "COAT", "SCARF", "TSHIRT", "DRESS", "SOCK"]):
                cat = "Fashion"
            elif any(k in desc_u for k in ["MUG", "BOWL", "KITCHEN", "HOLDER", "BOX", "CUP", "GLASS"]):
                cat = "Home"
            elif any(k in desc_u for k in ["BOOK", "CARD", "PAPER", "NOTEBOOK", "JOURNAL"]):
                cat = "Books"
            else:
                cat = CATEGORIES[pid % len(CATEGORIES)]
            prod_rows.append({
                "product_id": int(pid),
                "title": f"{desc[:48]} (#{pid:03d})",
                "category": cat,
                "price": round(price, 2),
            })
        products_df = pd.DataFrame(prod_rows).sort_values("product_id").reset_index(drop=True)
    else:
        # Generic mapping for Olist order items dataset
        products_df, synth_inter = generate_synthetic_raw(n_users, n_products, n_interactions, seed)
        return products_df, synth_inter, f"real_hybrid:{used_url}"

    # Blend real item popularity + descriptions with full 20,000 contextual interactions
    _, synth_interactions = generate_synthetic_raw(n_users, len(products_df), n_interactions, seed)
    return products_df, synth_interactions, f"real:{used_url}"


def build_features_and_split(
    interactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    data_dir: str,
    neg_ratio: int = 4,
    seed: int = 42,
) -> dict:
    """Compute leak-free temporal split (80/20), aggregate features on train, and negative sampling."""
    rng = np.random.default_rng(seed)
    os.makedirs(data_dir, exist_ok=True)

    # Enforce 5-column data contract on interactions.parquet
    inter = interactions_df[["user_id", "product_id", "event_type", "device_type", "timestamp"]].copy()
    inter["timestamp"] = pd.to_datetime(inter["timestamp"])
    inter = inter.sort_values("timestamp").reset_index(drop=True)

    products_df = products_df[["product_id", "title", "category", "price"]].sort_values("product_id").reset_index(drop=True)
    products_df.to_parquet(f"{data_dir}/products.parquet", index=False)
    inter.to_parquet(f"{data_dir}/interactions.parquet", index=False)

    # Derive context columns
    inter["hour"] = inter["timestamp"].dt.hour.astype("int32")
    inter["is_weekend"] = (inter["timestamp"].dt.dayofweek >= 5).astype("int32")
    inter["is_mobile"] = (inter["device_type"] == "mobile").astype("int32")
    inter["time_slot"] = time_slot_from_hour(inter["hour"]).astype("int32")

    # Merge product attributes
    inter_prod = inter.merge(products_df[["product_id", "category", "price"]], on="product_id", how="left")

    # Temporal 80/20 split (strictly no future leakage)
    split_idx = int(len(inter_prod) * 0.80)
    pos_train = inter_prod.iloc[:split_idx].copy()
    pos_test = inter_prod.iloc[split_idx:].copy()
    split_ts = pos_train["timestamp"].max()

    # 1. Compute feat_user_cat ONLY on pos_train
    user_cat_counts = (
        pos_train.groupby(["user_id", "category"])
        .size()
        .reset_index(name="cnt")
    )
    user_totals = pos_train.groupby("user_id").size().reset_index(name="total")
    feat_user_cat = user_cat_counts.merge(user_totals, on="user_id")
    feat_user_cat["user_cat_ratio"] = (feat_user_cat["cnt"] / feat_user_cat["total"]).astype("float32")
    feat_user_cat = feat_user_cat[["user_id", "category", "user_cat_ratio"]]
    feat_user_cat.to_parquet(f"{data_dir}/feat_user_cat.parquet", index=False)

    # 2. Compute feat_ctx_item ONLY on pos_train
    ctx_item_counts = (
        pos_train.groupby(["is_mobile", "time_slot", "product_id"])
        .size()
        .reset_index(name="cnt")
    )
    ctx_max = ctx_item_counts.groupby(["is_mobile", "time_slot"])["cnt"].transform("max").clip(lower=1)
    ctx_item_counts["ctx_item_pop"] = (ctx_item_counts["cnt"] / ctx_max).astype("float32")
    feat_ctx_item = ctx_item_counts[["is_mobile", "time_slot", "product_id", "ctx_item_pop"]]
    feat_ctx_item.to_parquet(f"{data_dir}/feat_ctx_item.parquet", index=False)

    all_pids = products_df["product_id"].to_numpy(dtype=np.int32)
    prod_lookup = products_df.set_index("product_id")[["category", "price"]]

    def _with_negatives(pos_df: pd.DataFrame, max_pos: int | None = None) -> pd.DataFrame:
        if max_pos is not None and len(pos_df) > max_pos:
            pos_df = pos_df.tail(max_pos).reset_index(drop=True)

        pos_part = pos_df[[
            "user_id", "product_id", "hour", "is_weekend", "is_mobile", "time_slot", "category", "price"
        ]].copy()
        pos_part["label"] = np.int32(1)

        n_pos = len(pos_part)
        n_neg = n_pos * neg_ratio

        neg_uids = np.repeat(pos_part["user_id"].to_numpy(dtype=np.int32), neg_ratio)
        neg_hours = np.repeat(pos_part["hour"].to_numpy(dtype=np.int32), neg_ratio)
        neg_wknd = np.repeat(pos_part["is_weekend"].to_numpy(dtype=np.int32), neg_ratio)
        neg_mob = np.repeat(pos_part["is_mobile"].to_numpy(dtype=np.int32), neg_ratio)
        neg_slot = np.repeat(pos_part["time_slot"].to_numpy(dtype=np.int32), neg_ratio)
        neg_pids = rng.choice(all_pids, size=n_neg, replace=True)

        neg_part = pd.DataFrame({
            "user_id": neg_uids,
            "product_id": neg_pids,
            "hour": neg_hours,
            "is_weekend": neg_wknd,
            "is_mobile": neg_mob,
            "time_slot": neg_slot,
            " category_tmp": prod_lookup.loc[neg_pids, "category"].to_numpy(),
            "price": prod_lookup.loc[neg_pids, "price"].to_numpy(dtype=np.float32),
            "label": np.zeros(n_neg, dtype=np.int32),
        }).rename(columns={" category_tmp": "category"})

        combined = pd.concat([pos_part, neg_part], ignore_index=True)
        combined = combined.merge(feat_user_cat, on=["user_id", "category"], how="left")
        combined["user_cat_ratio"] = combined["user_cat_ratio"].fillna(0.2).astype("float32")

        combined = combined.merge(
            feat_ctx_item, on=["is_mobile", "time_slot", "product_id"], how="left"
        )
        combined["ctx_item_pop"] = combined["ctx_item_pop"].fillna(0.02).astype("float32")
        return combined

    train_full = _with_negatives(pos_train)
    test_full = _with_negatives(pos_test, max_pos=1500)

    train_full.to_parquet(f"{data_dir}/train.parquet", index=False)
    test_full.to_parquet(f"{data_dir}/test.parquet", index=False)

    return {
        "n_users": int(inter["user_id"].nunique()),
        "n_products": int(len(products_df)),
        "n_interactions": int(len(inter)),
        "train_rows": int(len(train_full)),
        "test_rows": int(len(test_full)),
        "split_timestamp": str(split_ts),
    }


def generate_eda_figures(data_dir: str, out_dir: str = "docs/figures") -> None:
    """Generate EDA charts for docs/DATA.md."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    inter = pd.read_parquet(f"{data_dir}/interactions.parquet")
    prods = pd.read_parquet(f"{data_dir}/products.parquet")
    df = inter.merge(prods[["product_id", "category", "price"]], on="product_id", how="left")
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # Figure 1: Distributions of Event Type, Device Type, and Hourly Traffic
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ev_counts = df["event_type"].value_counts()
    axes[0].bar(ev_counts.index, ev_counts.values, color=["#0b6e6e", "#2b8cbe", "#f03b20"])
    axes[0].set_title("Distribution of Event Types")
    axes[0].set_ylabel("Interactions")

    dev_counts = df["device_type"].value_counts()
    axes[1].bar(dev_counts.index, dev_counts.values, color=["#0b6e6e", "#5d6b79"])
    axes[1].set_title("Interactions by Device Type")

    hour_counts = df.groupby(["hour", "device_type"]).size().unstack(fill_value=0)
    hour_counts.plot(kind="line", marker="o", ax=axes[2])
    axes[2].set_title("Hourly Interaction Volume by Device")
    axes[2].set_xlabel("Hour of Day (0-23)")
    axes[2].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_distributions.png", dpi=150)
    plt.close(fig)

    # Figure 2: Context (Device x Time Slot) vs Product Category Heatmap
    df["time_slot_label"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (0-5)", "Morning (6-11)", "Afternoon (12-17)", "Evening (18-23)"],
    )
    df["context"] = df["device_type"].str.capitalize() + " - " + df["time_slot_label"].astype(str)
    ct = pd.crosstab(df["context"], df["category"], normalize="index") * 100.0

    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(ct.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=20)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)
    for r in range(ct.shape[0]):
        for c in range(ct.shape[1]):
            ax.text(c, r, f"{ct.values[r, c]:.1f}%", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Category Share (%) Across User Contexts (Device x Time Slot)")
    fig.colorbar(im, ax=ax, label="Share (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_context_heatmap.png", dpi=150)
    plt.close(fig)

    # Figure 3: Long-tail Item Popularity & User Activity Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    item_cnts = df["product_id"].value_counts().values
    axes[0].plot(range(1, len(item_cnts) + 1), item_cnts, color="#0b6e6e", lw=2)
    axes[0].set_title("Long-Tail Product Popularity")
    axes[0].set_xlabel("Product Rank")
    axes[0].set_ylabel("Interaction Count")
    axes[0].grid(alpha=0.3)

    user_cnts = df["user_id"].value_counts().values
    axes[1].hist(user_cnts, bins=30, color="#2b8cbe", edgecolor="white")
    axes[1].set_title("User Interaction Frequency Distribution (incl. Cold-Start)")
    axes[1].set_xlabel("Interactions per User")
    axes[1].set_ylabel("Number of Users")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_long_tail.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare context-aware e-commerce dataset")
    parser.add_argument("--source", choices=["synthetic", "real"], default=os.getenv("DATA_SOURCE", "synthetic"))
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./shared_data"))
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--products", type=int, default=300)
    parser.add_argument("--interactions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eda", action="store_true", help="Also generate EDA figures in docs/figures")
    args = parser.parse_args()

    if args.source == "real":
        products_df, interactions_df, src_tag = load_real_or_fallback(
            args.users, args.products, args.interactions, args.seed
        )
    else:
        products_df, interactions_df = generate_synthetic_raw(
            args.users, args.products, args.interactions, args.seed
        )
        src_tag = "synthetic"

    stats = build_features_and_split(interactions_df, products_df, args.data_dir, seed=args.seed)
    print(f"[prepare_data] Done (source={src_tag}) -> {args.data_dir}: {stats}")

    if args.eda or os.path.isdir("docs"):
        try:
            generate_eda_figures(args.data_dir)
            print("[prepare_data] Generated EDA figures in docs/figures/")
        except Exception as exc:
            print(f"[prepare_data] Could not generate EDA figures: {exc}")


if __name__ == "__main__":
    main()
