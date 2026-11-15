"""Shared model wrappers and training functions for Context-Aware Recommender.

Separating PyFunc model classes into `src/models.py` ensures that MLflow can
serialize and deserialize models cleanly across processes and containers when
logged with `code_paths=[MODELS_CODE_PATH]`.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import lightgbm as lgb
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

CAT_MAP: Dict[str, int] = {
    "Electronics": 0,
    "Fashion": 1,
    "Home": 2,
    "Accessories": 3,
    "Books": 4,
}

LGB_FEATURES: List[str] = [
    "user_id",
    "product_id",
    "hour",
    "is_weekend",
    "is_mobile",
    "time_slot",
    "cat_code",
    "price",
    "user_cat_ratio",
    "ctx_item_pop",
]

LGB_FEATURES_NO_CTX: List[str] = [
    "user_id",
    "product_id",
    "cat_code",
    "price",
    "user_cat_ratio",
]

MODELS_CODE_PATH: str = os.path.abspath(__file__)


def time_slot_from_hour(hour: int | np.ndarray | pd.Series):
    """Map hour (0..23) to time_slot: 0=Night(0-5), 1=Morning(6-11), 2=Afternoon(12-17), 3=Evening(18-23)."""
    return np.asarray(hour, dtype=np.int32) // 6


def extract_tabular_features(df: pd.DataFrame, include_context: bool = True) -> pd.DataFrame:
    """Extract numeric tabular features for tree-based models (LightGBM / XGBoost)."""
    x = pd.DataFrame(index=df.index)
    x["user_id"] = df["user_id"].astype("int32")
    x["product_id"] = df["product_id"].astype("int32")
    x["hour"] = df["hour"].astype("int32")
    x["is_weekend"] = df["is_weekend"].astype("int32")
    x["is_mobile"] = df["is_mobile"].astype("int32")
    x["time_slot"] = df["time_slot"].astype("int32")
    if "cat_code" in df.columns:
        x["cat_code"] = df["cat_code"].astype("int32")
    else:
        x["cat_code"] = df["category"].map(CAT_MAP).fillna(0).astype("int32")
    x["price"] = df["price"].astype("float32")
    x["user_cat_ratio"] = df["user_cat_ratio"].astype("float32")
    x["ctx_item_pop"] = df["ctx_item_pop"].astype("float32")
    cols = LGB_FEATURES if include_context else LGB_FEATURES_NO_CTX
    return x[cols]


class PopularityBaselineWrapper(mlflow.pyfunc.PythonModel):
    """Non-contextual global popularity baseline."""

    def __init__(self, item_pop: np.ndarray):
        self.item_pop = item_pop

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        pids = np.clip(model_input["product_id"].to_numpy(dtype=np.int64), 0, len(self.item_pop) - 1)
        return self.item_pop[pids].astype(np.float32)


def train_popularity_baseline(train_df: pd.DataFrame, products_df: pd.DataFrame) -> PopularityBaselineWrapper:
    max_pid = int(max(products_df["product_id"].max(), train_df["product_id"].max()))
    item_pop = np.zeros(max_pid + 1, dtype=np.float32)
    pos_df = train_df[train_df["label"] == 1]
    counts = pos_df["product_id"].value_counts()
    for pid, cnt in counts.items():
        item_pop[int(pid)] = float(cnt)
    if item_pop.max() > 0:
        item_pop = item_pop / item_pop.max()
    return PopularityBaselineWrapper(item_pop=item_pop)


class ContentBasedWrapper(mlflow.pyfunc.PythonModel):
    """TF-IDF + Category + Price cosine similarity recommender."""

    def __init__(self, user_profiles: np.ndarray, item_vectors: np.ndarray, default_profile: np.ndarray):
        self.user_profiles = user_profiles
        self.item_vectors = item_vectors
        self.default_profile = default_profile

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        uids = model_input["user_id"].to_numpy(dtype=np.int64)
        pids = np.clip(model_input["product_id"].to_numpy(dtype=np.int64), 0, len(self.item_vectors) - 1)

        valid_mask = (uids >= 0) & (uids < len(self.user_profiles))
        safe_uids = np.where(valid_mask, uids, 0)
        u_vecs = self.user_profiles[safe_uids].copy()
        if not np.all(valid_mask):
            u_vecs[~valid_mask] = self.default_profile
        # Replace zero vectors (cold-start users inside range) with default profile
        zero_rows = np.linalg.norm(u_vecs, axis=1) == 0
        if np.any(zero_rows):
            u_vecs[zero_rows] = self.default_profile

        return np.sum(u_vecs * self.item_vectors[pids], axis=1).astype(np.float32)


def train_content_based(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    max_features: int = 64,
    cat_weight: float = 1.5,
) -> ContentBasedWrapper:
    prods = products_df.sort_values("product_id").reset_index(drop=True)
    corpus = (prods["category"].astype(str) + " " + prods["title"].astype(str)).tolist()
    tfidf = TfidfVectorizer(max_features=max_features).fit_transform(corpus).toarray().astype(np.float32)

    cat_oh = (
        pd.get_dummies(prods["category"])
        .reindex(columns=list(CAT_MAP.keys()), fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    max_price = max(float(prods["price"].max()), 1.0)
    price_norm = (prods["price"].to_numpy(dtype=np.float32) / max_price).reshape(-1, 1)

    norm_items = normalize(np.hstack([tfidf, cat_oh * float(cat_weight), price_norm])).astype(np.float32)

    max_pid = int(prods["product_id"].max())
    item_vectors = np.zeros((max_pid + 1, norm_items.shape[1]), dtype=np.float32)
    for row_idx, pid in enumerate(prods["product_id"]):
        item_vectors[int(pid)] = norm_items[row_idx]

    pos_df = train_df[train_df["label"] == 1]
    max_uid = int(train_df["user_id"].max())
    user_profiles = np.zeros((max_uid + 1, norm_items.shape[1]), dtype=np.float32)

    for uid, grp in pos_df.groupby("user_id"):
        vecs = item_vectors[grp["product_id"].to_numpy(dtype=np.int64)]
        user_profiles[int(uid)] = normalize(vecs.mean(axis=0, keepdims=True))[0]

    default_profile = normalize(item_vectors[1:].mean(axis=0, keepdims=True))[0].astype(np.float32)
    return ContentBasedWrapper(user_profiles, item_vectors, default_profile)


class MatrixFactorizationWrapper(mlflow.pyfunc.PythonModel):
    """Collaborative Filtering via TruncatedSVD latent factors."""

    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray, default_user_factor: np.ndarray):
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.default_user_factor = default_user_factor

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        uids = model_input["user_id"].to_numpy(dtype=np.int64)
        pids = np.clip(model_input["product_id"].to_numpy(dtype=np.int64), 0, len(self.item_factors) - 1)

        valid_mask = (uids >= 0) & (uids < len(self.user_factors))
        safe_uids = np.where(valid_mask, uids, 0)
        u_fac = self.user_factors[safe_uids].copy()
        if not np.all(valid_mask):
            u_fac[~valid_mask] = self.default_user_factor

        return np.sum(u_fac * self.item_factors[pids], axis=1).astype(np.float32)


def train_matrix_factorization(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    n_components: int = 24,
    seed: int = 42,
) -> MatrixFactorizationWrapper:
    max_uid = int(train_df["user_id"].max())
    max_pid = int(products_df["product_id"].max())

    mat = np.zeros((max_uid + 1, max_pid + 1), dtype=np.float32)
    pos_df = train_df[train_df["label"] == 1]
    for r in pos_df.itertuples(index=False):
        mat[int(r.user_id), int(r.product_id)] += 1.0

    n_comp = min(n_components, min(mat.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    user_factors = svd.fit_transform(mat).astype(np.float32)
    item_factors = svd.components_.T.astype(np.float32)
    default_user_factor = user_factors[1:].mean(axis=0).astype(np.float32)
    return MatrixFactorizationWrapper(user_factors, item_factors, default_user_factor)


class LightGBMWrapper(mlflow.pyfunc.PythonModel):
    """Context-Aware LightGBM GBDT classifier wrapper."""

    def __init__(self, booster, include_context: bool = True):
        self.booster = booster
        self.include_context = include_context

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        feats = extract_tabular_features(model_input, include_context=self.include_context)
        return self.booster.predict_proba(feats)[:, 1].astype(np.float32)


def train_lightgbm(
    train_df: pd.DataFrame,
    n_estimators: int = 180,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
    include_context: bool = True,
) -> LightGBMWrapper:
    x_train = extract_tabular_features(train_df, include_context=include_context)
    y_train = train_df["label"].to_numpy(dtype=np.int32)

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample_for_bin=50000,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(x_train, y_train)
    return LightGBMWrapper(clf, include_context=include_context)


class XGBoostWrapper(mlflow.pyfunc.PythonModel):
    """Context-Aware XGBoost classifier wrapper."""

    def __init__(self, model, include_context: bool = True):
        self.model = model
        self.include_context = include_context

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        feats = extract_tabular_features(model_input, include_context=self.include_context)
        return self.model.predict_proba(feats)[:, 1].astype(np.float32)


def train_xgboost(
    train_df: pd.DataFrame,
    n_estimators: int = 180,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    seed: int = 42,
    include_context: bool = True,
) -> XGBoostWrapper:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    x_train = extract_tabular_features(train_df, include_context=include_context)
    y_train = train_df["label"].to_numpy(dtype=np.int32)

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(x_train, y_train)
    return XGBoostWrapper(clf, include_context=include_context)


class ContextNCFNet(nn.Module):
    """Neural Collaborative Filtering fused with contextual dense features."""

    def __init__(self, num_users: int, num_items: int, num_cats: int = 5, emb_dim: int = 16):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, emb_dim)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim)
        self.cat_emb = nn.Embedding(num_cats, 4)
        in_dim = emb_dim * 2 + 4 + 7
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, u_idx, i_idx, c_idx, num_feats):
        x = torch.cat([self.user_emb(u_idx), self.item_emb(i_idx), self.cat_emb(c_idx), num_feats], dim=1)
        return self.mlp(x).squeeze(-1)


class NCFWrapper(mlflow.pyfunc.PythonModel):
    """PyTorch CPU Context-NCF wrapper for MLflow."""

    def __init__(
        self,
        state_dict: dict,
        num_users: int,
        num_items: int,
        emb_dim: int = 16,
        history: Optional[Dict[str, List[float]]] = None,
    ):
        self.state_dict = state_dict
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.history = history or {"train_loss": [], "val_loss": []}
        self._net = None

    def _get_net(self):
        if self._net is None:
            net = ContextNCFNet(self.num_users, self.num_items, emb_dim=self.emb_dim)
            net.load_state_dict(self.state_dict)
            net.eval()
            self._net = net
        return self._net

    def _prepare_tensors(self, df: pd.DataFrame):
        u_raw = df["user_id"].to_numpy(dtype=np.int64)
        i_raw = df["product_id"].to_numpy(dtype=np.int64)
        u = torch.tensor(np.clip(u_raw, 0, self.num_users), dtype=torch.long)
        i = torch.tensor(np.clip(i_raw, 0, self.num_items), dtype=torch.long)
        if "cat_code" in df.columns:
            c_arr = df["cat_code"].to_numpy(dtype=np.int64)
        else:
            c_arr = df["category"].map(CAT_MAP).fillna(0).to_numpy(dtype=np.int64)
        c = torch.tensor(np.clip(c_arr, 0, len(CAT_MAP) - 1), dtype=torch.long)
        num = np.column_stack([
            df["hour"].to_numpy(dtype=np.float32) / 23.0,
            df["is_weekend"].to_numpy(dtype=np.float32),
            df["is_mobile"].to_numpy(dtype=np.float32),
            df["time_slot"].to_numpy(dtype=np.float32) / 3.0,
            df["price"].to_numpy(dtype=np.float32) / 1200.0,
            df["user_cat_ratio"].to_numpy(dtype=np.float32),
            df["ctx_item_pop"].to_numpy(dtype=np.float32),
        ]).astype(np.float32)
        return u, i, c, torch.tensor(num, dtype=torch.float32)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        net = self._get_net()
        u, i, c, num = self._prepare_tensors(model_input)
        with torch.no_grad():
            return torch.sigmoid(net(u, i, c, num)).numpy().astype(np.float32)


def train_ncf(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    emb_dim: int = 16,
    epochs: int = 12,
    lr: float = 0.003,
    patience: int = 3,
    seed: int = 42,
) -> NCFWrapper:
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_users = int(train_df["user_id"].max())
    num_items = int(products_df["product_id"].max())

    if val_df is None:
        # Hold out last 10% of training rows for early stopping validation loss
        split_idx = int(len(train_df) * 0.9)
        tr_sub = train_df.iloc[:split_idx]
        va_sub = train_df.iloc[split_idx:]
    else:
        tr_sub = train_df
        va_sub = val_df

    net = ContextNCFNet(num_users, num_items, emb_dim=emb_dim)
    helper = NCFWrapper({}, num_users, num_items, emb_dim=emb_dim)
    u, i, c, num = helper._prepare_tensors(tr_sub)
    y = torch.tensor(tr_sub["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    vu, vi, vc, vnum = helper._prepare_tensors(va_sub)
    vy = torch.tensor(va_sub["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(TensorDataset(u, i, c, num, y), batch_size=1024, shuffle=True, generator=gen)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    no_improve = 0

    for _epoch in range(epochs):
        net.train()
        running_loss = 0.0
        n_batches = 0
        for bu, bi, bc, bnum, by in loader:
            optimizer.zero_grad()
            loss = criterion(net(bu, bi, bc, bnum), by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        avg_tr = running_loss / max(1, n_batches)
        net.eval()
        with torch.no_grad():
            val_loss = float(criterion(net(vu, vi, vc, vnum), vy).item())

        history["train_loss"].append(round(avg_tr, 5))
        history["val_loss"].append(round(val_loss, 5))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return NCFWrapper(best_state, num_users, num_items, emb_dim=emb_dim, history=history)
