import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from metrics import evaluate_model

import os
DATA_DIR = os.getenv("DATA_DIR", "./shared_data")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "Context_Aware_Recommender_CLO1"
REGISTERED_MODEL_NAME = "ContextAwareRecommender"

CAT_MAP = {"Electronics": 0, "Fashion": 1, "Home": 2, "Accessories": 3, "Books": 4}
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class ContentBasedWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, user_profiles: np.ndarray, item_vectors: np.ndarray):
        self.user_profiles = user_profiles
        self.item_vectors = item_vectors

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        uids = model_input["user_id"].to_numpy(dtype=np.int64)
        pids = model_input["product_id"].to_numpy(dtype=np.int64)
        return np.sum(self.user_profiles[uids] * self.item_vectors[pids], axis=1)


def train_content_based(train_df: pd.DataFrame, products_df: pd.DataFrame) -> ContentBasedWrapper:
    prods = products_df.sort_values("product_id").reset_index(drop=True)
    corpus = (prods["category"] + " " + prods["title"]).tolist()
    tfidf = TfidfVectorizer(max_features=64).fit_transform(corpus).toarray()

    cat_oh = pd.get_dummies(prods["category"]).reindex(columns=list(CAT_MAP.keys()), fill_value=0).to_numpy(dtype=np.float32)
    price_norm = (prods["price"].to_numpy(dtype=np.float32) / 1200.0).reshape(-1, 1)

    norm_items = normalize(np.hstack([tfidf, cat_oh * 1.5, price_norm]))

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

    return ContentBasedWrapper(user_profiles, item_vectors)


class MatrixFactorizationWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray):
        self.user_factors = user_factors
        self.item_factors = item_factors

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        uids = model_input["user_id"].to_numpy(dtype=np.int64)
        pids = model_input["product_id"].to_numpy(dtype=np.int64)
        return np.sum(self.user_factors[uids] * self.item_factors[pids], axis=1)


def train_matrix_factorization(train_df: pd.DataFrame, products_df: pd.DataFrame) -> MatrixFactorizationWrapper:
    max_uid = int(train_df["user_id"].max())
    max_pid = int(products_df["product_id"].max())

    mat = np.zeros((max_uid + 1, max_pid + 1), dtype=np.float32)
    pos_df = train_df[train_df["label"] == 1]
    for r in pos_df.itertuples(index=False):
        mat[int(r.user_id), int(r.product_id)] += 1.0

    svd = TruncatedSVD(n_components=24, random_state=SEED)
    user_factors = svd.fit_transform(mat).astype(np.float32)
    item_factors = svd.components_.T.astype(np.float32)
    return MatrixFactorizationWrapper(user_factors, item_factors)


LGB_FEATURES = [
    "user_id", "product_id", "hour", "is_weekend", "is_mobile",
    "time_slot", "cat_code", "price", "user_cat_ratio", "ctx_item_pop"
]


def extract_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame()
    x["user_id"] = df["user_id"].astype("int32")
    x["product_id"] = df["product_id"].astype("int32")
    x["hour"] = df["hour"].astype("int32")
    x["is_weekend"] = df["is_weekend"].astype("int32")
    x["is_mobile"] = df["is_mobile"].astype("int32")
    x["time_slot"] = df["time_slot"].astype("int32")
    x["cat_code"] = df["category"].map(CAT_MAP).fillna(0).astype("int32")
    x["price"] = df["price"].astype("float32")
    x["user_cat_ratio"] = df["user_cat_ratio"].astype("float32")
    x["ctx_item_pop"] = df["ctx_item_pop"].astype("float32")
    return x[LGB_FEATURES]


class LightGBMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, booster):
        self.booster = booster

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.booster.predict_proba(extract_tabular_features(model_input))[:, 1]


def train_lightgbm(train_df: pd.DataFrame) -> LightGBMWrapper:
    X_train = extract_tabular_features(train_df)
    y_train = train_df["label"].to_numpy(dtype=np.int32)

    clf = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        subsample_for_bin=50000,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    return LightGBMWrapper(clf)


class ContextNCFNet(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_cats: int = 5):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, 16)
        self.item_emb = nn.Embedding(num_items + 1, 16)
        self.cat_emb = nn.Embedding(num_cats, 4)
        self.mlp = nn.Sequential(
            nn.Linear(43, 64),
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
    def __init__(self, state_dict: dict, num_users: int, num_items: int):
        self.state_dict = state_dict
        self.num_users = num_users
        self.num_items = num_items
        self._net = None

    def _get_net(self):
        if self._net is None:
            net = ContextNCFNet(self.num_users, self.num_items)
            net.load_state_dict(self.state_dict)
            net.eval()
            self._net = net
        return self._net

    def _prepare_tensors(self, df: pd.DataFrame):
        u = torch.tensor(df["user_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        i = torch.tensor(df["product_id"].to_numpy(dtype=np.int64), dtype=torch.long)
        c = torch.tensor(df["category"].map(CAT_MAP).fillna(0).to_numpy(dtype=np.int64), dtype=torch.long)
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
            return torch.sigmoid(net(u, i, c, num)).numpy()


def train_ncf(train_df: pd.DataFrame, products_df: pd.DataFrame) -> NCFWrapper:
    num_users = int(train_df["user_id"].max())
    num_items = int(products_df["product_id"].max())

    net = ContextNCFNet(num_users, num_items)
    wrapper = NCFWrapper({}, num_users, num_items)
    u, i, c, num = wrapper._prepare_tensors(train_df)
    y = torch.tensor(train_df["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    loader = DataLoader(TensorDataset(u, i, c, num, y), batch_size=512, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss()

    net.train()
    for _ in range(5):
        for bu, bi, bc, bnum, by in loader:
            optimizer.zero_grad()
            loss = criterion(net(bu, bi, bc, bnum), by)
            loss.backward()
            optimizer.step()

    net.eval()
    return NCFWrapper({k: v.cpu() for k, v in net.state_dict().items()}, num_users, num_items)


def main():
    print("1. Dang doc du lieu tu ./shared_data ...")
    products_df = pd.read_parquet(f"{DATA_DIR}/products.parquet")
    train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    test_df = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    feat_user_cat = pd.read_parquet(f"{DATA_DIR}/feat_user_cat.parquet")
    feat_ctx_item = pd.read_parquet(f"{DATA_DIR}/feat_ctx_item.parquet")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    models_spec = [
        ("Content_Based_Filtering", {"algo": "TFIDF_Category_Cosine", "context_aware": False}, lambda: train_content_based(train_df, products_df)),
        ("Collaborative_Filtering_MF", {"algo": "TruncatedSVD", "n_components": 24, "context_aware": False}, lambda: train_matrix_factorization(train_df, products_df)),
        ("LightGBM_Context_Aware", {"algo": "LGBMClassifier", "n_estimators": 180, "lr": 0.05, "context_aware": True}, lambda: train_lightgbm(train_df)),
        ("NCF_PyTorch_CPU", {"algo": "ContextNCF_MLP", "emb_dim": 16, "epochs": 5, "context_aware": True}, lambda: train_ncf(train_df, products_df)),
    ]

    results = []
    best_model_name, best_ndcg, best_run_id = None, -1.0, None

    print("2. Bat dau huan luyen, danh gia & log 4 mo hinh CLO1 len MLflow...\n")
    for model_name, params, train_fn in models_spec:
        print(f"--> Dang chay: {model_name} ...")
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(params)
            pyfunc_model = train_fn()

            metrics = evaluate_model(
                predict_fn=lambda df_in, m=pyfunc_model: m.predict(None, df_in),
                test_df=test_df,
                products_df=products_df,
                feat_user_cat=feat_user_cat,
                feat_ctx_item=feat_ctx_item,
                k=10,
            )
            mlflow.log_metrics(metrics)
            mlflow.pyfunc.log_model(artifact_path="model", python_model=pyfunc_model)

            results.append({"Model": model_name, **metrics, "run_id": run.info.run_id})
            if metrics["ndcg_at_10"] > best_ndcg:
                best_ndcg = metrics["ndcg_at_10"]
                best_model_name = model_name
                best_run_id = run.info.run_id

    summary_df = pd.DataFrame(results).sort_values("ndcg_at_10", ascending=False)
    print("\n==================== BANG SO SANH 4 MO HINH (CLO1) ====================")
    print(summary_df.drop(columns=["run_id"]).to_string(index=False))

    print(f"\n3. Dang ky mo hinh tot nhat [{best_model_name}] (NDCG@10={best_ndcg}) len MLflow Registry...")
    mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "Production", mv.version)
    try:
        client.transition_model_version_stage(name=REGISTERED_MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True)
    except Exception:
        pass

    print(f"HOAN TAT! Model '{REGISTERED_MODEL_NAME}' version {mv.version} da duoc gan nhan 'Production'.")


if __name__ == "__main__":
    main()
