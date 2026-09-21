import json
import pickle

import numpy as np
import pandas as pd

import config as C


def _feature_columns(df, key):
    cols = [c for c in df.columns if c not in C.META_COLS]
    t = [c for c in cols if c.startswith("t_")]
    f = [c for c in cols if c.startswith("f_")]
    og = [c for c in cols if c.startswith("o_") and not c.startswith("o_theo_")]
    ot = [c for c in cols if c.startswith("o_theo_")]
    t_nd = [c for c in t if c not in C.DIM_TIME]
    t_dim = [c for c in t if c in C.DIM_TIME]
    return {"F1": t + f + og + ot, "F2": t_nd + og, "F3": og + ot, "F4": og,
            "F5": ot, "F6": t, "F7": f, "F8": t_nd + og + t_dim}[key]


def check_compatible(src, tgt, cols):
    miss_s = [c for c in cols if c not in src.columns]
    miss_t = [c for c in cols if c not in tgt.columns]
    if miss_s or miss_t:
        raise ValueError(f"feature set incompatible: missing in source={miss_s}, "
                         f"missing in target={miss_t}")
    bad = [c for c in cols if tgt[c].isna().all() or (tgt[c].std() == 0)]
    if bad:
        raise ValueError(f"target columns are degenerate (all-NaN or zero-variance): {bad}")
    return True


def load_source_features():
    df = pd.read_csv(C.SRC_FEATURES)
    print(f"[data_io] source {df.shape}, files={df['file_id'].nunique()}, "
          f"classes={df['fault_type'].value_counts().to_dict()}")
    return df


def load_target_features():
    df = pd.read_csv(C.TGT_FEATURES)
    if df["fault_type"].notna().any():
        raise ValueError("target features unexpectedly contain labels")
    print(f"[data_io] target {df.shape}, files={df['file_id'].nunique()}, "
          f"windows/file={sorted(df.groupby('file_id').size().unique())}")
    return df


def feature_columns(df, key):
    return _feature_columns(df, key)


def main_feature_columns(src, tgt=None):
    cols = _feature_columns(src, C.FEATURE_MAIN)
    if tgt is not None:
        check_compatible(src, tgt, cols)
    return cols


def load_source_model(path=None):
    p = path or C.SOURCE_MODEL_PKL
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run task 2 first")
    with open(p, "rb") as fh:
        obj = pickle.load(fh)
    print(f"[data_io] loaded source model: {obj['model_name']} on {obj['feature_set']} "
          f"({len(obj['columns'])} cols), classes={obj['classes']}")
    return obj


def load_splits():
    return json.loads(C.SPLITS_ALL.read_text(encoding="utf-8"))


def load_proxy_split(src_df, splits, proxy, fold):
    cfg = C.PROXY_TASKS[proxy]
    sp = splits[cfg["protocol"]][fold]
    classes = cfg["classes"]
    tr = src_df[src_df["file_id"].isin(set(sp["train"]))]
    te = src_df[src_df["file_id"].isin(set(sp["test"]))]
    tr = tr[tr["fault_type"].isin(classes)]
    te = te[te["fault_type"].isin(classes)]
    if tr.empty or te.empty:
        raise ValueError(f"empty proxy split {proxy}/{fold}")
    return tr, te


def build_xy(df, cols, classes):
    d = df[df["fault_type"].isin(classes)]
    X = d[cols].to_numpy(dtype=np.float64)
    y = d["fault_type"].map(C.CLASS_TO_IDX).to_numpy(dtype=np.int64)
    return X, y, d.reset_index(drop=True)


def build_X(df, cols):
    """Feature matrix for an unlabelled frame (target domain has all-NaN labels)."""
    return df[cols].to_numpy(dtype=np.float64)


def file_level(df, y_pred, class_names):
    d = pd.DataFrame({"file_id": df["file_id"].to_numpy(),
                      "pred": [class_names[i] for i in y_pred]})
    g = d.groupby("file_id")["pred"].agg(
        pred=lambda s: s.value_counts().idxmax(),
        agree=lambda s: s.value_counts().iloc[0] / len(s),
        n_win="size")
    return g.reset_index()


def proxy_folds(splits, proxy):
    return sorted(splits[C.PROXY_TASKS[proxy]["protocol"]].keys())


def class_counts(df, classes):
    return {c: int((df["fault_type"] == c).sum()) for c in classes}
