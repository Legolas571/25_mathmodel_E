import json

import numpy as np
import pandas as pd

import config as C


def load_source_features():
    df = pd.read_csv(C.SRC_FEATURES)
    missing = [c for c in C.META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"features_source.csv is missing meta columns: {missing}")
    print(f"[data_io] source table {df.shape}, files={df['file_id'].nunique()}, "
          f"classes={df['fault_type'].value_counts().to_dict()}")
    return df


def available_columns(df):
    return [c for c in df.columns if c not in C.META_COLS]


def feature_columns(df, key):
    cols = available_columns(df)
    t = [c for c in cols if c.startswith("t_")]
    f = [c for c in cols if c.startswith("f_")]
    og = [c for c in cols if c.startswith("o_") and not c.startswith("o_theo_")]
    ot = [c for c in cols if c.startswith("o_theo_")]
    t_nd = [c for c in t if c not in C.DIM_TIME]
    t_dim = [c for c in t if c in C.DIM_TIME]
    sets = {
        "F1": t + f + og + ot,
        "F2": t_nd + og,
        "F3": og + ot,
        "F4": og,
        "F5": ot,
        "F6": t,
        "F7": f,
        "F8": t_nd + og + t_dim,
    }
    if key not in sets:
        raise KeyError(f"unknown feature set {key}")
    out = sets[key]
    if not out:
        raise ValueError(f"feature set {key} is empty for this table")
    return out


def load_splits(protocol=None, path=None):
    p = path or (C.RES_DIR / "splits_all.json")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run make_splits.py first")
    data = json.loads(p.read_text(encoding="utf-8"))
    if protocol is None:
        return data
    if protocol not in data:
        raise KeyError(f"protocol {protocol} not in {p}")
    return data[protocol]


def build_xy(df, cols, classes=None):
    d = df
    if classes is not None:
        d = d[d["fault_type"].isin(classes)]
    X = d[cols].to_numpy(dtype=np.float64)
    y = d["fault_type"].map(C.CLASS_TO_IDX).to_numpy(dtype=np.int64)
    return X, y, d.reset_index(drop=True)


def group_split(df, split):
    tr_ids = set(split["train"])
    te_ids = set(split["test"])
    overlap = tr_ids & te_ids
    if overlap:
        raise AssertionError(f"group_split leakage: {len(overlap)} files in both sides")
    tr = df[df["file_id"].isin(tr_ids)]
    te = df[df["file_id"].isin(te_ids)]
    if tr.empty or te.empty:
        raise ValueError("empty train or test split")
    return tr, te


def fold_ids(protocol, splits):
    return sorted(splits[protocol].keys())


def class_counts(df):
    return df["fault_type"].value_counts().to_dict()
