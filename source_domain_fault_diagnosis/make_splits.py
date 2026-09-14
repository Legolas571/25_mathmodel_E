import json

import numpy as np
import pandas as pd

import config as C
import data_io as dio


def _file_table(df):
    return (df.groupby("file_id")
              .agg(fault_type=("fault_type", "first"), load=("load", "first"),
                   group=("group", "first"), fs=("fs", "first"),
                   n_win=("win_idx", "size"))
              .reset_index())


def _strat_folds(sub, k, seed):
    rng = np.random.RandomState(seed)
    folds = {i: [] for i in range(k)}
    for c in C.CLASSES:
        ids = sorted(sub.loc[sub["fault_type"] == c, "file_id"].tolist())
        rng.shuffle(ids)
        for j, fid in enumerate(ids):
            folds[j % k].append(fid)
    return folds


def _wrap(files, fold_map, keyfmt="fold{}"):
    all_ids = set(files["file_id"])
    out = {}
    for k, test in fold_map.items():
        test = sorted(test)
        out[keyfmt.format(k)] = {"test": test,
                                 "train": sorted(all_ids - set(test))}
    return out


def p1_random(files, k=5, seed=42):
    return _wrap(files, _strat_folds(files, k, seed))


def p2_leave_one_load(files):
    folds = {}
    for L in sorted(files["load"].dropna().unique()):
        folds[int(L)] = sorted(files.loc[files["load"] == L, "file_id"].tolist())
    return _wrap(files, folds, "load{}")


def p3_cross_sampling_rate(files):
    g = files[files["fault_type"].isin(C.CLASS3)]
    ids12 = sorted(g.loc[g["group"].isin(["12k_DE", "12k_FE"]), "file_id"].tolist())
    ids48 = sorted(g.loc[g["group"] == "48k_DE", "file_id"].tolist())
    return {
        "12k_to_48k": {"train": ids12, "test": ids48},
        "48k_to_12k": {"train": ids48, "test": ids12},
    }


def p4_cross_position(files):
    g = files[files["fault_type"].isin(C.CLASS3)]
    de = sorted(g.loc[g["group"] == "12k_DE", "file_id"].tolist())
    fe = sorted(g.loc[g["group"] == "12k_FE", "file_id"].tolist())
    return {
        "DE_to_FE": {"train": de, "test": fe},
        "FE_to_DE": {"train": fe, "test": de},
    }


def p5_clean_48k(files, k=5, seed=42):
    sub = files[files["fs"] == 48000].reset_index(drop=True)
    return _wrap(sub, _strat_folds(sub, k, seed))


def _check(name, proto, files):
    ids = set(files["file_id"])
    for fold, sp in proto.items():
        tr, te = set(sp["train"]), set(sp["test"])
        assert not (tr & te), f"{name}/{fold}: train/test overlap"
        assert tr <= ids and te <= ids, f"{name}/{fold}: unknown file_id"
        assert tr and te, f"{name}/{fold}: empty side"
    print(f"  {name}: {len(proto)} folds, ok")


def describe(files, proto):
    rows = []
    for fold, sp in proto.items():
        te = files[files["file_id"].isin(sp["test"])]
        tr = files[files["file_id"].isin(sp["train"])]
        ct = te["fault_type"].value_counts().reindex(C.CLASSES).fillna(0).astype(int)
        row = {"fold": fold, "n_train_files": len(tr), "n_test_files": len(te),
               "n_train_win": int(tr["n_win"].sum()), "n_test_win": int(te["n_win"].sum())}
        for c in C.CLASSES:
            row[f"test_{c}"] = int(ct[c])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = dio.load_source_features()
    files = _file_table(df)
    print(f"[make_splits] {len(files)} files, {len(df)} windows")

    protos = {
        "P1": p1_random(files),
        "P2": p2_leave_one_load(files),
        "P3": p3_cross_sampling_rate(files),
        "P4": p4_cross_position(files),
        "P5": p5_clean_48k(files),
    }
    print("[make_splits] leakage self-check")
    for name, proto in protos.items():
        _check(name, proto, files)

    t1 = json.loads(C.T1_SPLITS.read_text(encoding="utf-8"))
    same = all(set(t1[str(i)]["test"]) == set(protos["P1"][f"fold{i}"]["test"])
               for i in range(5))
    print(f"[make_splits] P1 identical to task-1 splits.json: {same}")

    out = C.RES_DIR / "splits_all.json"
    out.write_text(json.dumps(protos, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[make_splits] wrote {out}")

    lines = []
    for name, proto in protos.items():
        tbl = describe(files, proto)
        tbl.insert(1, "protocol", name)
        lines.append(tbl)
    summary = pd.concat(lines, ignore_index=True)
    summary.to_csv(C.RES_DIR / "splits_summary.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[make_splits] P3/P4 use only 3 classes {C.CLASS3} (N exists only at 48 kHz)")
    return protos


if __name__ == "__main__":
    main()
