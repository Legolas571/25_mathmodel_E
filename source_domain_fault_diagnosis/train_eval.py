import argparse
import json
import time

import numpy as np
import pandas as pd

import config as C
import data_io as dio
import feature_prep as fp
import models as M

RESULT_COLS = (
    ["protocol", "feature_set", "model", "seed", "fold",
     "n_train_files", "n_test_files", "n_train_windows", "n_test_windows",
     "n_features", "fit_seconds", "predict_ms_per_1k",
     "accuracy", "macro_f1", "weighted_f1", "n_classes",
     "file_accuracy", "file_macro_f1", "n_test_files_eval"]
    + [f"{m}_{c}" for c in C.CLASSES for m in ("precision", "recall", "f1")]
)


def _append(path, row, header):
    d = pd.DataFrame([row]).reindex(columns=RESULT_COLS)
    d.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
    return False


def file_level_vote(te_df, y_pred):
    d = pd.DataFrame({"file_id": te_df["file_id"].to_numpy(),
                      "true": te_df["fault_type"].to_numpy(),
                      "pred": [C.CLASSES[i] for i in y_pred]})
    g = d.groupby("file_id").agg(true=("true", "first"),
                                 pred=("pred", lambda s: s.value_counts().idxmax()))
    from sklearn.metrics import accuracy_score, f1_score
    return (float(accuracy_score(g["true"], g["pred"])),
            float(f1_score(g["true"], g["pred"], average="macro", zero_division=0)),
            int(len(g)))


def _row_base(protocol, fset, model, seed, fold, tr, te):
    return {
        "protocol": protocol, "feature_set": fset, "model": model, "seed": seed,
        "fold": fold, "n_train_files": tr["file_id"].nunique(),
        "n_test_files": te["file_id"].nunique(),
        "n_train_windows": len(tr), "n_test_windows": len(te),
        "n_features": 0, "fit_seconds": np.nan,
    }


def run_one(df, protocol, fset, model, seed, fold, split, cols):
    tr, te = dio.group_split(df, split)
    classes = C.PROTOCOL_CLASSES[protocol]
    Xtr, ytr, tr2 = dio.build_xy(tr, cols, classes)
    Xte, yte, te2 = dio.build_xy(te, cols, classes)
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return None, None

    pipe = fp.FeaturePipeline()
    if M.supports_scaling(model):
        Xtr, Xte = pipe.fit_transform(Xtr, Xte, columns=cols)
    else:
        pipe.columns_ = list(cols)

    cw = fp.compute_class_weight(ytr)
    base = _row_base(protocol, fset, model, seed, fold, tr2, te2)
    base["n_features"] = Xtr.shape[1]

    t0 = time.time()
    if model == "vote":
        est = [("svm", M.make_model("svm", len(classes), seed, cw)),
               ("rf", M.make_model("rf", len(classes), seed, cw)),
               ("hgb", M.make_model("hgb", len(classes), seed, cw))]
        clf = M.make_vote(est)
    elif model == "mlp":
        from model_torch import train_mlp
        y_pred, proba, fit_s = train_mlp(Xtr, ytr, Xte, len(classes), seed, cw,
                                         files=tr2["file_id"].to_numpy())
        m = M.metrics(yte, y_pred, np.unique(yte))
        row = dict(base, fit_seconds=fit_s)
        row.update({k: v for k, v in m.items() if not k.startswith("_")})
        fa, ff, nf = file_level_vote(te2, y_pred)
        row.update(file_accuracy=fa, file_macro_f1=ff, n_test_files_eval=nf)
        row["predict_ms_per_1k"] = 1000.0 * (time.time() - t0 - fit_s) / max(len(yte), 1) * 1000
        return row, m
    else:
        clf = M.make_model(model, len(classes), seed, cw)

    y_pred, proba, fit_s = M.fit_predict(clf, Xtr, ytr, Xte)
    elapsed = time.time() - t0
    m = M.metrics(yte, y_pred, np.unique(yte))
    row = dict(base, fit_seconds=fit_s)
    row.update({k: v for k, v in m.items() if not k.startswith("_")})
    fa, ff, nf = file_level_vote(te2, y_pred)
    row.update(file_accuracy=fa, file_macro_f1=ff, n_test_files_eval=nf)
    row["predict_ms_per_1k"] = 1000.0 * (elapsed - fit_s) / max(len(yte), 1) * 1000
    return row, m


def _done_keys(path):
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    return set(zip(d["protocol"], d["feature_set"], d["model"], d["seed"], d["fold"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default=",".join(C.PROTOCOLS))
    ap.add_argument("--feature-sets", default=",".join(C.MAIN_FEATURE_SETS))
    ap.add_argument("--models", default=",".join(C.MAIN_MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--fold-limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)

    protocols = args.protocols.split(",")
    fsets = args.feature_sets.split(",")
    mods = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    df = dio.load_source_features()
    splits = dio.load_splits()
    out = C.RES_DIR / f"runs_{args.tag}.csv"
    if args.fresh and out.exists():
        out.unlink()
    done = _done_keys(out)

    combos = [(p, f, m, s, fo)
              for p in protocols for f in fsets for m in mods for s in seeds
              for fo in sorted(splits[p].keys())]
    if args.fold_limit:
        combos = combos[:args.fold_limit]
    todo = [c for c in combos if c not in done]
    print(f"[train_eval] tag={args.tag} total={len(combos)} todo={len(todo)} "
          f"(cached {len(combos)-len(todo)})")

    header = not out.exists()
    t_start = time.time()
    n = 0
    for (p, fset, model, seed, fold) in todo:
        cols = dio.feature_columns(df, fset)
        t0 = time.time()
        try:
            row, _ = run_one(df, p, fset, model, seed, fold, splits[p][fold], cols)
        except Exception as e:
            print(f"  [FAIL] {p}/{fset}/{model}/seed{seed}/{fold}: {type(e).__name__}: {e}")
            continue
        if row is None:
            continue
        header = _append(out, row, header)
        n += 1
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t_start
            print(f"  {n}/{len(todo)}  elapsed {el/60:.1f} min  "
                  f"eta {el/max(n,1)*(len(todo)-n)/60:.1f} min  "
                  f"last: {p}/{fset}/{model}/s{seed}/{fold} "
                  f"winF1={row['macro_f1']:.3f} fileF1={row.get('file_macro_f1', float('nan')):.3f} "
                  f"({time.time()-t0:.1f}s)")
    print(f"[train_eval] wrote {out} (+{n} rows) in {(time.time()-t_start)/60:.1f} min")
    return out


if __name__ == "__main__":
    main()
