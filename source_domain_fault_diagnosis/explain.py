import argparse

import numpy as np
import pandas as pd

import config as C
import data_io as dio
import feature_prep as fp
import models as M

GROUPS = {
    "time_dimensionless": lambda c: c.startswith("t_") and c not in C.DIM_TIME,
    "time_raw_dim": lambda c: c in C.DIM_TIME,
    "freq_absolute": lambda c: c.startswith("f_"),
    "order_generic": lambda c: c.startswith("o_") and not c.startswith("o_theo_"),
    "order_theoretical": lambda c: c.startswith("o_theo_"),
}


def _eval(df, splits, protocol, cols, model="svm"):
    classes = C.PROTOCOL_CLASSES[protocol]
    scores, accs = [], []
    for fold, split in splits[protocol].items():
        tr, te = dio.group_split(df, split)
        Xtr, ytr, _ = dio.build_xy(tr, cols, classes)
        Xte, yte, _ = dio.build_xy(te, cols, classes)
        pipe = fp.FeaturePipeline()
        Xtr2, Xte2 = pipe.fit_transform(Xtr, Xte, columns=cols)
        clf = M.make_model(model, len(classes), C.SEEDS[0], fp.compute_class_weight(ytr))
        y_pred, _, _ = M.fit_predict(clf, Xtr2, ytr, Xte2)
        m = M.metrics(yte, y_pred, np.unique(yte))
        scores.append(m["macro_f1"])
        accs.append(m["accuracy"])
    return float(np.mean(scores)), float(np.mean(accs))


def group_ablation(protocols=("P1", "P3", "P4"), model="svm"):
    df = dio.load_source_features()
    splits = dio.load_splits()
    all_cols = dio.feature_columns(df, "F1")
    rows = []
    for p in protocols:
        full, _ = _eval(df, splits, p, all_cols, model)
        rows.append({"protocol": p, "removed": "-", "n_features": len(all_cols),
                     "macro_f1": full, "delta": 0.0})
        for gname, fn in GROUPS.items():
            keep = [c for c in all_cols if not fn(c)]
            if not keep or len(keep) == len(all_cols):
                continue
            s, _ = _eval(df, splits, p, keep, model)
            rows.append({"protocol": p, "removed": gname, "n_features": len(keep),
                         "macro_f1": s, "delta": s - full})
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "group_ablation.csv", index=False, encoding="utf-8")
    return out


def misclassified_files(protocol="P1", fset="F2", model="svm"):
    import report as R
    win, agg = R.error_analysis(protocol, fset, model)
    bad = win[~win["correct"]].sort_values("agree")
    bad.to_csv(C.RES_DIR / "misclassified_files.csv", index=False, encoding="utf-8")
    return win, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svm")
    args = ap.parse_args(argv)
    print("[explain] group ablation (leave-one-feature-group-out)")
    ab = group_ablation(model=args.model)
    print(ab.to_string(index=False))
    print(f"[explain] wrote {C.RES_DIR / 'group_ablation.csv'}")
    return ab


if __name__ == "__main__":
    main()
