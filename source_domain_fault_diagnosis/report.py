import json

import numpy as np
import pandas as pd

import config as C


def load_runs(tag="stage1"):
    p = C.RES_DIR / f"runs_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_all_runs(tags=None):
    if tags is None:
        files = sorted(C.RES_DIR.glob("runs_*.csv"))
    else:
        files = [C.RES_DIR / f"runs_{t}.csv" for t in tags]
    frames = []
    for p in files:
        if p.exists():
            d = pd.read_csv(p)
            d["run_tag"] = p.stem.replace("runs_", "")
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df):
    g = (df.groupby(["protocol", "feature_set", "model"])
           .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                n_folds=("fold", "nunique"), n_seeds=("seed", "nunique"),
                n_classes=("n_classes", "max"),
                fit_s=("fit_seconds", "mean"))
           .reset_index()
           .sort_values(["protocol", "macro_f1_mean"], ascending=[True, False]))
    return g


def protocol_shift(summary):
    best = summary.loc[summary.groupby("protocol")["macro_f1_mean"].idxmax()]
    base = best.loc[best["protocol"] == "P1", "macro_f1_mean"]
    base = float(base.iloc[0]) if len(base) else np.nan
    best = best.copy()
    best["drop_vs_P1"] = base - best["macro_f1_mean"]
    best["rel_drop_pct"] = 100.0 * best["drop_vs_P1"] / base if base == base else np.nan
    out = best[["protocol", "feature_set", "model", "macro_f1_mean", "macro_f1_std",
                "drop_vs_P1", "rel_drop_pct"]]
    out.to_csv(C.RES_DIR / "protocol_shift.csv", index=False, encoding="utf-8")
    return out


def feature_set_compare(summary):
    piv = (summary[summary["protocol"].isin(["P1", "P2", "P3", "P4"])]
           .pivot_table(index="feature_set", columns="protocol",
                        values="macro_f1_mean", aggfunc="max"))
    piv.to_csv(C.RES_DIR / "feature_set_compare.csv", encoding="utf-8")
    return piv


def plot_overview(summary, shift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    for p, sub in summary.groupby("protocol"):
        sub = sub.groupby("model")["macro_f1_mean"].max().sort_values()
        axes[0].barh([f"{p}/{m}" for m in sub.index], sub.values, label=p)
    axes[0].set_xlabel("best macro-F1 (over feature sets)")
    axes[0].set_title("Model comparison by protocol")
    axes[0].legend(fontsize=7)

    piv = (summary.pivot_table(index="feature_set", columns="protocol",
                               values="macro_f1_mean", aggfunc="max"))
    piv.plot(kind="bar", ax=axes[1], colormap="tab10")
    axes[1].set_ylabel("best macro-F1")
    axes[1].set_title("Feature set comparison (F3 vs F7 is the key contrast)")
    axes[1].tick_params(axis="x", rotation=0)

    axes[2].bar(shift["protocol"], shift["macro_f1_mean"], color="steelblue")
    axes[2].axhline(shift.loc[shift["protocol"] == "P1", "macro_f1_mean"].values[0],
                    color="r", ls="--", lw=1, label="P1 (i.i.d. upper bound)")
    axes[2].set_ylabel("best macro-F1")
    axes[2].set_title("Domain-shift strength across protocols")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "model_compare.png", dpi=140)
    plt.close(fig)


def plot_feature_set_compare(piv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_ylabel("best macro-F1")
    ax.set_title("Feature set vs protocol")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "feature_set_compare.png", dpi=140)
    plt.close(fig)


def confusion_for_best(df, summary, protocols=("P1", "P2", "P3", "P4", "P5")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    import data_io as dio
    import feature_prep as fp
    import models as M

    src = dio.load_source_features()
    splits = dio.load_splits()
    rows = []
    for p in protocols:
        sub = summary[summary["protocol"] == p]
        if sub.empty:
            continue
        best = sub.sort_values("macro_f1_mean", ascending=False).iloc[0]
        model, fset = best["model"], best["feature_set"]
        if model in ("mlp", "cnn1d", "vote"):
            sk = sub[sub["model"].isin(M.SKLEARN_MODELS)].sort_values(
                "macro_f1_mean", ascending=False)
            if sk.empty:
                continue
            best = sk.iloc[0]
            model, fset = best["model"], best["feature_set"]
        cols = dio.feature_columns(src, fset)
        classes = C.PROTOCOL_CLASSES[p]
        accs, cms, labels_all = [], [], sorted(C.CLASS_TO_IDX[c] for c in classes)
        for fold, split in splits[p].items():
            tr, te = dio.group_split(src, split)
            Xtr, ytr, _ = dio.build_xy(tr, cols, classes)
            Xte, yte, _ = dio.build_xy(te, cols, classes)
            pipe = fp.FeaturePipeline()
            Xtr2, Xte2 = pipe.fit_transform(Xtr, Xte, columns=cols)
            cw = fp.compute_class_weight(ytr)
            clf = M.make_model(model, len(classes), C.SEEDS[0], cw)
            y_pred, _, _ = M.fit_predict(clf, Xtr2, ytr, Xte2)
            cms.append(confusion_matrix(yte, y_pred, labels=labels_all))
            accs.append((yte, y_pred))
        cm = np.sum(cms, axis=0)
        names = [C.CLASSES[i] for i in labels_all]
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=names, yticklabels=names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{p} best: {model} on {fset}\nmacro-F1={best['macro_f1_mean']:.3f}")
        fig.tight_layout()
        fig.savefig(C.FIG_DIR / f"cm_{p}.png", dpi=140)
        plt.close(fig)
        rows.append({"protocol": p, "model": model, "feature_set": fset,
                     "macro_f1": best["macro_f1_mean"], "cm": json.dumps(cm.tolist()),
                     "labels": ",".join(names)})
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "confusion_matrices.csv", index=False, encoding="utf-8")
    return out


def permutation_importance(protocol="P1", fset="F2", model="rf", n_repeats=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance as perm_imp

    import data_io as dio
    import feature_prep as fp
    import models as M

    src = dio.load_source_features()
    splits = dio.load_splits()
    classes = C.PROTOCOL_CLASSES[protocol]
    cols = dio.feature_columns(src, fset)
    acc = np.zeros(len(cols))
    for fold, split in splits[protocol].items():
        tr, te = dio.group_split(src, split)
        Xtr, ytr, _ = dio.build_xy(tr, cols, classes)
        Xte, yte, _ = dio.build_xy(te, cols, classes)
        pipe = fp.FeaturePipeline()
        Xtr2, Xte2 = pipe.fit_transform(Xtr, Xte, columns=cols)
        clf = M.make_model(model, len(classes), C.SEEDS[0], fp.compute_class_weight(ytr))
        clf.fit(Xtr2, ytr)
        r = perm_imp(clf, Xte2, yte, n_repeats=n_repeats,
                     random_state=C.SEEDS[0], scoring="f1_macro", n_jobs=C.N_JOBS)
        acc += r.importances_mean
    acc /= max(len(splits[protocol]), 1)
    out = pd.DataFrame({"feature": cols, "importance": acc}).sort_values(
        "importance", ascending=False)
    out.to_csv(C.RES_DIR / "permutation_importance.csv", index=False, encoding="utf-8")

    top = out.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="teal")
    ax.set_xlabel("mean drop in macro-F1 when shuffled")
    ax.set_title(f"Permutation importance ({protocol}, {fset}, {model})")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "permutation_importance.png", dpi=140)
    plt.close(fig)
    return out


def tsne_plot(protocol="P1", fset="F2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    import data_io as dio
    import feature_prep as fp

    src = dio.load_source_features()
    splits = dio.load_splits()
    classes = C.PROTOCOL_CLASSES[protocol]
    cols = dio.feature_columns(src, fset)
    fold = sorted(splits[protocol].keys())[0]
    tr, te = dio.group_split(src, splits[protocol][fold])
    Xtr, ytr, _ = dio.build_xy(tr, cols, classes)
    Xte, yte, _ = dio.build_xy(te, cols, classes)
    pipe = fp.FeaturePipeline()
    Xtr2, Xte2 = pipe.fit_transform(Xtr, Xte, columns=cols)
    n = min(len(Xtr2), 1500)
    idx = np.random.RandomState(C.SEEDS[0]).choice(len(Xtr2), n, replace=False)
    Z = np.vstack([Xtr2[idx], Xte2])
    emb = TSNE(n_components=2, perplexity=30, max_iter=800, init="pca",
               random_state=C.SEEDS[0], learning_rate="auto", n_jobs=C.N_JOBS).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pal = {"B": "tab:blue", "IR": "tab:orange", "OR": "tab:green", "N": "tab:red"}
    for c in classes:
        m = ytr[idx] == C.CLASS_TO_IDX[c]
        if m.any():
            axes[0].scatter(emb[:n][m, 0], emb[:n][m, 1], s=6, alpha=0.6,
                            c=pal[c], label=c)
    axes[0].set_title(f"train fold features ({protocol}/{fset}) by class")
    axes[0].legend(markerscale=2, fontsize=8)
    for c in classes:
        m = yte == C.CLASS_TO_IDX[c]
        if m.any():
            axes[1].scatter(emb[n:][m, 0], emb[n:][m, 1], s=10, alpha=0.8,
                            c=pal[c], label=c)
    axes[1].set_title("test fold features by true class")
    axes[1].legend(markerscale=2, fontsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"tsne_{protocol}.png", dpi=140)
    plt.close(fig)


def error_analysis(protocol="P1", fset="F2", model="rf"):
    import data_io as dio
    import feature_prep as fp
    import models as M

    src = dio.load_source_features()
    audit = pd.read_csv(C.AUDIT_SOURCE)
    splits = dio.load_splits()
    classes = C.PROTOCOL_CLASSES[protocol]
    cols = dio.feature_columns(src, fset)
    rows = []
    for fold, split in splits[protocol].items():
        tr, te = dio.group_split(src, split)
        Xtr, ytr, _ = dio.build_xy(tr, cols, classes)
        Xte, yte, te2 = dio.build_xy(te, cols, classes)
        pipe = fp.FeaturePipeline()
        Xtr2, Xte2 = pipe.fit_transform(Xtr, Xte, columns=cols)
        clf = M.make_model(model, len(classes), C.SEEDS[0], fp.compute_class_weight(ytr))
        y_pred, _, _ = M.fit_predict(clf, Xtr2, ytr, Xte2)
        d = te2[["file_id", "fault_type", "load", "group", "fs"]].copy()
        d["pred"] = [C.CLASSES[i] for i in y_pred]
        d["fold"] = fold
        rows.append(d)
    allp = pd.concat(rows, ignore_index=True)
    win = allp.groupby(["file_id", "fault_type", "load", "group", "fs"]).agg(
        n_win=("pred", "size"),
        pred=("pred", lambda s: s.value_counts().idxmax()),
        agree=("pred", lambda s: s.value_counts().iloc[0] / len(s))).reset_index()
    win["correct"] = win["fault_type"] == win["pred"]
    win.to_csv(C.RES_DIR / "file_level_predictions.csv", index=False, encoding="utf-8")
    agg = (win.groupby(["fault_type", "load"])
              .agg(n=("correct", "size"), acc=("correct", "mean"))
              .reset_index())
    agg.to_csv(C.RES_DIR / "error_by_load.csv", index=False, encoding="utf-8")
    return win, agg


def save_deployable(fset="F2", model=None):
    import pickle

    import data_io as dio
    import feature_prep as fp
    import models as M

    src = dio.load_source_features()
    cols = dio.feature_columns(src, fset)
    X, y, _ = dio.build_xy(src, cols, C.CLASSES)
    pipe = fp.FeaturePipeline()
    Xs = pipe.fit_transform(X, X, columns=cols)[0]
    cw = fp.compute_class_weight(y)
    if model is None:
        s = summarize(load_all_runs())
        cand = s[(s["feature_set"] == fset) & (s["protocol"] == "P1") &
                 (s["model"].isin(M.SKLEARN_MODELS))]
        model = (cand.sort_values("macro_f1_mean", ascending=False).iloc[0]["model"]
                 if len(cand) else "hgb")
    clf = M.make_model(model, len(C.CLASSES), C.SEEDS[0], cw)
    clf.fit(Xs, y)
    obj = dict(model_name=model, feature_set=fset, columns=cols,
               classes=C.CLASSES, class_to_idx=C.CLASS_TO_IDX,
               pipeline=pipe, estimator=clf,
               trained_on="all 161 source-domain files",
               n_train_windows=int(len(y)), scaler=C.SCALER)
    p = C.MODEL_DIR / f"source_model_{fset}.pkl"
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    card = (f"model      : {model}\nfeature set: {fset} ({len(cols)} columns)\n"
            f"classes    : {C.CLASSES}\ntrain data : all 161 source files, "
            f"{len(y)} windows\nscaler     : {C.SCALER}\n"
            f"note       : only F2/F4 columns exist in features_target.csv, so this "
            f"artifact is the one task 3 can apply to the target domain\n")
    (C.MODEL_DIR / f"model_card_{fset}.txt").write_text(card, encoding="utf-8")
    print(f"[report] saved deployable model -> {p} ({model}, {len(cols)} features)")
    return p


def load_deployable(fset="F2"):
    import pickle
    with open(C.MODEL_DIR / f"source_model_{fset}.pkl", "rb") as fh:
        return pickle.load(fh)


def main():
    df = load_all_runs()
    if df.empty:
        print("[report] no runs found - run train_eval first")
        return
    summary = summarize(df)
    summary.to_csv(C.RES_DIR / "summary_by_model.csv", index=False, encoding="utf-8")
    shift = protocol_shift(summary)
    piv = feature_set_compare(summary)
    plot_overview(summary, shift)
    plot_feature_set_compare(piv)
    confusion_for_best(df, summary)
    try:
        permutation_importance()
    except Exception as e:
        print(f"[report] permutation importance skipped: {e}")
    try:
        tsne_plot()
    except Exception as e:
        print(f"[report] tsne skipped: {e}")
    try:
        error_analysis()
    except Exception as e:
        print(f"[report] error analysis skipped: {e}")
    try:
        save_deployable("F2")
    except Exception as e:
        print(f"[report] deployable export skipped: {e}")

    print("\n" + "=" * 78)
    print("TASK-2 SUMMARY (best per protocol)")
    print("=" * 78)
    print(shift.to_string(index=False))
    print("\n[feature set x protocol: best macro-F1]")
    print(piv.round(4).to_string())
    print("\n[top 12 configs overall]")
    print(summary.head(12).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
