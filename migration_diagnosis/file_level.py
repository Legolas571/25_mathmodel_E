"""File-level (per-segment) transfer labelling.

Rationale for the redo:
  * window-level probability voting amplifies the source prior (OR = 47.8% of
    source windows), which produced a degenerate 13 OR / 3 B / 0 IR / 0 N label
    set that contradicts the problem statement (target contains OR, IR, B, N);
  * the 16 target segments are separable on their own (silhouette 0.36 at k=4);
  * the problem *gives* us the information that four classes are present, which
    is a hard constraint we are allowed to enforce.

Everything here is validated on the proxy tasks (which do have ground truth)
before being applied to the real target.
"""
import numpy as np

import config as C          # must come before sklearn: it sets LOKY/JOBLIB env vars
import data_io as dio
import methods_shallow as MS
import models_common as MC

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FILE_METHODS = ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"]
FILE_METHOD_DESC = {
    "N0": "file-level source model, no adaptation",
    "N1": "file-level + CORAL",
    "N2": "file-level + CORAL + class-existence constraint (>=1 each)",
    "N3": "file-level + CORAL + balanced prior constraint",
    "N4": "cluster-then-match (target clusters matched to source prototypes)",
    "N5": "file-level + CORAL + cluster fusion + constraint",
    "N6": "file-level, NO adaptation, + class-existence constraint",
    "N7": "file-level + PCA (10 dims) + CORAL + class-existence constraint",
    "N8": "N6 evidence + CORAL-aligned prototype evidence, fused, + constraint",
}


def _norm_rows(A):
    A = np.asarray(A, dtype=np.float64)
    lo, hi = A.min(axis=1, keepdims=True), A.max(axis=1, keepdims=True)
    return (A - lo) / (hi - lo + 1e-12)


def fused_scores(Xs_f, ys_f, Xt_f, classes, seed=0, w_align=0.5):
    """Two independent evidences averaged into one score matrix:
      (a) unaligned file-level classifier posterior  (N0/N6 evidence)
      (b) negative distance to CORAL-aligned source class prototypes
    """
    n_classes = len(classes)
    sc = MC.source_scaler(Xs_f)
    Xs_u, Xt_u = sc.transform(Xs_f), sc.transform(Xt_f)
    clf = MC.make_classifier(ys_f, seed, n_classes).fit(Xs_u, ys_f)
    proba = clf.predict_proba(Xt_u)
    try:
        from sklearn.preprocessing import RobustScaler
        sc2 = RobustScaler().fit(Xs_f)
        Xs_a, Xt_a = MS.coral_align(sc2.transform(Xs_f), sc2.transform(Xt_f))
        protos = _prototypes(Xs_a, ys_f, n_classes)
        negd = -np.linalg.norm(Xt_a[:, None, :] - protos[None, :, :], axis=2)
    except Exception:
        negd = np.zeros_like(proba)
    return (1 - w_align) * _norm_rows(proba) + w_align * _norm_rows(negd)


def aggregate_to_files(df, cols):
    g = df.groupby("file_id")
    out = g[cols].mean()
    meta = g.agg(fault_type=("fault_type", "first"))
    out = out.join(meta)
    return out.reset_index()


def constrained_assign(scores, min_per_class=1, balanced=False):
    """Assign each row to a class.  scores: (n, c), higher = better.
    min_per_class guarantees every class appears; balanced forces equal counts."""
    scores = np.asarray(scores, dtype=np.float64)
    n, c = scores.shape
    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(c, dtype=int)
    if balanced:
        cap = np.full(c, n // c)
        for j in range(n - cap.sum()):
            cap[j] += 1
    else:
        cap = np.full(c, n)

    if min_per_class:
        for _ in range(min_per_class):
            for j in np.argsort(-scores.max(0)):
                if counts[j] >= min_per_class:
                    continue
                cand = np.where(labels < 0)[0]
                if len(cand) == 0:
                    break
                i = cand[int(np.argmax(scores[cand, j]))]
                labels[i] = j
                counts[j] += 1

    for i in np.argsort(-scores.max(1)):
        if labels[i] >= 0:
            continue
        allowed = np.where(counts < cap)[0]
        if len(allowed) == 0:
            allowed = np.arange(c)
        j = int(allowed[int(np.argmax(scores[i, allowed]))])
        labels[i] = j
        counts[j] += 1
    return labels


def _prototypes(Xs, ys, n_classes):
    return np.stack([Xs[ys == c].mean(0) if (ys == c).any() else np.zeros(Xs.shape[1])
                     for c in range(n_classes)])


def _neg_dist(scores_like, protos):
    d = np.linalg.norm(scores_like[:, None, :] - protos[None, :, :], axis=2)
    return -d


def cluster_match(Xt, protos, n_clusters, seed=0):
    """Cluster the target and match each cluster to the closest source prototype."""
    Z = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9)
    k = int(min(n_clusters, len(Xt)))
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    sil = float(silhouette_score(Z, km.labels_)) if k > 1 and len(set(km.labels_)) > 1 else np.nan
    cents = km.cluster_centers_ * Xt.std(0) + Xt.mean(0)
    d = np.linalg.norm(cents[:, None, :] - protos[None, :, :], axis=2)
    cluster_to_class = d.argmin(1)
    return cluster_to_class[km.labels_], sil, km.labels_


def run_file_method(name, Xs_f, ys_f, Xt_f, classes, seed=0, extra=None):
    """Xs_f/Xt_f are file-level feature matrices; returns (labels, scores)."""
    n_classes = len(classes)
    sc = MC.source_scaler(Xs_f)
    if name in ("N0", "N6"):
        Xs_a, Xt_a = sc.transform(Xs_f), sc.transform(Xt_f)
        clf = MC.make_classifier(ys_f, seed, n_classes).fit(Xs_a, ys_f)
        proba = clf.predict_proba(Xt_a)
        if name == "N6":
            return constrained_assign(proba, min_per_class=1), proba
        return np.asarray(clf.classes_)[proba.argmax(1)], proba
    if name == "N7":
        from sklearn.decomposition import PCA
        Xs_s, Xt_s = sc.transform(Xs_f), sc.transform(Xt_f)
        d = int(min(10, Xs_s.shape[1] - 1, max(2, len(Xs_f) // 4)))
        pca = PCA(n_components=d, random_state=seed).fit(Xs_s)
        Xs_a, Xt_a = MS.coral_align(pca.transform(Xs_s), pca.transform(Xt_s))
        clf = MC.make_classifier(ys_f, seed, n_classes).fit(Xs_a, ys_f)
        proba = clf.predict_proba(Xt_a)
        return constrained_assign(proba, min_per_class=1), proba
    if name == "N8":
        s = fused_scores(Xs_f, ys_f, Xt_f, classes, seed)
        return constrained_assign(s, min_per_class=1), s
    if name == "N4":
        Xs_a, Xt_a = sc.transform(Xs_f), sc.transform(Xt_f)
        protos = _prototypes(Xs_a, ys_f, n_classes)
        lab, sil, _ = cluster_match(Xt_a, protos, n_classes, seed)
        sc_mat = _neg_dist(Xt_a, protos)
        return lab, sc_mat

    Xs_a, Xt_a = MS.coral_align(sc.transform(Xs_f), sc.transform(Xt_f))
    clf = MC.make_classifier(ys_f, seed, n_classes).fit(Xs_a, ys_f)
    proba = clf.predict_proba(Xt_a)
    if name == "N1":
        return np.asarray(clf.classes_)[proba.argmax(1)], proba
    if name in ("N2", "N5"):
        lab = constrained_assign(proba, min_per_class=extra.get("min_per_class", 1)
                                 if extra else 1)
        return lab, proba
    if name == "N3":
        lab = constrained_assign(proba, balanced=True)
        return lab, proba
    raise KeyError(name)


def run_file_method_fused(name, Xs_f, ys_f, Xt_f, classes, seed=0):
    """N5 = constrained assignment biased by the target's own cluster structure."""
    n_classes = len(classes)
    sc = MC.source_scaler(Xs_f)
    Xs_a, Xt_a = MS.coral_align(sc.transform(Xs_f), sc.transform(Xt_f))
    protos = _prototypes(Xs_a, ys_f, n_classes)
    clf = MC.make_classifier(ys_f, seed, n_classes).fit(Xs_a, ys_f)
    proba = clf.predict_proba(Xt_a)
    cl_lab, sil, _ = cluster_match(Xt_a, protos, n_classes, seed)
    boost = np.zeros_like(proba)
    for i, j in enumerate(cl_lab):
        boost[i, int(j)] += 0.5
    lab = constrained_assign(proba + boost, min_per_class=1)
    return lab, proba + boost


def dispatch(name, Xs_f, ys_f, Xt_f, classes, seed=0):
    if name == "N5":
        return run_file_method_fused(name, Xs_f, ys_f, Xt_f, classes, seed)
    return run_file_method(name, Xs_f, ys_f, Xt_f, classes, seed)


COMPACT6 = ["t_kurtosis", "t_crest", "t_impulse", "t_clearance", "t_shape", "o_entropy"]


def _select_cols(src, key):
    if key == "compact6":
        return [c for c in COMPACT6 if c in src.columns]
    return dio.feature_columns(src, key)


def _proxy_rows(src, splits, cols, proxy, fold, classes):
    tr, te = dio.load_proxy_split(src, splits, proxy, fold)
    sf = aggregate_to_files(tr, cols)
    tf = aggregate_to_files(te, cols)
    sf = sf[sf["fault_type"].isin(classes)]
    tf = tf[tf["fault_type"].isin(classes)]
    Xs = sf[cols].to_numpy(float)
    ys = sf["fault_type"].map(C.CLASS_TO_IDX).to_numpy()
    Xt = tf[cols].to_numpy(float)
    yt = tf["fault_type"].map(C.CLASS_TO_IDX).to_numpy()
    return Xs, ys, Xt, yt


def proxy_main(argv=None):
    import argparse

    import data_io as dio
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxies", default=",".join(C.PROXY_ORDER))
    ap.add_argument("--methods", default=",".join(FILE_METHODS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--feature-set", default=C.FEATURE_MAIN)
    ap.add_argument("--tag", default="proxy_file")
    args = ap.parse_args(argv)

    src = dio.load_source_features()
    splits = dio.load_splits()
    cols = _select_cols(src, args.feature_set)
    print(f"[file_level] feature set {args.feature_set}: {len(cols)} columns")
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for proxy in args.proxies.split(","):
        classes = C.PROXY_TASKS[proxy]["classes"]
        for fold in dio.proxy_folds(splits, proxy):
            Xs, ys, Xt, yt = _proxy_rows(src, splits, cols, proxy, fold, classes)
            for method in args.methods.split(","):
                for seed in seeds:
                    try:
                        lab, sc = dispatch(method, Xs, ys, Xt, classes, seed)
                    except Exception as e:
                        print(f"  [FAIL] {proxy}/{fold}/{method}: {type(e).__name__}: {e}")
                        continue
                    m = MC.evaluate(yt, lab, classes)
                    rows.append({"proxy": proxy, "fold": fold, "method": method,
                                 "seed": seed, "n_source_files": len(ys),
                                 "n_target_files": len(yt), **m})
    d = pd.DataFrame(rows)
    out = C.RES_DIR / f"runs_{args.tag}.csv"
    d.to_csv(out, index=False, encoding="utf-8")
    s = d.groupby(["proxy", "method"]).agg(
        macro_f1=("macro_f1", "mean"), std=("macro_f1", "std"),
        n_runs=("macro_f1", "size")).reset_index()
    base = s[s["method"] == "N0"].set_index("proxy")["macro_f1"]
    s["gain_vs_N0"] = s.apply(lambda r: r["macro_f1"] - base.get(r["proxy"], np.nan), axis=1)
    s.to_csv(C.RES_DIR / "file_method_comparison.csv", index=False, encoding="utf-8")
    print("\n=== file-level methods: macro-F1 ===")
    print(s.pivot(index="method", columns="proxy", values="macro_f1").round(4).to_string())
    print("\n=== gain vs N0 ===")
    print(s.pivot(index="method", columns="proxy", values="gain_vs_N0").round(4).to_string())
    print(f"[file_level] wrote {out}")
    return s


if __name__ == "__main__":
    import pandas as pd

    proxy_main()

