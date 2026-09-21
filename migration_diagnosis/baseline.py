import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

import config as C
import data_io as dio


def mmd_rbf(Xs, Xt, gamma=None, max_n=1500):
    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    rs = np.random.RandomState(C.SEEDS[0])
    if len(Xs) > max_n:
        Xs = Xs[rs.choice(len(Xs), max_n, replace=False)]
    if len(Xt) > max_n:
        Xt = Xt[rs.choice(len(Xt), max_n, replace=False)]
    Z = np.vstack([Xs, Xt])
    if gamma is None:
        d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2) if len(Z) < 800 else None
        if d2 is None:
            idx = rs.choice(len(Z), 800, replace=False)
            d2 = np.sum((Z[idx, None, :] - Z[None, :, :]) ** 2, axis=2)
        gamma = 1.0 / (np.median(d2) + 1e-12)
    def k(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * d)
    return float(k(Xs, Xs).mean() + k(Xt, Xt).mean() - 2 * k(Xs, Xt).mean())


def wasserstein_1d(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n = max(len(a), len(b))
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def a_distance(Xs, Xt, seed=0, max_n=2000):
    """Ben-David et al.: A-distance = 2(1 - 2*error) of a domain classifier.
    With accuracy a, error = 1-a, so A-distance = 2(2a - 1); 0 = indistinguishable
    domains, 2 = perfectly separable. Clipped at 0 because a < 0.5 is noise."""
    rs = np.random.RandomState(seed)
    if len(Xs) > max_n:
        Xs = Xs[rs.choice(len(Xs), max_n, replace=False)]
    if len(Xt) > max_n:
        Xt = Xt[rs.choice(len(Xt), max_n, replace=False)]
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    acc = float(np.mean(accs))
    return max(0.0, 2.0 * (2.0 * acc - 1.0)), acc


def domain_shift_report(src, tgt, cols):
    rows = []
    for c in cols:
        a = src[c].to_numpy(float)
        b = tgt[c].to_numpy(float)
        ks, p = ks_2samp(a, b)
        rows.append({"feature": c, "ks_stat": float(ks), "ks_p": float(p),
                     "src_median": float(np.median(a)), "tgt_median": float(np.median(b)),
                     "src_mean": float(a.mean()), "tgt_mean": float(b.mean()),
                     "wasserstein": wasserstein_1d(a, b)})
    df = pd.DataFrame(rows).sort_values("ks_stat")
    df.to_csv(C.RES_DIR / "domain_shift_features.csv", index=False, encoding="utf-8")
    return df


def zero_shot_predict(src, tgt, cols, model_obj=None):
    obj = model_obj or dio.load_source_model()
    if list(obj["columns"]) != list(cols):
        raise ValueError("model columns do not match requested feature set")
    Xt = obj["pipeline"].transform(tgt[cols].to_numpy(float))
    proba = obj["estimator"].predict_proba(Xt)
    pred = obj["estimator"].predict(Xt)
    names = list(obj["classes"])
    d = pd.DataFrame(proba, columns=names)
    d.insert(0, "file_id", tgt["file_id"].to_numpy())
    d.insert(1, "pred", [names[i] for i in pred])
    d.to_csv(C.RES_DIR / "baseline_predictions.csv", index=False, encoding="utf-8")
    g = dio.file_level(tgt, pred, names)
    return proba, g, names


def main():
    src = dio.load_source_features()
    tgt = dio.load_target_features()
    cols = dio.main_feature_columns(src, tgt)
    print(f"[baseline] feature set {C.FEATURE_MAIN}: {len(cols)} columns, "
          f"compatible with target: {dio.check_compatible(src, tgt, cols)}")

    proba, g, names = zero_shot_predict(src, tgt, cols)
    print("\n=== M0 no-adaptation baseline on target ===")
    print("  window-level predicted counts:",
          pd.Series([names[i] for i in proba.argmax(1)]).value_counts().to_dict())
    print(f"  mean max-probability: {proba.max(axis=1).mean():.3f}")
    print("  file-level labels:", dict(zip(g["file_id"], g["pred"])))
    print("  file-level distribution:", g["pred"].value_counts().to_dict())
    print(f"  mean within-file agreement: {g['agree'].mean():.3f}")

    sh = domain_shift_report(src, tgt, cols)
    Xs_raw = src[cols].to_numpy(float)
    Xt_raw = tgt[cols].to_numpy(float)
    sc = RobustScaler().fit(Xs_raw)
    m = mmd_rbf(sc.transform(Xs_raw), sc.transform(Xt_raw))
    ad, eps = a_distance(sc.transform(Xs_raw), sc.transform(Xt_raw))
    print(f"\n=== domain distance (before adaptation) ===")
    print(f"  MMD(RBF)      = {m:.4f}")
    print(f"  A-distance    = {ad:.4f}  (domain classifier accuracy {eps:.3f})")
    print(f"  most shifted features: {sh.tail(3)['feature'].tolist()}")
    print(f"  most invariant features: {sh.head(3)['feature'].tolist()}")

    with open(C.RES_DIR / "domain_distance_baseline.txt", "w", encoding="utf-8") as fh:
        fh.write(f"MMD(RBF)={m:.6f}\nA_distance={ad:.6f}\ndomain_clf_acc={eps:.6f}\n")
    return g


if __name__ == "__main__":
    main()
