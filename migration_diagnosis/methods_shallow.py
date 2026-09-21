import numpy as np
from sklearn.decomposition import PCA

import config as C
import models_common as MC


def _cov(X):
    X = np.asarray(X, dtype=np.float64)
    return np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6


def coral_align(Xs, Xt):
    Cs, Ct = _cov(Xs), _cov(Xt)
    ds, V = np.linalg.eigh(Cs)
    ds = np.clip(ds, 1e-9, None)
    Cs_inv_sqrt = V @ np.diag(1.0 / np.sqrt(ds)) @ V.T
    dt, Vt = np.linalg.eigh(Ct)
    dt = np.clip(dt, 1e-9, None)
    Ct_sqrt = Vt @ np.diag(np.sqrt(dt)) @ Vt.T
    A = Cs_inv_sqrt @ Ct_sqrt
    return (Xs - Xs.mean(0)) @ A + Xt.mean(0), Xt


def subspace_align(Xs, Xt, dim=None):
    dim = dim or min(20, Xs.shape[1] - 1, max(2, Xt.shape[0] // 4))
    ps = PCA(n_components=dim).fit(Xs)
    pt = PCA(n_components=dim).fit(Xt)
    Ps, Pt = ps.components_.T, pt.components_.T      # (d, dim)
    M = ps.components_ @ pt.components_.T            # (dim, dim)
    back = Ps @ M @ Pt.T                             # (d, d), keeps 49-dim output
    return (Xs - Xs.mean(0)) @ back + Xt.mean(0), Xt


def mmd_linear_align(Xs, Xt):
    """Linear-kernel MMD between two blobs reduces to matching first moments;
    we additionally match the per-feature scale (mean + variance matching)."""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    sd_s = Xs.std(0) + 1e-9
    sd_t = Xt.std(0) + 1e-9
    Zs = (Xs - mu_s) / sd_s * sd_t + mu_t
    return Zs, Xt


def per_domain_standardise(Xs, Xt):
    a = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    b = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9)
    return a, b


ALIGNERS = {
    "M2": ("coral", coral_align),
    "M3": ("sa", subspace_align),
    "M4": ("mmd", mmd_linear_align),
}


def align(name, Xs, Xt):
    if name == "M0":
        return Xs, Xt
    if name == "M1":
        return per_domain_standardise(Xs, Xt)
    if name in ALIGNERS:
        return ALIGNERS[name][1](Xs, Xt)
    raise KeyError(name)


def run_shallow(name, Xs, ys, Xt, seed=0, classes=None):
    if name in ("M0", "M1"):
        if name == "M0":
            sc = MC.source_scaler(Xs)
            Xs2, Xt2 = sc.transform(Xs), sc.transform(Xt)
        else:
            Xs2, Xt2 = align(name, np.asarray(Xs, float), np.asarray(Xt, float))
    else:
        sc = MC.source_scaler(Xs)
        base_s, base_t = sc.transform(Xs), sc.transform(Xt)
        Xs2, Xt2 = align(name, base_s, base_t)
    y_pred, proba, _ = MC.fit_predict(Xs2, ys, Xt2, seed=seed,
                                      n_classes=len(classes) if classes else None)
    return y_pred, proba
