import numpy as np

import config as C
import models_common as MC
import methods_shallow as MS


def sinkhorn_assign(proba, prior=None, eps=None, n_iter=None):
    """Entropic optimal transport: rows = target windows (uniform marginal),
    columns = classes (marginal = class prior). Returns soft assignment T."""
    eps = eps or C.SINKHORN_EPS
    n_iter = n_iter or C.SINKHORN_ITERS
    P = np.clip(np.asarray(proba, dtype=np.float64), 1e-9, 1.0)
    C_ = -np.log(P)
    K = np.exp(-(C_ - C_.min()) / eps)
    n, c = K.shape
    a = np.full(n, 1.0 / n)
    b = np.asarray(prior if prior is not None else np.full(c, 1.0 / c), dtype=np.float64)
    b = b / b.sum()
    u = np.ones(n)
    v = np.ones(c)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    T = u[:, None] * K * v[None, :]
    return T / (T.sum(axis=1, keepdims=True) + 1e-12)


def _prior_vector(classes, prior):
    if prior is None:
        return None
    if prior == "balanced":
        return np.full(len(classes), 1.0 / len(classes))
    return np.asarray(prior, dtype=np.float64)


def balanced_assign(proba, prior):
    """Quota-constrained labelling: highest-confidence samples are assigned first
    but each class can only take `prior[c] * n` samples.  This *guarantees* the
    class proportions, which is what actually breaks the OR-collapse."""
    proba = np.asarray(proba, dtype=np.float64)
    n, c = proba.shape
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    quota = np.floor(prior * n).astype(int)
    rest = n - quota.sum()
    for i in np.argsort(-prior)[:rest]:
        quota[i] += 1
    order = np.argsort(-proba.max(1))
    labels = np.full(n, -1, dtype=int)
    for i in order:
        cand = np.where(quota > 0)[0]
        if len(cand) == 0:
            labels[i] = int(proba[i].argmax())
            continue
        k = cand[int(np.argmax(proba[i, cand]))]
        labels[i] = int(k)
        quota[k] -= 1
    return labels


def pseudo_label_train(Xs, ys, Xt, classes, seed=0, method="M5",
                       conf=None, max_iter=None, use_prior=None, files=None,
                       log=None):
    """Self-training.  M5 = confidence-filtered pseudo labels,
    M6 = CORAL alignment first, M7 = class-balanced OT assignment."""
    conf = C.PSEUDO_CONF_THRESHOLD if conf is None else conf
    max_iter = max_iter or C.PSEUDO_MAX_ITER
    use_prior = C.USE_CLASS_PRIOR if use_prior is None else use_prior

    if method == "M6":
        sc = MC.source_scaler(Xs)
        Xs_a, Xt_a = MS.coral_align(sc.transform(Xs), sc.transform(Xt))
    else:
        sc = MC.source_scaler(Xs)
        Xs_a, Xt_a = sc.transform(Xs), sc.transform(Xt)

    n_classes = len(classes)
    prior = _prior_vector(classes, "balanced") if use_prior else None

    X_cur, y_cur = Xs_a.copy(), ys.copy()
    w_cur = np.ones(len(ys))
    history = []
    clf = None
    for it in range(max_iter):
        clf = MC.make_classifier(y_cur, seed, n_classes)
        clf.fit(X_cur, y_cur, sample_weight=w_cur)
        proba_t = clf.predict_proba(Xt_a)
        if method == "M7" and prior is not None:
            T = sinkhorn_assign(proba_t, prior)
            take = np.ones(len(Xt_a), dtype=bool)
            new_y = T.argmax(1)
            new_w = T.max(1)
        else:
            conf_t = proba_t.max(1)
            take = conf_t >= conf
            new_y = proba_t.argmax(1)
            new_w = conf_t
        dist = np.bincount(new_y[take], minlength=n_classes) if take.any() else np.zeros(n_classes)
        history.append({"iter": it, "n_pseudo": int(take.sum()),
                        "class_dist": dist.tolist(),
                        "mean_conf": float(proba_t.max(1).mean())})
        if not take.any():
            break
        X_cur = np.vstack([Xs_a, Xt_a[take]])
        y_cur = np.concatenate([ys, new_y[take]])
        w_cur = np.concatenate([np.ones(len(ys)), new_w[take]])
        if len(history) >= 2 and history[-1]["n_pseudo"] == history[-2]["n_pseudo"]:
            break

    if clf is None:
        clf = MC.make_classifier(ys, seed, n_classes).fit(Xs_a, ys)
    proba = clf.predict_proba(Xt_a)
    if method == "M7" and prior is not None:
        # the transport plan gives a balanced SOFT assignment used for training,
        # and a quota-constrained HARD assignment used as the final prediction
        T = sinkhorn_assign(proba, prior)
        pred = balanced_assign(proba, prior)
        proba = T
    else:
        pred = np.asarray(clf.classes_)[proba.argmax(1)]
    if log is not None:
        log.extend([dict(method=method, **h) for h in history])
    return pred, proba, {"history": history, "Xs": Xs_a, "Xt": Xt_a}


def run_pseudo(name, Xs, ys, Xt, seed=0, classes=None, log=None):
    pred, proba, info = pseudo_label_train(Xs, ys, Xt, classes, seed=seed,
                                           method=name, log=log)
    return pred, proba
