import time

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import config as C

SKLEARN_MODELS = ["lr", "svm", "rf", "hgb"]
ALL_MODELS = SKLEARN_MODELS + ["mlp", "cnn1d", "vote"]


def supports_scaling(name):
    return name not in ("rf", "hgb")


def make_model(name, n_classes, seed=0, class_weight=None):
    cw = class_weight
    if name == "lr":
        return LogisticRegression(max_iter=3000, class_weight=cw, random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                   class_weight=cw, cache_size=800, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight=cw or "balanced_subsample",
                                      n_jobs=C.N_JOBS, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              class_weight=cw, random_state=seed)
    raise KeyError(f"unknown model {name}")


def make_vote(estimators, weights=None):
    return VotingClassifier(estimators=estimators, voting="soft", weights=weights)


def fit_predict(model, Xtr, ytr, Xte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xte)
    else:
        d = model.decision_function(Xte)
        d = np.atleast_2d(d)
        if d.shape[0] != Xte.shape[0]:
            d = d.T
        e = np.exp(d - d.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
    y_pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
    return y_pred, proba, fit_s


def metrics(y_true, y_pred, labels):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    labels = sorted(set(labels))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                      zero_division=0)),
        "n_classes": len(labels),
    }
    for i, lab in enumerate(labels):
        nm = C.CLASSES[int(lab)]
        out[f"precision_{nm}"] = float(p[i])
        out[f"recall_{nm}"] = float(r[i])
        out[f"f1_{nm}"] = float(f1[i])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["_cm"] = cm
    out["_labels"] = list(labels)
    return out
