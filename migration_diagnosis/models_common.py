import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import RobustScaler

import config as C
from feature_prep import compute_class_weight


def make_classifier(y=None, seed=0, n_classes=None):
    if n_classes is None:
        n_classes = int(np.max(y)) + 1
    cw = compute_class_weight(y, n_classes) if y is not None else None
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          class_weight=cw, random_state=seed)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def fit_predict(Xtr, ytr, Xte, seed=0, model=None, n_classes=None):
    clf = model if model is not None else make_classifier(ytr, seed, n_classes)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    pred = np.asarray(clf.classes_)[proba.argmax(1)]
    return pred, proba, clf


def source_scaler(Xs):
    return RobustScaler().fit(np.asarray(Xs, dtype=np.float64))


def evaluate(y_true, y_pred, classes):
    labels = sorted(C.CLASS_TO_IDX[c] for c in classes)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  zero_division=0)
    out = {"macro_f1": macro_f1(y_true, y_pred),
           "accuracy": float(accuracy_score(y_true, y_pred)),
           "n_classes": len(labels)}
    for i, lab in enumerate(labels):
        nm = C.CLASSES[int(lab)]
        out[f"f1_{nm}"] = float(f1[i])
        out[f"recall_{nm}"] = float(r[i])
    return out
