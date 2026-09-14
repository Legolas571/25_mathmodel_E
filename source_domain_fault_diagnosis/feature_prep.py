import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

import config as C


class FeaturePipeline:
    def __init__(self, scaler=None, winsorize=None, pca=None):
        self.scaler_name = scaler or C.SCALER
        self.winsorize = winsorize if winsorize is not None else C.WINSORIZE
        self.pca_components = pca
        self.scaler = None
        self.lo = None
        self.hi = None
        self.pca = None
        self.columns_ = None

    def fit(self, X, columns=None):
        X = np.asarray(X, dtype=np.float64)
        self.columns_ = list(columns) if columns is not None else None
        if self.winsorize:
            self.lo = np.percentile(X, self.winsorize[0] * 100, axis=0)
            self.hi = np.percentile(X, self.winsorize[1] * 100, axis=0)
        self.scaler = (RobustScaler() if self.scaler_name == "robust" else StandardScaler())
        self.scaler.fit(self._clip(X))
        if self.pca_components:
            from sklearn.decomposition import PCA
            Z = self.scaler.transform(self._clip(X))
            self.pca = PCA(n_components=self.pca_components, random_state=C.SEEDS[0]).fit(Z)
        return self

    def _clip(self, X):
        if self.lo is None:
            return X
        return np.clip(X, self.lo, self.hi)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = self.scaler.transform(self._clip(X))
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return Z

    def fit_transform(self, Xtr, Xte, columns=None):
        self.fit(Xtr, columns=columns)
        return self.transform(Xtr), self.transform(Xte)

    def state(self):
        return {
            "scaler_name": self.scaler_name,
            "winsorize": self.winsorize,
            "columns": self.columns_,
            "center": getattr(self.scaler, "center_", None),
            "scale": getattr(self.scaler, "scale_", None),
            "mean": getattr(self.scaler, "mean_", None),
        }


def compute_class_weight(y, n_classes=None):
    y = np.asarray(y)
    n_classes = n_classes or int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return {i: float(w[i]) for i in range(n_classes)}


def torch_class_weights(y, n_classes=None):
    import torch
    w = compute_class_weight(y, n_classes)
    return torch.tensor([w[i] for i in sorted(w)], dtype=torch.float32)
