import numpy as np
from scipy.stats import kurtosis, skew

EPS = 1e-12

TIME_FEATURES = [
    "t_mean_abs", "t_rms", "t_std", "t_var", "t_peak", "t_p2p",
    "t_kurtosis", "t_skewness", "t_crest", "t_impulse", "t_clearance", "t_shape",
]


def time_features(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    absx = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(absx.max())
    mean_abs = float(absx.mean())
    sd = float(x.std())
    mean_sqrt = float(np.sqrt(absx).mean())
    return {
        "t_mean_abs": mean_abs,
        "t_rms": rms,
        "t_std": sd,
        "t_var": float(x.var()),
        "t_peak": peak,
        "t_p2p": float(x.max() - x.min()),
        "t_kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        "t_skewness": float(skew(x, bias=False)),
        "t_crest": peak / (rms + EPS),
        "t_impulse": peak / (mean_abs + EPS),
        "t_clearance": peak / (mean_sqrt ** 2 + EPS),
        "t_shape": rms / (mean_abs + EPS),
    }


def batch_time_features(X):
    return np.array([[time_features(w)[k] for k in TIME_FEATURES] for w in X], dtype=np.float64)
