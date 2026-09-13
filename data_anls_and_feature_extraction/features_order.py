import numpy as np
from scipy.signal import find_peaks

import config as C
import preprocess as pp
from bearing_kinematics import bearing_orders

EPS = 1e-12
FAULT_KEYS = ["BPFO", "BPFI", "BSF", "FTF"]
ORDER_BANDS = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 32.0)]

GENERIC_ORDER_FEATURES = (
    [f"o_pk{i}_{s}" for i in range(C.N_TOP_PEAKS) for s in ("order", "amp", "ratio")]
    + ["o_entropy", "o_kurtosis", "o_crest", "o_gini"]
    + [f"o_bandfrac{i}" for i in range(len(ORDER_BANDS))]
    + ["o_shaft1", "o_shaft2", "o_shaft3", "o_shaft21"]
)

THEO_ORDER_FEATURES = (
    [f"o_theo_{k}_{m}_ratio" for k in FAULT_KEYS for m in range(1, 5)]
    + [f"o_theo_{k}_frac" for k in FAULT_KEYS]
)


def envelope_order_spectrum(x, fs, fr, band, m_per_rev=None, n_orders=None):
    if not np.isfinite(fr) or fr <= 0:
        return None, None
    m_per_rev = m_per_rev or C.ORDER_POINTS_PER_REV
    xb = pp.bandpass(x, fs, band)
    env = pp.hilbert_envelope(xb)
    env = pp.resample_angular(env, fs, fr, m_per_rev)
    if env is None or env.size < 16:
        return None, None
    n = env.size
    E = np.abs(np.fft.rfft(env * np.hanning(n), n=n)) ** 2
    orders = np.fft.rfftfreq(n, d=1.0 / m_per_rev)
    return orders, E


def _gini(v):
    v = np.sort(np.abs(v))
    n = v.size
    if n == 0 or v.sum() <= EPS:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def _band_peak(orders, E, center, tol_rel):
    lo, hi = center * (1 - tol_rel), center * (1 + tol_rel)
    m = (orders >= lo) & (orders <= hi)
    return float(E[m].max()) if m.any() else 0.0


def generic_order_features(orders, E):
    out = {}
    m = (orders >= 0.5) & (orders <= C.N_ORDERS)
    o = orders[m]
    e = E[m]
    tot = e.sum() + EPS
    p = e / tot

    pk_idx, _ = find_peaks(e, distance=2)
    if pk_idx.size == 0:
        pk_idx = np.array([int(np.argmax(e))])
    top = pk_idx[np.argsort(e[pk_idx])[::-1][:C.N_TOP_PEAKS]]
    med = np.median(e) + EPS
    amps = e[top]
    order_sorted = np.sort(amps)[::-1]
    order_pos = [o[i] for i in top[np.argsort(amps)[::-1]]]
    for i in range(C.N_TOP_PEAKS):
        if i < len(order_pos):
            out[f"o_pk{i}_order"] = float(order_pos[i])
            out[f"o_pk{i}_amp"] = float(order_sorted[i] / tot)
            out[f"o_pk{i}_ratio"] = float(order_sorted[i] / med)
        else:
            out[f"o_pk{i}_order"] = 0.0
            out[f"o_pk{i}_amp"] = 0.0
            out[f"o_pk{i}_ratio"] = 0.0

    out["o_entropy"] = float(-(p * np.log(p + EPS)).sum())
    mu = e.mean()
    sd = e.std() + EPS
    out["o_kurtosis"] = float((((e - mu) / sd) ** 4).mean())
    out["o_crest"] = float(e.max() / (mu + EPS))
    out["o_gini"] = _gini(e)

    for i, (lo, hi) in enumerate(ORDER_BANDS):
        mm = (o >= lo) & (o < hi)
        out[f"o_bandfrac{i}"] = float(e[mm].sum() / tot) if mm.any() else 0.0

    s = {k: _band_peak(orders, E, k, 0.05) / tot for k in (1.0, 2.0, 3.0)}
    out["o_shaft1"], out["o_shaft2"], out["o_shaft3"] = s[1.0], s[2.0], s[3.0]
    out["o_shaft21"] = s[2.0] / (s[1.0] + EPS)
    return out


def theoretical_order_features(orders, E, theo):
    out = {}
    m = (orders >= 0.5) & (orders <= C.N_ORDERS)
    o = orders[m]
    e = E[m]
    tot = e.sum() + EPS
    for k in FAULT_KEYS:
        f0 = theo[k]
        harm = 0.0
        for mm in range(1, 5):
            c = mm * f0
            if c > C.N_ORDERS:
                out[f"o_theo_{k}_{mm}_ratio"] = 0.0
                continue
            peak = _band_peak(o, e, c, C.PEAK_TOL_REL)
            wide = (o >= c * 0.85) & (o <= c * 1.15)
            bg = np.median(e[wide]) + EPS if wide.any() else EPS
            out[f"o_theo_{k}_{mm}_ratio"] = float(peak / bg)
            harm += peak
        out[f"o_theo_{k}_frac"] = float(harm / tot)
    return out


def order_features(x, fs, fr, band, bearing=None):
    orders, E = envelope_order_spectrum(x, fs, fr, band)
    out = {}
    if orders is None:
        out.update({k: np.nan for k in GENERIC_ORDER_FEATURES})
    else:
        out.update(generic_order_features(orders, E))
    if bearing is not None:
        if orders is None:
            out.update({k: np.nan for k in THEO_ORDER_FEATURES})
        else:
            out.update(theoretical_order_features(orders, E, bearing_orders(bearing)))
    return out
