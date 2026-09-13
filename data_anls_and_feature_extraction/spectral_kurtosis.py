import numpy as np
from scipy.signal import stft

import config as C


def kurtogram(x, fs, nperseg=256):
    nperseg = int(min(nperseg, max(64, x.size // 8)))
    f, _t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                    window="hann", detrend=False, boundary=None)
    P = np.abs(Z) ** 2
    mu = P.mean(axis=1, keepdims=True)
    sd = P.std(axis=1, keepdims=True) + 1e-15
    K = (((P - mu) / sd) ** 4).mean(axis=1)
    return f, K


def _resolve_band(f, K, fs, min_bw, f_lo, f_hi):
    nyq = fs / 2.0
    lo_lim = max(f_lo, 0.02 * nyq)
    hi_lim = min(f_hi, 0.95 * nyq)
    mask = (f >= lo_lim) & (f <= hi_lim)
    if not mask.any():
        return list(C.DEFAULT_BAND)
    fc = f[mask][np.argmax(K[mask])]
    kmax = K[mask].max()
    thr = 0.5 * kmax
    keep = mask & (K >= thr)
    idx = np.where(keep)[0]
    f1, f2 = f[idx].min(), f[idx].max()
    if f2 - f1 < min_bw:
        f1, f2 = fc - min_bw / 2.0, fc + min_bw / 2.0
    f1 = max(f1, 1.0)
    f2 = min(f2, nyq * 0.98)
    if f2 - f1 < 20.0:
        return list(C.DEFAULT_BAND)
    return [float(f1), float(f2)]


def best_band(x, fs, min_bw=None):
    min_bw = min_bw or C.MIN_BAND_WIDTH
    try:
        f, K = kurtogram(x, fs)
        return _resolve_band(f, K, fs, min_bw, 0.02 * fs, 0.45 * fs)
    except Exception:
        return list(C.DEFAULT_BAND)


def main():
    import data_loader as dl
    import preprocess as pp
    recs = [r for r in dl.scan_source() if r["group"] == "12k_DE" and r["fault_type"] == "B"][:3]
    for r in recs:
        x, fs, _ = dl.load_signal(r)
        w, _ = pp.segment(x, fs, C.WIN_SEC_FEATURE, 0.0)
        band = best_band(w[0], fs)
        print(f"{r['file_id']:<40s} fs={fs}  band={band[0]:.0f}-{band[1]:.0f} Hz")


if __name__ == "__main__":
    main()
