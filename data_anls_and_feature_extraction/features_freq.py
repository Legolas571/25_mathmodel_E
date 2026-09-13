import numpy as np
from scipy.stats import kurtosis

import config as C
import preprocess as pp
from bearing_kinematics import bearing_freqs

EPS = 1e-12
FAULT_KEYS = ["BPFO", "BPFI", "BSF", "FTF"]

FREQ_FEATURES = (
    ["f_centroid", "f_msf", "f_rmsf", "f_variance", "f_peak_freq",
     "f_entropy", "f_spec_kurt", "f_env_log_energy"]
    + [f"f_{k}_ratio1" for k in FAULT_KEYS]
    + [f"f_{k}_harm_frac" for k in FAULT_KEYS]
)


def amplitude_spectrum(x, fs):
    x = pp.detrend_dc(x)
    n = x.size
    w = np.hanning(n)
    A = np.abs(np.fft.rfft(x * w, n=n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return f, A


def spectrum_stats(f, A):
    P = A ** 2
    tot = P.sum() + EPS
    Pn = P / tot
    centroid = float((f * Pn).sum())
    msf = float((f ** 2 * Pn).sum())
    var = float(((f - centroid) ** 2 * Pn).sum())
    rmsf = float(np.sqrt(msf))
    entropy = float(-(Pn * np.log(Pn + EPS)).sum())
    return {
        "f_centroid": centroid,
        "f_msf": msf,
        "f_rmsf": rmsf,
        "f_variance": var,
        "f_peak_freq": float(f[np.argmax(A)]),
        "f_entropy": entropy,
        "f_spec_kurt": float(kurtosis(A, fisher=True, bias=False)),
    }


def envelope_spectrum(x, fs, band=None):
    xb = pp.bandpass(x, fs, band) if band is not None else pp.detrend_dc(x)
    env = pp.hilbert_envelope(xb)
    env = env - env.mean()
    n = env.size
    w = np.hanning(n)
    E = np.abs(np.fft.rfft(env * w, n=n)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return f, E


def _band_peak(f, E, center, tol_rel):
    lo, hi = center * (1 - tol_rel), center * (1 + tol_rel)
    m = (f >= lo) & (f <= hi)
    return float(E[m].max()) if m.any() else 0.0


def fault_freq_ratios(f, E, freqs, tol_rel=0.02):
    out = {}
    total = E.sum() + EPS
    for k in FAULT_KEYS:
        f0 = freqs[k]
        peak = _band_peak(f, E, f0, tol_rel) if f0 < f[-1] else 0.0
        harm = 0.0
        for m_ in range(1, 5):
            if m_ * f0 < f[-1]:
                harm += _band_peak(f, E, m_ * f0, tol_rel)
        out[f"f_{k}_ratio1"] = peak / total
        out[f"f_{k}_harm_frac"] = harm / total
    return out


def freq_features(x, fs, band, bearing=None, rpm=np.nan):
    f, A = amplitude_spectrum(x, fs)
    out = spectrum_stats(f, A)
    _, E = envelope_spectrum(x, fs, band)
    out["f_env_log_energy"] = float(np.log10(E.sum() + EPS))
    for k in FAULT_KEYS:
        out.setdefault(f"f_{k}_ratio1", np.nan)
        out.setdefault(f"f_{k}_harm_frac", np.nan)
    if bearing is not None and np.isfinite(rpm):
        out.update(fault_freq_ratios(f, E, bearing_freqs(bearing, rpm)))
    return out


def verify_mechanism(recs, n_files=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import data_loader as dl
    import spectral_kurtosis as sk

    pick = [r for r in recs if r["group"] == "12k_DE" and r["fault_type"] in ("B", "IR", "OR")]
    pick = pick[:n_files]
    for r in pick:
        x, fs, rpm = dl.load_signal(r)
        w, _ = pp.segment(x, fs, C.WIN_SEC_FEATURE, 0.0)
        band = sk.best_band(w[0], fs)
        f, E = envelope_spectrum(w[0], fs, band)
        freqs = bearing_freqs(r["bearing"], rpm)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.semilogy(f, E + EPS, lw=0.6)
        for k, v in freqs.items():
            for m_ in range(1, 4):
                if m_ * v < f[-1]:
                    ax.axvline(m_ * v, color="r", ls="--", lw=0.6, alpha=0.6)
        ax.set_xlim(0, min(1000, f[-1]))
        ax.set_title(f"{r['file_id']}  band {band[0]:.0f}-{band[1]:.0f} Hz  rpm {rpm:.0f}")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("envelope PSD")
        fig.tight_layout()
        fig.savefig(C.FIG_DIR / f"mechanism_env_spectrum_{r['file_id'].replace('/', '_')}.png", dpi=130)
        plt.close(fig)
    print(f"[features_freq] mechanism figures: {len(pick)}")
