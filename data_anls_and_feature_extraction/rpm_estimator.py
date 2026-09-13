import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import preprocess as pp
import spectral_kurtosis as sk

EPS = 1e-12
GRID_STEP = 0.01
N_HARM_HPS = 5
N_HARM_COMB = 4
HPS_FMAX = 600.0
PROM_THRESHOLD = 3.0


def _spectrum(x, fs):
    x = pp.detrend_dc(x)
    n = x.size
    A = np.abs(np.fft.rfft(x * np.hanning(n), n=n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return f, A


def _prominence(curve):
    c = np.asarray(curve, dtype=np.float64)
    med = np.median(c)
    sd = c.std() + EPS
    return float((c.max() - med) / sd)


def hps_curve(f, A, f_lo, f_hi, step=GRID_STEP, n_harm=N_HARM_HPS, f_max=HPS_FMAX):
    m = f <= f_max
    ff, AA = f[m], A[m] + EPS
    grid = np.arange(f_lo, f_hi + step, step)
    score = np.zeros_like(grid)
    for k in range(1, n_harm + 1):
        fk = k * grid
        ok = fk <= f_max
        score[ok] += np.log(np.interp(fk[ok], ff, AA))
    return grid, score


def comb_curve(x, fs, band, f_lo, f_hi, step=GRID_STEP,
               orders=np.arange(2.4, 5.01, 0.02), n_harm=N_HARM_COMB):
    from features_freq import envelope_spectrum
    _, E = envelope_spectrum(x, fs, band)
    f = np.fft.rfftfreq(E.size * 2 - 2, d=1.0 / fs)
    EE = E + EPS
    grid = np.arange(f_lo, f_hi + step, step)
    best = np.full_like(grid, -np.inf)
    for o in orders:
        score = np.zeros_like(grid)
        ok_all = True
        for k in range(1, n_harm + 1):
            fk = k * o * grid
            ok = fk <= f[-1]
            if not ok.any():
                ok_all = False
                break
            score[ok] += np.log(np.interp(fk[ok], f, EE))
        if ok_all:
            best = np.maximum(best, score)
    best = np.where(np.isfinite(best), best, 0.0)
    return grid, best


def _pick(grid, curve, prior_fr):
    med = np.median(curve)
    sd = curve.std() + EPS
    prom = (curve.max() - med) / sd
    i = int(np.argmax(curve))
    fr = float(grid[i])
    if abs(fr - prior_fr) / prior_fr > C.RPM_SEARCH_REL:
        fr = prior_fr
        prom = 0.0
    return fr, float(prom)


def estimate_target_rpm(rec, verbose=False):
    x, fs, _ = dl.load_signal(rec)
    prior_fr = C.RPM_TGT_PRIOR / 60.0
    f_lo = prior_fr * (1 - C.RPM_SEARCH_REL)
    f_hi = prior_fr * (1 + C.RPM_SEARCH_REL)

    f, A = _spectrum(x, fs)
    g1, s1 = hps_curve(f, A, f_lo, f_hi)
    fr_hps, prom_hps = _pick(g1, s1, prior_fr)

    band = sk.best_band(x, fs)
    g2, s2 = comb_curve(x, fs, band, f_lo, f_hi)
    fr_comb, prom_comb = _pick(g2, s2, prior_fr)

    if prom_hps >= PROM_THRESHOLD:
        fr, src = fr_hps, "hps"
    elif prom_comb >= PROM_THRESHOLD:
        fr, src = fr_comb, "comb"
    else:
        fr, src = prior_fr, "prior"

    out = dict(file_id=rec["file_id"], fs=fs, band_lo=band[0], band_hi=band[1],
               fr_prior=prior_fr, fr_hps=fr_hps, prom_hps=prom_hps,
               fr_comb=fr_comb, prom_comb=prom_comb,
               fr_final=fr, rpm_final=fr * 60.0, method=src)
    if verbose:
        print(f"  {rec['file_id']}: hps={fr_hps*60:.1f}rpm(p={prom_hps:.2f}) "
              f"comb={fr_comb*60:.1f}rpm(p={prom_comb:.2f}) -> {out['rpm_final']:.1f} ({src})")
    return out, (g1, s1, g2, s2)


def run_all(recs=None, verbose=True):
    recs = recs if recs is not None else dl.scan_target()
    rows, curves = [], {}
    for r in recs:
        row, cur = estimate_target_rpm(r, verbose=verbose)
        rows.append(row)
        curves[r["file_id"]] = cur
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "target_rpm.csv", index=False, encoding="utf-8")
    _plot(df, curves)
    return df


def _plot(df, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for fid, (g1, s1, g2, s2) in curves.items():
        axes[0].plot(g1 * 60, s1, lw=0.7, alpha=0.6)
        axes[1].plot(g2 * 60, s2, lw=0.7, alpha=0.6)
    axes[0].axvline(C.RPM_TGT_PRIOR, color="k", ls="--", lw=1)
    axes[1].axvline(C.RPM_TGT_PRIOR, color="k", ls="--", lw=1)
    axes[0].set_title("HPS score vs rpm (low-freq shaft harmonics)")
    axes[1].set_title("Envelope fault-comb score vs rpm")
    for a in axes:
        a.set_xlabel("rpm")
        a.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "target_rpm_search.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(df["file_id"], df["rpm_final"], color="steelblue")
    ax.axhline(C.RPM_TGT_PRIOR, color="r", ls="--", lw=1, label="prior 600 rpm")
    ax.set_ylabel("estimated rpm")
    ax.set_title(f"target rpm estimate (method: {df['method'].value_counts().to_dict()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "target_rpm_estimate.png", dpi=140)
    plt.close(fig)


def main():
    df = run_all()
    print(df[["file_id", "fr_final", "rpm_final", "method", "prom_hps", "prom_comb"]].to_string(index=False))
    print(f"[rpm_estimator] mean={df['rpm_final'].mean():.2f} std={df['rpm_final'].std():.3f} "
          f"rel_std={100*df['rpm_final'].std()/df['rpm_final'].mean():.2f}%")
    return df


if __name__ == "__main__":
    main()
