import numpy as np
from scipy.ndimage import zoom

import config as C


def _morl_scales(fs, f_lo, f_hi):
    fc = 0.8125
    s_hi = fc / (f_lo / fs)
    s_lo = fc / (f_hi / fs)
    n = 24
    return np.linspace(max(s_lo, 1.0), max(s_hi, 2.0), n)


def stft_image(x, fs, nperseg=256):
    from scipy.signal import stft
    nperseg = int(min(nperseg, max(32, x.size // 8)))
    _f, _t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    return np.log1p(np.abs(Z))


def cwt_image(x, fs, f_lo=500.0, f_hi=None):
    import pywt
    f_hi = f_hi or min(0.45 * fs, 6000.0)
    scales = _morl_scales(fs, f_lo, f_hi)
    coef, _freqs = pywt.cwt(x, scales, "morl", sampling_period=1.0 / fs)
    return np.log1p(np.abs(coef))


def to_fixed_size(img, size=64):
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    zy = size / img.shape[0]
    zx = size / img.shape[1]
    out = zoom(img, (zy, zx), order=1)
    out = out[:size, :size]
    if out.shape != (size, size):
        pad = np.zeros((size, size), dtype=np.float64)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    out = out - out.mean()
    s = out.std()
    if s > 1e-12:
        out = out / s
    return out.astype(np.float32)


def batch_timefreq(windows, fs, size=64, mode="cwt"):
    out = np.zeros((len(windows), size, size), dtype=np.float32)
    for i, w in enumerate(windows):
        img = cwt_image(w, fs) if mode == "cwt" else stft_image(w, fs)
        out[i] = to_fixed_size(img, size)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import data_loader as dl

    src = dl.scan_source()
    tgt = dl.scan_target()
    picks = []
    for cls in C.CLASSES:
        c = next((r for r in src if r["fault_type"] == cls and r["group"] == "12k_DE"), None)
        if c is None:
            c = next((r for r in src if r["fault_type"] == cls), None)
        if c is not None:
            picks.append(c)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, rec in zip(axes[0], picks):
        x, fs, _ = dl.load_signal(rec)
        img = to_fixed_size(cwt_image(x[:int(0.5 * fs)], fs, f_hi=0.45 * fs), 64)
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"src {rec['fault_type']} ({fs//1000}kHz)")
        ax.set_xticks([]); ax.set_yticks([])
    for ax, rec in zip(axes[1], tgt[:4]):
        x, fs, _ = dl.load_signal(rec)
        img = to_fixed_size(cwt_image(x[:int(0.5 * fs)], fs, f_hi=0.45 * fs), 64)
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"tgt {rec['file_id']} ({fs//1000}kHz)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CWT time-frequency maps (log magnitude, 64x64, z-scored)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "timefreq_samples.png", dpi=140)
    plt.close(fig)

    demo = {}
    for rec in picks + tgt[:4]:
        x, fs, _ = dl.load_signal(rec)
        demo[rec["file_id"].replace("/", "_")] = to_fixed_size(
            cwt_image(x[:int(0.5 * fs)], fs, f_hi=0.45 * fs), 64)
    np.savez_compressed(C.OUT_DIR / "timefreq_demo.npz", **demo)
    print(f"[features_timefreq] pywt demo images: {len(demo)} -> timefreq_samples.png / timefreq_demo.npz")


if __name__ == "__main__":
    main()
