import numpy as np
from scipy.signal import butter, filtfilt, hilbert

import config as C


def detrend_dc(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return x - x.mean()
    t = np.arange(n, dtype=np.float64)
    coef = np.polyfit(t, x, 1)
    return x - np.polyval(coef, t)


def bandpass(x, fs, band, order=4):
    f1, f2 = float(band[0]), float(band[1])
    nyq = fs / 2.0
    f1 = max(f1, 1.0)
    f2 = min(f2, nyq * 0.98)
    if f2 - f1 < 20.0:
        return x
    b, a = butter(order, [f1 / nyq, f2 / nyq], btype="band")
    pad = min(3 * max(len(a), len(b)), x.size - 1)
    if pad < 1:
        return x
    return filtfilt(b, a, x, padlen=pad)


def hilbert_envelope(x):
    return np.abs(hilbert(x))


def resample_angular(x, fs, fr, m_per_rev=None):
    if not np.isfinite(fr) or fr <= 0:
        return None
    m_per_rev = m_per_rev or C.ORDER_POINTS_PER_REV
    n = x.size
    n_out = int((n / fs) * fr * m_per_rev)
    if n_out < 2 * m_per_rev:
        return None
    t_src = np.arange(n, dtype=np.float64) / fs
    t_dst = np.arange(n_out, dtype=np.float64) / (m_per_rev * fr)
    out = np.interp(t_dst, t_src, x)
    return out - out.mean()


def normalize(x, mode="zscore"):
    x = np.asarray(x, dtype=np.float64)
    if mode == "zscore":
        s = x.std()
        return (x - x.mean()) / s if s > 1e-12 else x - x.mean()
    m = np.max(np.abs(x))
    return x / m if m > 1e-12 else x


def segment(x, fs, win_sec, overlap):
    win = int(round(win_sec * fs))
    step = max(1, int(round(win * (1.0 - overlap))))
    if x.size < win:
        return np.empty((0, win)), np.empty(0, dtype=int)
    starts = np.arange(0, x.size - win + 1, step, dtype=int)
    idx = starts[:, None] + np.arange(win)[None, :]
    return x[idx], starts


def make_windows(x, fs, win_sec, overlap, file_id=None):
    w, starts = segment(x, fs, win_sec, overlap)
    meta = dict(file_id=file_id, starts=starts, win_len=w.shape[1] if w.size else 0)
    return w, meta
