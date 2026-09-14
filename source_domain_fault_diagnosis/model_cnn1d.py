import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly

import config as C
import data_io as dio


class WDCNN(nn.Module):
    def __init__(self, n_classes, win_len=1536):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 64, stride=16, padding=24), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4, 100), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(100, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def load_windows():
    p = C.WINDOWS_SRC
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run task 1 with: python build_dataset.py --save-windows")
    z = np.load(p, allow_pickle=False)
    ids = z["file_ids"]
    labels = z["labels"]
    fss = z["fs"]
    groups = z["groups"]
    out = []
    for i, fid in enumerate(ids):
        W = z[f"w{i}"].astype(np.float32)
        fs = int(fss[i])
        if fs != C.CNN_RESAMPLE_FS:
            W = resample_poly(W, C.CNN_RESAMPLE_FS, fs, axis=1).astype(np.float32)
        if W.shape[1] < C.CNN_WIN_POINTS:
            pad = C.CNN_WIN_POINTS - W.shape[1]
            W = np.pad(W, ((0, 0), (0, pad)))
        W = W[:, :C.CNN_WIN_POINTS]
        mu = W.mean(axis=1, keepdims=True)
        sd = W.std(axis=1, keepdims=True) + 1e-8
        W = (W - mu) / sd
        out.append((fid, labels[i], groups[i], W))
    print(f"[cnn1d] loaded {len(out)} files, "
          f"{sum(w.shape[0] for _, _, _, w in out)} windows, "
          f"{C.CNN_WIN_POINTS} points each (resampled to {C.CNN_RESAMPLE_FS} Hz)")
    return out


def _tensor(windows, device):
    return torch.tensor(windows, dtype=torch.float32, device=device).unsqueeze(1)


def train_fold(train_w, train_y, test_w, test_y, n_classes, seed, epochs=40, batch=128, lr=1e-3):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xtr = _tensor(train_w, dev)
    ytr = torch.tensor(train_y, dtype=torch.long, device=dev)
    Xte = _tensor(test_w, dev)
    model = WDCNN(n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float32)
    w = torch.tensor(len(train_y) / (n_classes * np.maximum(counts, 1)), device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    t0 = time.time()
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    fit_s = time.time() - t0
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], 512):
            preds.append(model(Xte[i:i + 512]).argmax(1).cpu().numpy())
    return np.concatenate(preds), fit_s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="P1,P2")
    ap.add_argument("--tag", default="cnn1d")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args(argv)

    data = load_windows()
    by_file = {fid: (lab, W) for fid, lab, _g, W in data}
    splits = dio.load_splits()
    out = C.RES_DIR / f"runs_{args.tag}.csv"
    if out.exists():
        out.unlink()
    header = True
    for p in args.protocols.split(","):
        classes = C.PROTOCOL_CLASSES[p]
        for seed in [int(s) for s in args.seeds.split(",")]:
            for fold, split in sorted(splits[p].items()):
                trf = [f for f in split["train"] if f in by_file]
                tef = [f for f in split["test"] if f in by_file]
                trf = [f for f in trf if by_file[f][0] in classes]
                tef = [f for f in tef if by_file[f][0] in classes]
                if not trf or not tef:
                    continue
                Xtr = np.concatenate([by_file[f][1] for f in trf])
                ytr = np.concatenate([[C.CLASS_TO_IDX[by_file[f][0]]] * by_file[f][1].shape[0]
                                      for f in trf])
                Xte = np.concatenate([by_file[f][1] for f in tef])
                yte = np.concatenate([[C.CLASS_TO_IDX[by_file[f][0]]] * by_file[f][1].shape[0]
                                      for f in tef])
                y_pred, fit_s = train_fold(Xtr, ytr, Xte, yte, len(classes), seed,
                                           epochs=args.epochs)
                m = models_metrics(yte, y_pred)
                row = {"protocol": p, "feature_set": "raw_signal", "model": "cnn1d",
                       "seed": seed, "fold": fold, "n_train_files": len(trf),
                       "n_test_files": len(tef), "n_train_windows": len(ytr),
                       "n_test_windows": len(yte), "n_features": C.CNN_WIN_POINTS,
                       "fit_seconds": fit_s, "predict_ms_per_1k": np.nan}
                row.update(m)
                import train_eval as TE
                import pandas as _pd
                _pd.DataFrame([row]).reindex(columns=TE.RESULT_COLS).to_csv(
                    out, mode="a", header=header, index=False, encoding="utf-8")
                header = False
                print(f"  {p}/{fold}: macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} "
                      f"({fit_s:.1f}s, {len(ytr)} train windows)")
    print(f"[cnn1d] wrote {out}")
    return out


def models_metrics(y_true, y_pred):
    import models as M
    m = M.metrics(y_true, y_pred, np.unique(y_true))
    return {k: v for k, v in m.items() if not k.startswith("_")}


if __name__ == "__main__":
    main()
