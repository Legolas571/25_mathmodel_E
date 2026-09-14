import time

import numpy as np
import torch
import torch.nn as nn

import config as C


class MLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=(256, 128), p_drop=0.3):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        p = logp.exp()
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = p.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _val_split(X, y, files, seed, frac=0.15):
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(files)))
    if len(uniq) < 6:
        return None
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * frac))
    val_files = set(uniq[:n_val])
    m = np.array([f in val_files for f in files])
    if m.all() or (~m).all() or len(np.unique(y[m])) < 2:
        return None
    return ~m, m


def train_mlp(Xtr, ytr, Xte, n_classes, seed, class_weight=None,
              files=None, epochs=120, patience=20, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)

    if files is not None:
        sp = _val_split(Xtr, ytr, np.asarray(files), seed)
    else:
        sp = None
    if sp is None:
        tr_idx = np.ones(len(ytr), dtype=bool)
        va_idx = np.zeros(len(ytr), dtype=bool)
    else:
        tr_idx, va_idx = sp

    alpha = None
    if class_weight is not None:
        alpha = torch.tensor([class_weight[i] for i in range(n_classes)],
                             dtype=torch.float32)
        alpha = alpha / alpha.sum() * n_classes

    model = MLP(Xtr.shape[1], n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = FocalLoss(alpha=alpha, gamma=C.FOCAL_GAMMA)

    Xa = Xtr_t[tr_idx]
    ya = ytr_t[tr_idx]
    Xv = Xtr_t[va_idx] if va_idx.any() else None
    yv = ytr_t[va_idx] if va_idx.any() else None

    t0 = time.time()
    best_state, best_score, bad = None, -1.0, 0
    n = Xa.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xa[idx]), ya[idx])
            loss.backward()
            opt.step()
        sched.step()
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                pred = model(Xv).argmax(1)
            score = _macro_f1(yv.cpu().numpy(), pred.cpu().numpy())
            if score > best_score + 1e-4:
                best_score, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    fit_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba.argmax(1), proba, fit_s


def _macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
