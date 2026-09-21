import time

import numpy as np
import torch
import torch.nn as nn

import config as C
import models_common as MC


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DANNNet(nn.Module):
    def __init__(self, d_in, n_classes, hidden=(128, 64), p=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]), nn.ReLU())
        self.classifier = nn.Linear(hidden[1], n_classes)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden[1], 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, lam=1.0):
        f = self.encoder(x)
        y = self.classifier(f)
        d = self.discriminator(GRL.apply(f, lam))
        return y, d, f


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _lam(p):
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def train_dann(Xs, ys, Xt, classes, seed=0, epochs=200, batch=128, lr=1e-3,
               mode="dann", patience=30):
    dev = _device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = len(classes)
    d_in = Xs.shape[1]

    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=dev)
    ys_t = torch.tensor(ys, dtype=torch.long, device=dev)
    Xt_t = torch.tensor(Xt, dtype=torch.float32, device=dev)
    from feature_prep import compute_class_weight
    cw = compute_class_weight(ys, n_classes)
    alpha = torch.tensor([cw[i] for i in range(n_classes)], dtype=torch.float32, device=dev)
    alpha = alpha / alpha.sum() * n_classes

    model = DANNNet(d_in, n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=alpha)
    ce_d = nn.CrossEntropyLoss()

    t0 = time.time()
    n_s, n_t = Xs_t.shape[0], Xt_t.shape[0]
    best_state, best_mmd, bad = None, np.inf, 0
    for ep in range(epochs):
        model.train()
        lam = _lam(ep / max(epochs - 1, 1))
        perm_s = torch.randperm(n_s, device=dev)
        perm_t = torch.randperm(n_t, device=dev)
        for i in range(0, n_s, batch):
            idx_s = perm_s[i:i + batch]
            j = i % max(n_t - 1, 1)
            idx_t = perm_t[j: j + batch]
            m = int(min(idx_s.numel(), idx_t.numel()))
            if m < 2:
                continue
            idx_s, idx_t = idx_s[:m], idx_t[:m]
            xb = torch.cat([Xs_t[idx_s], Xt_t[idx_t]], 0)
            yb = ys_t[idx_s]
            db = torch.cat([torch.zeros(m, dtype=torch.long, device=dev),
                            torch.ones(m, dtype=torch.long, device=dev)])
            opt.zero_grad()
            logits, dom, feat = model(xb, lam)
            loss_cls = ce(logits[:m], yb)
            if mode == "dann":
                loss_dom = ce_d(dom, db)
            else:
                loss_dom = _lmmd(feat[:m], yb, feat[m:], logits[:m].argmax(1), n_classes)
            loss = loss_cls + lam * loss_dom
            loss.backward()
            opt.step()
        with torch.no_grad():
            model.eval()
            f_s = model.encoder(Xs_t)
            f_t = model.encoder(Xt_t)
            cur = float(torch.cdist(f_s[:800], f_t[:800]).mean())
        if cur < best_mmd - 1e-5:
            best_mmd, bad = cur, 0
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
        logits, _, _ = model(Xt_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba.argmax(1), proba, fit_s


def _lmmd(fs, ys, ft, yt_hat, n_classes):
    loss = fs.new_tensor(0.0)
    total = 0
    for c in range(n_classes):
        a = fs[ys == c]
        b = ft[yt_hat == c]
        if a.numel() == 0 or b.numel() == 0:
            continue
        loss = loss + ((a.mean(0) - b.mean(0)) ** 2).sum()
        total += 1
    return loss / max(total, 1)


def run_deep(name, Xs, ys, Xt, seed=0, classes=None, epochs=200):
    sc = MC.source_scaler(Xs)
    Xs2, Xt2 = sc.transform(Xs), sc.transform(Xt)
    mode = "dann" if name == "M8" else "lmmd"
    pred, proba, fit_s = train_dann(Xs2, ys, Xt2, classes, seed=seed,
                                    epochs=epochs, mode=mode)
    return pred, proba, fit_s
