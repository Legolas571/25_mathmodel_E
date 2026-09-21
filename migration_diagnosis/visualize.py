import json

import numpy as np
import pandas as pd

import config as C
import data_io as dio

MAT_SRC = C.T1_OUT.parent.parent / "数据集" / "源域数据集"
MAT_TGT = C.T1_OUT.parent.parent / "数据集" / "目标域数据集"
SOURCE_REPS = [
    ("B", "12kHz_DE_data/B/0007/B007_0.mat", 12000),
    ("IR", "12kHz_DE_data/IR/0007/IR007_0.mat", 12000),
    ("OR", "12kHz_DE_data/OR/Centered/0007/OR007@6_0.mat", 12000),
    ("N", "48kHz_Normal_data/N_0.mat", 48000),
]
PAL = {"B": "tab:blue", "IR": "tab:orange", "OR": "tab:green", "N": "tab:red"}


def _load_mat_signal(path):
    from scipy.io import loadmat, whosmat
    names = [n for n, _s, _d in whosmat(str(path))]
    key = next((n for n in names if n.upper().endswith("_DE_TIME")), None)
    if key is None:
        key = next((n for n in names if n.upper().endswith("_TIME")), None)
    if key is None:
        key = next((n for n in names if not n.startswith("__")), names[0])
    rpmk = next((n for n in names if n.upper().endswith("RPM")), None)
    wanted = [key] + ([rpmk] if rpmk else [])
    md = loadmat(str(path), variable_names=wanted)
    x = np.asarray(md[key]).ravel().astype(np.float64)
    rpm = float(np.asarray(md[rpmk]).ravel()[0]) if rpmk and rpmk in md else np.nan
    return x, rpm


def _eos(x, fs, fr, band=(2000.0, 6000.0), m_per_rev=256, n_orders=26):
    from scipy.signal import butter, filtfilt, hilbert
    f1, f2 = band[0], min(band[1], fs / 2 * 0.98)
    if f2 - f1 < 50:
        xb = x
    else:
        b, a = butter(4, [f1 / (fs / 2), f2 / (fs / 2)], btype="band")
        xb = filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))
    env = np.abs(hilbert(xb))
    n = env.size
    n_out = int((n / fs) * fr * m_per_rev)
    if n_out < 2 * m_per_rev or not np.isfinite(fr) or fr <= 0:
        return None, None
    t_src = np.arange(n) / fs
    t_dst = np.arange(n_out) / (m_per_rev * fr)
    e = np.interp(t_dst, t_src, env)
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(e.size))) ** 2
    orders = np.fft.rfftfreq(e.size, d=1.0 / m_per_rev)
    m = orders <= n_orders
    o, E = orders[m], E[m]
    return o, E / (E.sum() + 1e-15)


def plot_eos_overlay():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import resample_poly

    src_curves, tgt_curves = {}, {}
    for cls, rel, fs in SOURCE_REPS:
        p = MAT_SRC / rel
        if not p.exists():
            continue
        try:
            x, rpm = _load_mat_signal(p)
            x = x[:int(0.5 * fs)]
            o, E = _eos(x, fs, rpm / 60.0)
            if o is not None:
                src_curves[cls] = (o, E)
        except Exception as e:
            print(f"  [vis] source {cls} skipped: {e}")

    rpm_t = pd.read_csv(C.TARGET_RPM)
    for _, row in rpm_t.iterrows():
        p = MAT_TGT / f"{row['file_id']}.mat"
        if not p.exists():
            continue
        try:
            x, _ = _load_mat_signal(p)
            o, E = _eos(x[:16000], 32000, row["fr_final"])
            if o is not None:
                tgt_curves[row["file_id"]] = (o, E)
        except Exception as e:
            print(f"  [vis] target {row['file_id']} skipped: {e}")

    n = len(tgt_curves)
    if not n:
        print("  [vis] eos_overlay skipped (no curves)")
        return
    fig, axes = plt.subplots(4, 4, figsize=(17, 11), sharex=True)
    axes = axes.ravel()
    for ax, (fid, (o, E)) in zip(axes, sorted(tgt_curves.items())):
        for cls, (so, sE) in src_curves.items():
            ax.plot(so, sE, lw=0.9, alpha=0.55, color=PAL[cls], label=f"src {cls}")
        ax.plot(o, E, lw=1.2, color="k", label=f"tgt {fid}")
        ax.set_title(f"target {fid}", fontsize=10)
        ax.set_xlim(0, 12)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle("Envelope order spectra: source class prototypes vs target segments")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eos_overlay_target.png", dpi=130)
    plt.close(fig)
    print(f"  [vis] eos_overlay_target.png ({n} target curves, "
          f"{len(src_curves)} source prototypes)")


def plot_tsne_before_after(cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import RobustScaler

    import methods_shallow as MS

    src = dio.load_source_features()
    tgt = dio.load_target_features()
    rs = np.random.RandomState(C.SEEDS[0])
    s = src.iloc[rs.choice(len(src), min(len(src), 1800), replace=False)]
    Xs = s[cols].to_numpy(float)
    Xt = tgt[cols].to_numpy(float)

    sc = RobustScaler().fit(Xs)
    Xs0, Xt0 = sc.transform(Xs), sc.transform(Xt)
    Xs1, Xt1 = MS.coral_align(Xs0, Xt0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (A, B, title) in zip(axes, [
            (Xs0, Xt0, "before adaptation"),
            (Xs1, Xt1, "after CORAL alignment")]):
        Z = np.vstack([A, B])
        emb = TSNE(n_components=2, perplexity=30, max_iter=800, init="pca",
                   random_state=C.SEEDS[0], learning_rate="auto", n_jobs=1).fit_transform(Z)
        n = len(A)
        for cls in C.CLASSES:
            m = (s["fault_type"] == cls).to_numpy()
            ax.scatter(emb[:n][m, 0], emb[:n][m, 1], s=6, alpha=0.5, c=PAL[cls],
                       label=f"src {cls}")
        ax.scatter(emb[n:, 0], emb[n:, 1], s=16, alpha=0.9, c="k", marker="x",
                   label="target")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=7, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "tsne_before_after.png", dpi=140)
    plt.close(fig)
    print("  [vis] tsne_before_after.png")


def plot_domain_distance(cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import RobustScaler

    import baseline as BL
    import methods_shallow as MS

    src = dio.load_source_features()
    tgt = dio.load_target_features()
    Xs, Xt = src[cols].to_numpy(float), tgt[cols].to_numpy(float)
    sc = RobustScaler().fit(Xs)
    Xs0, Xt0 = sc.transform(Xs), sc.transform(Xt)

    variants = {"raw": (Xs0, Xt0)}
    variants["M1 per-domain std"] = MS.per_domain_standardise(Xs0, Xt0)
    variants["M2 CORAL"] = MS.coral_align(Xs0, Xt0)
    a, b = MS.subspace_align(Xs0, Xt0)
    variants["M3 SA"] = (a, b)
    a, b = MS.mmd_linear_align(Xs0, Xt0)
    variants["M4 linMMD"] = (a, b)

    rows = []
    for name, (A, B) in variants.items():
        rows.append({"variant": name, "mmd": BL.mmd_rbf(A, B),
                     "a_distance": BL.a_distance(A, B)[0]})
    df = pd.DataFrame(rows)
    df.to_csv(C.RES_DIR / "domain_distance_variants.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(df["variant"], df["mmd"], color="teal")
    axes[0].set_ylabel("MMD (RBF)")
    axes[0].set_title("Domain distance before/after alignment")
    axes[1].bar(df["variant"], df["a_distance"], color="slateblue")
    axes[1].set_ylabel("A-distance")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "domain_distance.png", dpi=140)
    plt.close(fig)
    print("  [vis] domain_distance.png")
    print(df.round(4).to_string(index=False))


def plot_method_comparison():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = C.RES_DIR / "method_comparison.csv"
    if not p.exists():
        print("  [vis] method_comparison skipped (run proxy_eval first)")
        return
    d = pd.read_csv(p)
    piv = d.pivot(index="method", columns="proxy", values="macro_f1_mean")
    gain = d.pivot(index="method", columns="proxy", values="gain_vs_M0")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    piv.plot(kind="bar", ax=axes[0], colormap="tab10")
    axes[0].set_ylabel("macro-F1 (pseudo-target)")
    axes[0].set_title("Adaptation methods on proxy tasks")
    gain.plot(kind="bar", ax=axes[1], colormap="tab10")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].axhline(C.MIN_ACCEPT_GAIN, color="r", ls="--", lw=1,
                    label=f"accept threshold {C.MIN_ACCEPT_GAIN}")
    axes[1].set_ylabel("gain vs M0")
    axes[1].set_title("Relative gain over no-adaptation baseline")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "method_comparison.png", dpi=140)
    plt.close(fig)
    print("  [vis] method_comparison.png")


def plot_proxy_confusion():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import RobustScaler

    import methods_shallow as MS
    import models_common as MC
    import proxy_eval as PE

    if not (C.RES_DIR / "recommended_method.json").exists():
        print("  [vis] proxy confusion skipped")
        return
    rec = json.loads((C.RES_DIR / "recommended_method.json").read_text(encoding="utf-8"))
    best = rec["method"]
    proxy = rec["proxy_used"]
    src = dio.load_source_features()
    splits = dio.load_splits()
    cols = dio.feature_columns(src, C.FEATURE_MAIN)
    classes = C.PROXY_TASKS[proxy]["classes"]
    labels = sorted(C.CLASS_TO_IDX[c] for c in classes)
    cms = {"M0": np.zeros((len(labels),) * 2, dtype=int),
           best: np.zeros((len(labels),) * 2, dtype=int)}
    for fold in dio.proxy_folds(splits, proxy):
        tr, te = dio.load_proxy_split(src, splits, proxy, fold)
        Xs, ys, _ = dio.build_xy(tr, cols, classes)
        Xt, yt, _ = dio.build_xy(te, cols, classes)
        for meth in ("M0", best):
            pred, _, _ = PE.run_method(meth, Xs, ys, Xt, C.SEEDS[0], classes)
            cms[meth] += confusion_matrix(yt, pred, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    names = [C.CLASSES[i] for i in labels]
    for ax, meth in zip(axes, ("M0", best)):
        sns.heatmap(cms[meth], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=names, yticklabels=names)
        ax.set_title(f"{proxy} / {meth}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "cm_proxy_M0_vs_best.png", dpi=140)
    plt.close(fig)
    print(f"  [vis] cm_proxy_M0_vs_best.png (best={best})")


def plot_label_confidence():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = C.LABEL_DIR / "target_labels.csv"
    if not p.exists():
        print("  [vis] label confidence skipped")
        return
    d = pd.read_csv(p).sort_values("file_id")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(d["file_id"], d["confidence"], color="steelblue")
    axes[0].axhline(0.5, color="r", ls="--", lw=1, label="0.5 reference")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("confidence")
    axes[0].set_title("Target labelling confidence (0-1)")
    axes[0].legend(fontsize=8)

    mat = pd.DataFrame(0.0, index=sorted(C.CLASSES), columns=d["file_id"])
    for _, r in d.iterrows():
        mat.loc[r["label"], r["file_id"]] = r["confidence"]
    im = axes[1].imshow(mat.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_yticks(range(len(mat.index)))
    axes[1].set_yticklabels(mat.index)
    axes[1].set_xticks(range(len(mat.columns)))
    axes[1].set_xticklabels(mat.columns)
    axes[1].set_title("Assigned label x confidence")
    fig.colorbar(im, ax=axes[1], label="confidence")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "label_confidence.png", dpi=140)
    plt.close(fig)
    print("  [vis] label_confidence.png")
    print("  label distribution:", d["label"].value_counts().to_dict())


def plot_cluster_check():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    src = dio.load_source_features()
    tgt = dio.load_target_features()
    cols = dio.main_feature_columns(src, tgt)
    lab_p = C.LABEL_DIR / "target_labels.csv"
    if not lab_p.exists():
        print("  [vis] cluster check skipped")
        return
    lab = pd.read_csv(lab_p)
    X = tgt[cols].to_numpy(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    km = KMeans(n_clusters=4, n_init=10, random_state=C.SEEDS[0]).fit(Xs)
    Z = PCA(n_components=2, random_state=C.SEEDS[0]).fit_transform(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for c in range(4):
        m = km.labels_ == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=f"cluster {c}")
    axes[0].legend(fontsize=8)
    axes[0].set_title("KMeans clusters (target windows)")
    for cls in C.CLASSES:
        m = (tgt["file_id"].map(dict(zip(lab['file_id'], lab['label']))) == cls).to_numpy()
        axes[1].scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, c=PAL[cls], label=f"pred {cls}")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Predicted labels (target windows)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "cluster_vs_prediction.png", dpi=140)
    plt.close(fig)
    print("  [vis] cluster_vs_prediction.png")


def main():
    src = dio.load_source_features()
    tgt = dio.load_target_features()
    cols = dio.main_feature_columns(src, tgt)
    print("[visualize] generating figures ...")
    plot_method_comparison()
    plot_label_confidence()
    plot_tsne_before_after(cols)
    plot_domain_distance(cols)
    plot_proxy_confusion()
    plot_cluster_check()
    plot_eos_overlay()
    print(f"[visualize] figures in {C.FIG_DIR}")


if __name__ == "__main__":
    main()
