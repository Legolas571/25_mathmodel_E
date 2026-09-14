import warnings

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore")
TOPK = 12

RAW_DIM_TIME = {"t_mean_abs", "t_rms", "t_std", "t_var", "t_peak", "t_p2p"}


def _transferable(cols):
    out = []
    for c in cols:
        if c in RAW_DIM_TIME:
            continue
        if c.startswith("t_") or (c.startswith("o_") and not c.startswith("o_theo_")):
            out.append(c)
    return out


def load_features():
    src = pd.read_csv(C.OUT_DIR / "features_source.csv")
    tgt = pd.read_csv(C.OUT_DIR / "features_target.csv")
    cols = [c for c in _transferable(src.columns) if c in tgt.columns]
    cols = [c for c in cols if src[c].notna().any()]
    return src, tgt, cols


def _robust_scale(X):
    med = np.nanmedian(X, axis=0)
    iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    iqr = np.where(iqr < 1e-12, 1.0, iqr)
    return (X - med) / iqr


def plot_separability(src, cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    show = [c for c in ["t_kurtosis", "t_crest", "t_rms_rel", "o_entropy", "o_crest",
                        "o_shaft1", "o_bandfrac2", "o_pk0_order"] if c in cols]
    if not show:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), show):
        sns.boxplot(data=src, x="fault_type", y=c, order=C.CLASSES, ax=ax,
                    showfliers=False, color="lightsteelblue")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("Source: transferable feature separability by class (12k+48k)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "analysis_feature_boxplots.png", dpi=130)
    plt.close(fig)


def _embed(X, seed):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p = PCA(n_components=min(20, X.shape[1]), random_state=seed).fit_transform(X)
    ts = TSNE(n_components=2, perplexity=30, max_iter=800, init="pca",
              random_state=seed, learning_rate="auto", n_jobs=1)
    return p[:, :2], ts.fit_transform(p)


def plot_embeddings(src, tgt, cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(C.SEED)
    s = src
    if len(s) > 3000:
        s = s.iloc[rng.choice(len(s), 3000, replace=False)].reset_index(drop=True)
    Xs = _robust_scale(s[cols].to_numpy(float))
    pca_s, ts_s = _embed(Xs, C.SEED)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    pal = dict(zip(C.CLASSES, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    for cls in C.CLASSES:
        m = (s["fault_type"] == cls).to_numpy()
        if m.any():
            axes[0].scatter(pca_s[m, 0], pca_s[m, 1], s=6, alpha=0.6, c=pal[cls], label=cls)
    axes[0].set_title("Source PCA - by class")
    axes[0].legend(markerscale=2, fontsize=8)

    for fs, c in zip(sorted(s["fs"].unique()), ["tab:purple", "tab:brown"]):
        m = (s["fs"] == fs).to_numpy()
        axes[1].scatter(pca_s[m, 0], pca_s[m, 1], s=6, alpha=0.6, c=c, label=f"{fs//1000} kHz")
    axes[1].set_title("Source PCA - by sampling rate (confounding check)")
    axes[1].legend(markerscale=2, fontsize=8)

    Xt = _robust_scale(tgt[cols].to_numpy(float))
    n = min(len(s), 1500)
    both = np.vstack([Xs[:n], Xt])
    ts_all = _embed(both, C.SEED)[1]
    axes[2].scatter(ts_all[:n, 0], ts_all[:n, 1], s=6, alpha=0.5, c="tab:blue", label="source")
    axes[2].scatter(ts_all[n:, 0], ts_all[n:, 1], s=10, alpha=0.8, c="tab:red", label="target")
    axes[2].set_title("t-SNE source vs target (transferable features)")
    axes[2].legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "analysis_embeddings.png", dpi=130)
    plt.close(fig)
    return ts_s


def feature_rpm_corr(src, cols):
    from scipy.stats import spearmanr
    rows = []
    for c in cols:
        v = src[c].to_numpy(float)
        r = src["rpm"].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(r)
        if ok.sum() < 30:
            continue
        rho, p = spearmanr(v[ok], r[ok])
        rows.append(dict(feature=c, spearman_rho=rho, p_value=p,
                         group="time" if c.startswith("t_") else "order"))
    df = pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    df.to_csv(C.OUT_DIR / "feature_rpm_corr.csv", index=False, encoding="utf-8")
    return df


def feature_ranking(src, cols):
    from sklearn.feature_selection import mutual_info_classif
    X = np.nan_to_num(src[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = src["fault_type"].map(C.CLASS_TO_IDX).to_numpy()
    mi = mutual_info_classif(X, y, random_state=C.SEED, n_jobs=1)
    rows = []
    for j, c in enumerate(cols):
        v = X[:, j]
        mu = [v[y == i].mean() for i in range(len(C.CLASSES)) if (y == i).any()]
        sd = np.sqrt(np.mean([v[y == i].var() for i in range(len(C.CLASSES)) if (y == i).any()])) + 1e-12
        fisher = float(np.var(mu) / (sd ** 2))
        rows.append(dict(feature=c, mutual_info=float(mi[j]), fisher_ratio=fisher,
                         group="time" if c.startswith("t_") else "order"))
    df = pd.DataFrame(rows).sort_values("mutual_info", ascending=False)
    df.to_csv(C.OUT_DIR / "feature_ranking.csv", index=False, encoding="utf-8")
    return df


def domain_shift(src, tgt, cols, max_n=2000):
    from scipy.stats import ks_2samp
    rng = np.random.RandomState(C.SEED)
    s = src.sample(min(len(src), max_n), random_state=C.SEED)
    t = tgt.sample(min(len(tgt), max_n), random_state=C.SEED)
    rows = []
    for c in cols:
        a = s[c].to_numpy(float)
        b = t[c].to_numpy(float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 20 or b.size < 20:
            continue
        ks, p = ks_2samp(a, b)
        rows.append(dict(feature=c, ks_stat=float(ks), ks_p=float(p),
                         src_median=float(np.median(a)), tgt_median=float(np.median(b))))
    df = pd.DataFrame(rows).sort_values("ks_stat")
    df.to_csv(C.OUT_DIR / "domain_shift.csv", index=False, encoding="utf-8")
    _plot_shift(df)
    return df


def _plot_shift(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.head(TOPK)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["feature"], d["ks_stat"], color="teal")
    ax.set_xlabel("KS statistic (lower = more domain-invariant)")
    ax.set_title("Least domain-shifted transferable features (source vs target)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "analysis_domain_shift.png", dpi=130)
    plt.close(fig)


def _plot_rpm_corr(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.concat([df[df["group"] == "order"].head(10), df[df["group"] == "time"].head(10)])
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:green" if g == "order" else "tab:gray" for g in d["group"]]
    ax.barh(d["feature"], d["spearman_rho"].abs(), color=colors)
    ax.set_xlabel("|Spearman rho| with rpm (lower = more speed-invariant)")
    ax.set_title("Speed dependence of features (green = order features)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "analysis_rpm_corr.png", dpi=130)
    plt.close(fig)


def main():
    src, tgt, cols = load_features()
    print(f"[analysis_report] source windows={len(src)}  target windows={len(tgt)}  "
          f"transferable features={len(cols)}")
    plot_separability(src, cols)
    corr = feature_rpm_corr(src, cols)
    rank = feature_ranking(src, cols)
    shift = domain_shift(src, tgt, cols)
    _plot_rpm_corr(corr)
    plot_embeddings(src, tgt, cols)

    summary = []
    add = summary.append
    add("=" * 74)
    add("FEATURE ANALYSIS SUMMARY")
    add("=" * 74)
    add(f"source windows {len(src)} | target windows {len(tgt)} | features {len(cols)}")
    add("")
    add("[top-10 by mutual information]")
    add(rank.head(10).to_string(index=False))
    add("")
    add("[order vs time: mean |rho| with rpm]  (lower is better for transfer)")
    add(corr.groupby("group")["spearman_rho"].apply(lambda s: s.abs().mean()).to_string())
    add("")
    add("[top-10 most domain-invariant features (lowest KS)]")
    add(shift.head(10).to_string(index=False))
    add("")
    add("[top-10 most domain-shifted features (highest KS)]")
    add(shift.tail(10).to_string(index=False))
    add("=" * 74)
    text = "\n".join(summary)
    (C.OUT_DIR / "analysis_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return summary


if __name__ == "__main__":
    main()
