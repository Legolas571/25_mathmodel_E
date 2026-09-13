import numpy as np
import pandas as pd

import config as C
import data_loader as dl


def build_table(recs):
    df = pd.DataFrame(dl.record_table(recs))
    rpms = []
    for r in recs:
        rpm = np.nan
        src = "none"
        if r.get("rpm_var"):
            _, _, v = dl.load_signal(r)
            if np.isfinite(v):
                rpm, src = v, "var"
        if not np.isfinite(rpm) and r.get("rpm_filename"):
            rpm, src = float(r["rpm_filename"]), "filename"
        rpms.append((rpm, src))
    df["rpm"] = [p[0] for p in rpms]
    df["rpm_source"] = [p[1] for p in rpms]
    df["is_irregular"] = ~df["n_points"].isin([96000, 192000, 384000, 256000])
    df["subset_tags"] = [";".join(dl.subset_of(r)) for r in recs]
    return df


def summarize(src, tgt):
    lines = []
    add = lines.append
    add("=" * 70)
    add("DATA AUDIT SUMMARY")
    add("=" * 70)
    add(f"source files: {len(src)}   target files: {len(tgt)}")

    add("")
    add("[1] class x group (source)")
    ct = pd.crosstab(src["fault_type"], src["group"])
    add(ct.to_string())

    add("")
    add("[2] class x sampling rate  <-- evidence of N-only-at-48k confounding")
    tmp = src.copy()
    tmp["fs_khz"] = (tmp["fs"] / 1000).astype(int).astype(str) + "kHz"
    add(pd.crosstab(tmp["fault_type"], tmp["fs_khz"]).to_string())

    add("")
    add("[3] class x load")
    add(pd.crosstab(src["fault_type"], src["load"]).to_string())

    add("")
    add("[4] class x fault size")
    add(pd.crosstab(src["fault_type"], src["fault_size"].fillna("-")).to_string())

    add("")
    add("[5] OR mounting position")
    orc = src[src["fault_type"] == "OR"]["or_position"].dropna().astype(int).value_counts().sort_index()
    add("\n".join(f"    {int(k)} o'clock ({C.OR_POSITION.get(int(k), '?')}): {v}" for k, v in orc.items()))

    add("")
    add("[6] rpm distribution (source)")
    add(src["rpm"].describe().to_string())
    add("rpm source counts: " + str(src["rpm_source"].value_counts().to_dict()))

    add("")
    add("[7] channel completeness")
    add(src["channels"].value_counts().to_string())

    irr = src[src["is_irregular"]]
    add("")
    add(f"[8] irregular-length files: {len(irr)}")
    for _, r in irr.iterrows():
        add(f"    {r['file_id']:<44s} n={r['n_points']:<8d} ({r['duration_s']} s)")

    add("")
    add("[9] class imbalance")
    vc = src["fault_type"].value_counts()
    for k, v in vc.items():
        add(f"    {k:<3s} {v:>4d}  {100.0 * v / len(src):5.1f}%")
    add(f"    max/min ratio = {vc.max() / vc.min():.1f}")

    add("")
    add("[10] target")
    add(f"    files: {len(tgt)}, points each: {sorted(tgt['n_points'].unique())}, "
        f"duration: {tgt['duration_s'].iloc[0]} s @ {tgt['fs'].iloc[0]} Hz")
    add("=" * 70)
    return "\n".join(lines)


def plot_class_dist(src, tgt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ct = pd.crosstab(src["fault_type"], src["group"]).reindex(C.CLASSES).fillna(0)
    ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20")
    axes[0].set_title("Source: class x group")
    axes[0].set_xlabel("fault type")
    axes[0].set_ylabel("file count")
    axes[0].tick_params(axis="x", rotation=0)

    tmp = src.copy()
    tmp["fs_khz"] = (tmp["fs"] / 1000).astype(int).astype(str) + " kHz"
    ct2 = pd.crosstab(tmp["fault_type"], tmp["fs_khz"]).reindex(C.CLASSES).fillna(0)
    ct2.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
    axes[1].set_title("Source: class x sampling rate (N only at 48 kHz)")
    axes[1].set_xlabel("fault type")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "audit_class_dist.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.histplot(src["rpm"].dropna(), bins=25, ax=axes[0], color="steelblue")
    axes[0].set_title("Source rpm distribution")
    axes[0].set_xlabel("rpm")
    sns.countplot(data=src, x="fault_type", order=C.CLASSES, ax=axes[1], color="salmon")
    axes[1].set_title("Source class imbalance")
    axes[1].set_xlabel("fault type")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "audit_rpm_dist.png", dpi=150)
    plt.close(fig)


def main():
    src_recs = dl.scan_source()
    tgt_recs = dl.scan_target()
    src = build_table(src_recs)
    tgt = build_table(tgt_recs)

    src.to_csv(C.OUT_DIR / "audit_source.csv", index=False, encoding="utf-8")
    tgt.to_csv(C.OUT_DIR / "audit_target.csv", index=False, encoding="utf-8")

    text = summarize(src, tgt)
    (C.OUT_DIR / "audit_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    plot_class_dist(src, tgt)
    print("[data_audit] wrote audit_source.csv / audit_target.csv / audit_summary.txt + 2 figures")
    return src, tgt


if __name__ == "__main__":
    main()
