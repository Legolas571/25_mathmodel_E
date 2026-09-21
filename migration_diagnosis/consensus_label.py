import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config as C
import data_io as dio

ORDER_BANDS = {  # plausible fault-characteristic order ranges
    "OR": (3.1, 3.9),
    "IR": (4.9, 5.9),
    "B": (1.7, 2.7),
    "N": (0.0, 0.0),
}
SHAFT_LOW = (0.3, 1.2)


def method_consensus(votes):
    """Fraction of (method, seed) runs that agree with the file's majority label."""
    per_file = votes.groupby(["file_id", "method", "seed"])["pred"].agg(
        lambda s: s.value_counts().idxmax()).reset_index()
    out = []
    for fid, g in per_file.groupby("file_id"):
        vc = g["pred"].value_counts()
        out.append({"file_id": fid, "label": vc.idxmax(),
                    "method_consensus": float(vc.iloc[0] / len(g)),
                    "n_methods": int(len(g)),
                    "votes": vc.to_dict()})
    return pd.DataFrame(out)


def window_consensus(votes):
    g = votes.groupby("file_id")["agree"].mean().rename("window_agreement")
    return g.reset_index()


def cluster_check(tgt, cols, labels, n_clusters=4, seed=0):
    X = tgt[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Xs)
    lab = km.labels_
    sil = float(silhouette_score(Xs, lab)) if len(set(lab)) > 1 else float("nan")
    y_pred = tgt["file_id"].map(dict(zip(labels["file_id"], labels["label"])))
    y_pred = pd.Categorical(y_pred, categories=C.CLASSES).codes
    ok = y_pred >= 0
    ari = float(adjusted_rand_score(y_pred[ok], lab[ok])) if ok.sum() > 1 else float("nan")
    per_file = pd.DataFrame({"file_id": tgt["file_id"], "cluster": lab}) \
        .groupby("file_id")["cluster"].agg(lambda s: s.value_counts().idxmax())
    return {"silhouette": sil, "ari_vs_pred": ari,
            "cluster_per_file": per_file.to_dict()}


def mechanism_check(tgt):
    rows = []
    for fid, g in tgt.groupby("file_id"):
        peaks = np.concatenate([g[f"o_pk{i}_order"].to_numpy(float) for i in range(4)])
        amps = np.concatenate([g[f"o_pk{i}_amp"].to_numpy(float) for i in range(4)])
        lo = float(np.mean(peaks[(peaks >= SHAFT_LOW[0]) & (peaks <= SHAFT_LOW[1])]
                           if ((peaks >= SHAFT_LOW[0]) & (peaks <= SHAFT_LOW[1])).any()
                           else [np.nan]))
        hits = {}
        for cls, (a, b) in ORDER_BANDS.items():
            if b <= a:
                hits[cls] = np.nan
                continue
            m = (peaks >= a) & (peaks <= b)
            hits[cls] = float(amps[m].mean()) if m.any() else 0.0
        best = max([k for k in hits if np.isfinite(hits[k])],
                   key=lambda k: hits[k]) if any(np.isfinite(list(hits.values()))) else None
        rows.append({"file_id": fid, "peak_order_mean": float(np.mean(peaks)),
                     "peak_order_top": float(peaks[np.argmax(amps)]),
                     "mech_hint": best,
                     **{f"mech_{k}": v for k, v in hits.items()}})
    return pd.DataFrame(rows)


def fuse(cons, wcons, mech, cluster_info, out_labels=None):
    df = cons.merge(wcons, on="file_id", how="left").merge(mech, on="file_id", how="left")
    df["mech_agree"] = (df["mech_hint"] == df["label"]).astype(float)
    df.loc[df["mech_hint"].isna(), "mech_agree"] = 0.5
    df["cluster"] = df["file_id"].map(cluster_info["cluster_per_file"])
    cl_pred = df["cluster"]
    df["cluster_agree"] = (cl_pred == cl_pred.mode().iloc[0]).astype(float)
    df["confidence"] = (0.5 * df["method_consensus"]
                        + 0.3 * df["window_agreement"]
                        + 0.2 * df["mech_agree"]).round(4)
    df = df.sort_values("file_id").reset_index(drop=True)
    if out_labels is not None:
        out = df[["file_id", "label", "confidence", "method_consensus",
                  "window_agreement", "mech_agree", "mech_hint", "n_methods"]]
        out.to_csv(out_labels, index=False, encoding="utf-8")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba", default=str(C.RES_DIR / "target_window_proba.csv"))
    ap.add_argument("--votes", default=str(C.RES_DIR / "target_file_votes.csv"))
    args = ap.parse_args(argv)

    tgt = dio.load_target_features()
    src = dio.load_source_features()
    cols = dio.main_feature_columns(src, tgt)
    votes = pd.read_csv(args.votes)

    cons = method_consensus(votes)
    wcons = window_consensus(votes)
    mech = mechanism_check(tgt)
    cl = cluster_check(tgt, cols, cons, n_clusters=4)

    print("=== method consensus ===")
    print(cons[["file_id", "label", "method_consensus", "n_methods"]].to_string(index=False))
    print("\n=== mechanism hints ===")
    print(mech[["file_id", "peak_order_top", "mech_hint"]].to_string(index=False))
    print(f"\ncluster silhouette={cl['silhouette']:.3f}  ARI vs predicted={cl['ari_vs_pred']:.3f}")

    df = fuse(cons, wcons, mech, cl, out_labels=C.LABEL_DIR / "target_labels.csv")
    df.drop(columns=["votes"]).to_csv(C.RES_DIR / "consensus_report.csv", index=False,
                                      encoding="utf-8")

    low = df[df["confidence"] < df["confidence"].median()]
    low[["file_id", "label", "confidence", "method_consensus", "window_agreement",
         "mech_hint"]].to_csv(C.RES_DIR / "low_confidence_list.csv", index=False,
                              encoding="utf-8")

    print("\n=== FINAL TARGET LABELS ===")
    print(df[["file_id", "label", "confidence", "method_consensus",
              "window_agreement", "mech_hint"]].to_string(index=False))
    print("\nlabel distribution:", df["label"].value_counts().to_dict())
    print(f"mean confidence: {df['confidence'].mean():.3f}")
    print(f"wrote {C.LABEL_DIR / 'target_labels.csv'}")
    return df


if __name__ == "__main__":
    main()
