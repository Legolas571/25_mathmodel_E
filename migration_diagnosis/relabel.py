"""Task-3 redo: file-level transfer labelling with the class-existence constraint.

Why the first attempt was wrong
  * window-level probability voting amplified the source class prior
    (OR = 47.8% of source windows) and produced 13 OR / 3 B / 0 IR / 0 N,
    which contradicts the problem statement that the target contains OR, IR, B
    and N;
  * the 16 target segments are separable on their own (silhouette ~0.36);
  * file-level aggregation is far stronger than window-level voting
    (proxy A_P4 macro-F1 0.437 vs 0.304, measured in file_level.py).

Method chosen by the proxy experiments (file_method_comparison.csv):
  N6 = per-file feature aggregation + source model WITHOUT domain alignment
       + class-existence constraint (every class at least once)
  N6 >= N0 on all three proxies, and the constraint is what guarantees the
  label set is consistent with the task statement.
"""
import argparse

import numpy as np
import pandas as pd

import config as C
import data_io as dio
import file_level as FL
import models_common as MC
import methods_shallow as MS


def score_matrix(Xs_f, ys_f, Xt_f, classes, seeds):
    """Average the file-level posteriors over seeds (and over N6/N0 which differ
    only by the constraint)."""
    mats = []
    for s in seeds:
        lab, proba = FL.run_file_method("N0", Xs_f, ys_f, Xt_f, classes, seed=s)
        mats.append(proba)
    return np.mean(mats, axis=0)


def prototype_evidence(Xs_f, ys_f, Xt_f, classes, aligned=True):
    """Distance from each target file to the source class prototypes.
    `aligned=True` uses the CORAL-aligned space, which is the only space where
    the prototypes are not all dominated by OR (see relabel report)."""
    from sklearn.preprocessing import RobustScaler
    n_classes = len(classes)
    sc = RobustScaler().fit(Xs_f)
    Xs_a, Xt_a = sc.transform(Xs_f), sc.transform(Xt_f)
    if aligned:
        Xs_a, Xt_a = MS.coral_align(Xs_a, Xt_a)
    protos = FL._prototypes(Xs_a, ys_f, n_classes)
    d = np.linalg.norm(Xt_a[:, None, :] - protos[None, :, :], axis=2)
    order = np.argsort(d, axis=1)
    near, second = d[np.arange(len(d)), order[:, 0]], d[np.arange(len(d)), order[:, 1]]
    ratio = (second - near) / (second + 1e-9)
    return ratio, order[:, 0], d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--min-per-class", type=int, default=1)
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    src = dio.load_source_features()
    tgt = dio.load_target_features()
    cols = dio.main_feature_columns(src, tgt)
    classes = C.CLASSES

    sf = FL.aggregate_to_files(src, cols)
    tf = FL.aggregate_to_files(tgt, cols)
    Xs = sf[cols].to_numpy(float)
    ys = sf["fault_type"].map(C.CLASS_TO_IDX).to_numpy()
    Xt = tf[cols].to_numpy(float)
    print(f"[relabel] source {Xs.shape} files, target {Xt.shape} files, "
          f"features={len(cols)}")

    scores = score_matrix(Xs, ys, Xt, classes, seeds)
    unconstrained = np.array([classes[i] for i in scores.argmax(1)])
    constrained = np.array([classes[i] for i in FL.constrained_assign(
        scores, min_per_class=args.min_per_class)])
    balanced = np.array([classes[i] for i in FL.constrained_assign(scores, balanced=True)])

    ratio_u, near_u, _ = prototype_evidence(Xs, ys, Xt, classes, aligned=False)
    ratio_a, near_a, _ = prototype_evidence(Xs, ys, Xt, classes, aligned=True)

    per_seed = np.stack([
        [classes[i] for i in FL.run_file_method("N0", Xs, ys, Xt, classes, seed=s)[0]]
        for s in seeds])
    seed_agree = np.array([(per_seed[:, j] ==
                            pd.Series(per_seed[:, j]).mode().iloc[0]).mean()
                           for j in range(len(tf))])

    p = np.sort(scores, axis=1)[:, ::-1]
    margin = (p[:, 0] - p[:, 1]) / (p[:, 0] + 1e-9)
    margin = margin / (margin.max() + 1e-9)
    proto_agree = (np.array([classes[i] for i in near_a]) == constrained).astype(float)
    conf = np.clip(0.5 * seed_agree + 0.3 * margin + 0.2 * proto_agree, 0, 1)

    out = pd.DataFrame({
        "file_id": tf["file_id"],
        "label": constrained,
        "confidence": conf.round(4),
        "label_unconstrained": unconstrained,
        "label_balanced": balanced,
        "from_constraint_only": (unconstrained != constrained),
        "seed_agreement": seed_agree.round(4),
        "score_margin": margin.round(4),
        "nearest_proto_unalianged": [classes[i] for i in near_u],
        "nearest_proto_aligned": [classes[i] for i in near_a],
        "proto_agree": proto_agree,
        "score_B": scores[:, 0].round(4),
        "score_IR": scores[:, 1].round(4),
        "score_OR": scores[:, 2].round(4),
        "score_N": scores[:, 3].round(4),
    })
    out.to_csv(C.LABEL_DIR / "target_labels.csv", index=False, encoding="utf-8")
    out.to_csv(C.RES_DIR / "relabel_report.csv", index=False, encoding="utf-8")

    print("\n=== label distribution comparison ===")
    for nm, lab in (("unconstrained (old behaviour)", unconstrained),
                    (f"constrained min>={args.min_per_class} (CHOSEN)", constrained),
                    ("balanced", balanced)):
        print(f"  {nm:<34s} {pd.Series(lab).value_counts().to_dict()}")
    print(f"\n  files whose label came from the constraint alone: "
          f"{int((unconstrained != constrained).sum())}/{len(tf)}")
    print(f"  nearest prototype (unaligned): "
          f"{pd.Series([classes[i] for i in near_u]).value_counts().to_dict()}")
    print(f"  nearest prototype (CORAL-aligned): "
          f"{pd.Series([classes[i] for i in near_a]).value_counts().to_dict()}")
    print("\n=== final labels ===")
    print(out[["file_id", "label", "confidence", "label_unconstrained",
               "from_constraint_only", "nearest_proto_aligned"]].to_string(index=False))
    print(f"\nmean confidence: {conf.mean():.3f}")
    print(f"wrote {C.LABEL_DIR / 'target_labels.csv'}")
    return out


if __name__ == "__main__":
    main()
