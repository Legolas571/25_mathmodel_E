import argparse
import json
import time

import numpy as np
import pandas as pd

import config as C
import data_io as dio
import methods_shallow as MS
import proxy_eval as PE
import pseudo_label as PL


def infer_one(method, Xs, ys, Xt, seed, classes, log=None):
    t0 = time.time()
    if method in ("M0", "M1", "M2", "M3", "M4"):
        pred, proba = MS.run_shallow(method, Xs, ys, Xt, seed=seed, classes=classes)
    elif method in ("M5", "M6", "M7"):
        pred, proba = PL.run_pseudo(method, Xs, ys, Xt, seed=seed, classes=classes,
                                    log=log)
    elif method in ("M8", "M9"):
        import methods_deep as MD
        pred, proba, _ = MD.run_deep(method, Xs, ys, Xt, seed=seed, classes=classes,
                                     epochs=150)
    elif method == "M10":
        pred, proba = PE.consensus_soft(Xs, ys, Xt, seed, classes)
    else:
        raise KeyError(method)
    return pred, proba, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=None,
                    help="comma separated; default = recommended + top-k")
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS[:3])))
    ap.add_argument("--top-k", type=int, default=C.CONSENSUS_TOP_K)
    args = ap.parse_args(argv)

    src = dio.load_source_features()
    tgt = dio.load_target_features()
    cols = dio.main_feature_columns(src, tgt)
    classes = C.CLASSES
    Xs, ys, _ = dio.build_xy(src, cols, classes)
    Xt = dio.build_X(tgt, cols)
    tgt2 = tgt.reset_index(drop=True)

    if args.methods:
        methods = args.methods.split(",")
    else:
        methods = ["M0"]
        rp = C.RES_DIR / "recommended_method.json"
        if rp.exists():
            rec = json.loads(rp.read_text(encoding="utf-8"))
            methods = ["M0", rec["method"]]
            extra = [m for m in sorted(rec.get("ranking", {}),
                                       key=lambda k: -rec["ranking"][k])][:args.top_k]
            methods = list(dict.fromkeys(methods + extra))
        else:
            methods = ["M0", "M2", "M5", "M7"]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[target_infer] methods={methods} seeds={seeds} "
          f"features={len(cols)} source_windows={len(ys)} target_windows={len(Xt)}")

    vite = tgt2["file_id"].to_numpy()
    proba_rows, vote_rows, iter_log = [], [], []
    t_start = time.time()
    for method in methods:
        for seed in seeds:
            pred, proba, fit_s = infer_one(method, Xs, ys, Xt, seed, classes,
                                           log=iter_log)
            d = pd.DataFrame(proba, columns=classes)
            d.insert(0, "file_id", vite)
            d.insert(1, "method", method)
            d.insert(2, "seed", seed)
            d["pred"] = [classes[i] for i in pred]
            proba_rows.append(d)

            g = dio.file_level(tgt2, pred, classes)
            g["method"] = method
            g["seed"] = seed
            vote_rows.append(g)
            print(f"  {method}/seed{seed}: file labels="
                  f"{g['pred'].value_counts().to_dict()} "
                  f"mean_agree={g['agree'].mean():.3f} ({fit_s:.1f}s)")

    wp = pd.concat(proba_rows, ignore_index=True)
    wp.to_csv(C.RES_DIR / "target_window_proba.csv", index=False, encoding="utf-8")
    vp = pd.concat(vote_rows, ignore_index=True)
    vp.to_csv(C.RES_DIR / "target_file_votes.csv", index=False, encoding="utf-8")
    if iter_log:
        pd.DataFrame(iter_log).to_csv(C.RES_DIR / "pseudo_iter_log.csv", index=False,
                                      encoding="utf-8")

    summ = (vp.groupby(["method", "file_id"])["pred"]
              .agg(pred=lambda s: s.value_counts().idxmax())
              .reset_index())
    print("\n=== file-level label per method (majority over seeds) ===")
    print(summ.pivot(index="file_id", columns="method", values="pred").to_string())
    print(f"\n[target_infer] finished in {(time.time()-t_start)/60:.1f} min")
    return wp, vp


if __name__ == "__main__":
    main()
