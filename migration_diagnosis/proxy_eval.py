import argparse
import json
import time

import numpy as np
import pandas as pd

import config as C
import data_io as dio
import methods_shallow as MS
import models_common as MC
import pseudo_label as PL

RESULT_COLS = (["proxy", "fold", "method", "seed", "n_source_files", "n_target_files",
                "n_source_windows", "n_target_windows", "n_features", "fit_seconds",
                "macro_f1", "accuracy", "n_classes"]
               + [f"f1_{c}" for c in C.CLASSES] + [f"recall_{c}" for c in C.CLASSES])


def _append(path, row, header):
    pd.DataFrame([row]).reindex(columns=RESULT_COLS).to_csv(
        path, mode="a", header=header, index=False, encoding="utf-8")
    return False


def run_method(method, Xs, ys, Xt, seed, classes):
    t0 = time.time()
    if method in ("M0", "M1", "M2", "M3", "M4"):
        pred, proba = MS.run_shallow(method, Xs, ys, Xt, seed=seed, classes=classes)
        fit_s = time.time() - t0
    elif method in ("M5", "M6", "M7"):
        pred, proba = PL.run_pseudo(method, Xs, ys, Xt, seed=seed, classes=classes)
        fit_s = time.time() - t0
    elif method in ("M8", "M9"):
        import methods_deep as MD
        pred, proba, fit_s = MD.run_deep(method, Xs, ys, Xt, seed=seed,
                                         classes=classes, epochs=150)
    elif method == "M10":
        pred, proba = consensus_soft(Xs, ys, Xt, seed, classes)
        fit_s = time.time() - t0
    else:
        raise KeyError(method)
    return pred, proba, fit_s


def consensus_soft(Xs, ys, Xt, seed, classes, members=None):
    members = members or ["M0", "M2", "M4", "M5"]
    probs = []
    for m in members:
        try:
            _, p = MS.run_shallow(m, Xs, ys, Xt, seed=seed, classes=classes) \
                if m in ("M0", "M1", "M2", "M3", "M4") else \
                PL.run_pseudo(m, Xs, ys, Xt, seed=seed, classes=classes)
            probs.append(p)
        except Exception:
            continue
    if not probs:
        return MS.run_shallow("M0", Xs, ys, Xt, seed=seed, classes=classes)
    P = np.mean(probs, axis=0)
    return P.argmax(1), P


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxies", default=",".join(C.PROXY_ORDER))
    ap.add_argument("--methods", default=",".join(C.METHODS_ALL))
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS[:2])))
    ap.add_argument("--tag", default="proxy")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)

    proxies = args.proxies.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    src = dio.load_source_features()
    splits = dio.load_splits()
    cols = dio.feature_columns(src, C.FEATURE_MAIN)
    out = C.RES_DIR / f"runs_{args.tag}.csv"
    if args.fresh and out.exists():
        out.unlink()
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        done = set(zip(d["proxy"], d["fold"], d["method"], d["seed"]))

    combos = [(p, f, m, s) for p in proxies for m in methods for s in seeds
              for f in dio.proxy_folds(splits, p)]
    todo = [c for c in combos if c not in done]
    print(f"[proxy_eval] tag={args.tag} total={len(combos)} todo={len(todo)}")
    header = not out.exists()
    t_start = time.time()
    n = 0
    for (p, fold, method, seed) in todo:
        classes = C.PROXY_TASKS[p]["classes"]
        tr, te = dio.load_proxy_split(src, splits, p, fold)
        Xs, ys, _ = dio.build_xy(tr, cols, classes)
        Xt, yt, _ = dio.build_xy(te, cols, classes)
        try:
            pred, proba, fit_s = run_method(method, Xs, ys, Xt, seed, classes)
        except Exception as e:
            print(f"  [FAIL] {p}/{fold}/{method}/s{seed}: {type(e).__name__}: {e}")
            continue
        m = MC.evaluate(yt, pred, classes)
        row = {"proxy": p, "fold": fold, "method": method, "seed": seed,
               "n_source_files": tr["file_id"].nunique(),
               "n_target_files": te["file_id"].nunique(),
               "n_source_windows": len(ys), "n_target_windows": len(yt),
               "n_features": Xs.shape[1], "fit_seconds": fit_s}
        row.update(m)
        header = _append(out, row, header)
        n += 1
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t_start
            print(f"  {n}/{len(todo)} elapsed {el/60:.1f} min eta "
                  f"{el/max(n,1)*(len(todo)-n)/60:.1f} min | last {p}/{fold}/{method} "
                  f"macroF1={m['macro_f1']:.3f}")
    print(f"[proxy_eval] wrote {out} (+{n}) in {(time.time()-t_start)/60:.1f} min")

    summarize()
    return out


def summarize(path=None):
    """Aggregate every runs_proxy*.csv so that staged runs (shallow/pseudo tag
    and deep tag) do not overwrite each other's summary."""
    if path is None:
        files = sorted(C.RES_DIR.glob("runs_proxy*.csv"))
        files = [f for f in files if f.name != "runs_proxy_"]
        if not files:
            print("[proxy_eval] no runs_proxy*.csv found")
            return pd.DataFrame(), None
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        d = pd.read_csv(path)
    d = d.drop_duplicates(subset=["proxy", "fold", "method", "seed"], keep="last")
    base = d[d["method"] == "M0"].groupby("proxy")["macro_f1"].mean()
    rows = []
    for (p, m), g in d.groupby(["proxy", "method"]):
        rows.append({"proxy": p, "method": m, "macro_f1_mean": g["macro_f1"].mean(),
                     "macro_f1_std": g["macro_f1"].std(),
                     "accuracy_mean": g["accuracy"].mean(),
                     "n_runs": len(g), "fit_s": g["fit_seconds"].mean()})
    s = pd.DataFrame(rows)
    s["gain_vs_M0"] = s.apply(lambda r: r["macro_f1_mean"] - base.get(r["proxy"], np.nan),
                              axis=1)
    s = s.sort_values(["proxy", "macro_f1_mean"], ascending=[True, False])
    s.to_csv(C.RES_DIR / "method_comparison.csv", index=False, encoding="utf-8")

    pivot = s.pivot(index="method", columns="proxy", values="macro_f1_mean")
    gain = s.pivot(index="method", columns="proxy", values="gain_vs_M0")
    print("\n=== method x proxy macro-F1 ===")
    print(pivot.round(4).to_string())
    print("\n=== gain vs M0 ===")
    print(gain.round(4).to_string())

    key = "A_P4"
    rec = None
    if key in gain.columns:
        g = gain[key].dropna().sort_values(ascending=False)
        g = g[g.index != "M0"]
        if len(g):
            best = g.index[0]
            rec = {"proxy_used": key, "method": best,
                   "gain": float(g.iloc[0]),
                   "macro_f1": float(pivot.loc[best, key]),
                   "meets_threshold": bool(g.iloc[0] >= C.MIN_ACCEPT_GAIN),
                   "ranking": {k: float(v) for k, v in g.items()}}
            (C.RES_DIR / "recommended_method.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\n[proxy_eval] recommended method on {key}: {best} "
                  f"(gain {g.iloc[0]:+.4f})")
    return s, rec


if __name__ == "__main__":
    main()
