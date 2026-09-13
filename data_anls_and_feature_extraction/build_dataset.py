import argparse
import hashlib
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
import data_loader as dl
import features_freq as ff
import features_order as fo
import features_time as ft
import preprocess as pp
import spectral_kurtosis as sk

META_COLS = ["domain", "file_id", "group", "fs", "win_idx", "win_start", "rpm",
             "fault_type", "fault_size", "or_position", "load"]
DIM_TIME = ["t_mean_abs", "t_rms", "t_std", "t_var", "t_peak", "t_p2p"]

ERR_LOG = C.OUT_DIR / "errors.log"


def add_relative_dims(df):
    for c in DIM_TIME:
        if c not in df.columns:
            continue
        med = df.groupby("file_id")[c].transform("median")
        df[f"{c}_rel"] = df[c] / (med.abs() + 1e-15)
    return df


def _params(args):
    return dict(win=args.win_sec_feature, ov=args.overlap, mp=C.ORDER_POINTS_PER_REV,
                no=C.N_ORDERS, ch=args.channel, band=args.band_mode, sk=args.sk_band)


def _cache_path(file_id, params):
    h = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return C.CACHE_DIR / (file_id.replace("/", "_") + f"_{h}.csv")


def _band_of(rec, W, args):
    if args.band_mode == "fixed":
        return list(C.DEFAULT_BAND)
    if args.band_mode == "none":
        return None
    return sk.best_band(W[0], rec["fs"])


def extract_file(rec, args, rpm_override=None):
    x, fs, rpm = dl.load_signal(rec, channel=args.channel)
    if rpm_override is not None:
        rpm = rpm_override
    if not np.isfinite(rpm):
        rpm = C.DEFAULT_RPM
    fr = rpm / 60.0

    W, starts = pp.segment(x, fs, args.win_sec_feature, args.overlap)
    if W.shape[0] == 0:
        return pd.DataFrame(), 0

    band = _band_of(rec, W, args)
    rows = []
    for i, w in enumerate(W):
        row = {"file_id": rec["file_id"], "win_idx": i, "win_start": int(starts[i]), "rpm": rpm}
        row.update(ft.time_features(w))
        row.update(ff.freq_features(w, fs, band, rec["bearing"], rpm))
        row.update(fo.order_features(w, fs, fr, band, rec["bearing"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "domain", rec["domain"])
    df.insert(2, "group", rec["group"])
    df.insert(3, "fs", fs)
    for c in ("fault_type", "fault_size", "or_position", "load"):
        df[c] = rec.get(c)
    return df, W.shape[0]


def process(recs, args, rpm_map=None, desc=""):
    frames = []
    for rec in tqdm(recs, desc=desc, ncols=90, file=sys.stdout):
        cp = _cache_path(rec["file_id"], _params(args))
        if cp.exists() and not args.no_cache:
            frames.append(pd.read_csv(cp))
            continue
        try:
            df, _n = extract_file(rec, args, (rpm_map or {}).get(rec["file_id"]))
            if df.empty:
                continue
            df.to_csv(cp, index=False, encoding="utf-8")
            frames.append(df)
        except Exception:
            with open(ERR_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"=== {rec['file_id']} ===\n{traceback.format_exc()}\n")
            print(f"  [FAIL] {rec['file_id']} (see errors.log)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_splits(src_df, n_folds, seed):
    files = src_df.groupby("file_id").first().reset_index()[["file_id", "fault_type"]]
    rng = np.random.RandomState(seed)
    folds = {i: [] for i in range(n_folds)}
    for cls in C.CLASSES:
        ids = sorted(files.loc[files["fault_type"] == cls, "file_id"].tolist())
        rng.shuffle(ids)
        for j, fid in enumerate(ids):
            folds[j % n_folds].append(fid)
    all_ids = set(files["file_id"])
    out = {}
    for i in range(n_folds):
        test = sorted(folds[i])
        out[str(i)] = {"test": test, "train": sorted(all_ids - set(test))}
    return out, files


def _fold_table(files, splits):
    rows = []
    for k, v in splits.items():
        sub = files[files["file_id"].isin(v["test"])]
        row = {"fold": k, "n_files": len(v["test"])}
        for cls in C.CLASSES:
            row[f"n_{cls}"] = int((sub["fault_type"] == cls).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_card(src, tgt, splits, fold_tbl, args, t0):
    lines = []
    add = lines.append
    add("=" * 74)
    add("DATASET CARD - task 1 (data analysis & feature extraction)")
    add("=" * 74)
    add(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed: {time.time()-t0:.1f} s")
    add(f"signal window: {args.win_sec_signal} s   feature window: {args.win_sec_feature} s")
    add(f"overlap: {args.overlap}   channel: {args.channel}   band mode: {args.band_mode}")
    add("")
    add(f"[source] windows={len(src)}  files={src['file_id'].nunique()}")
    add(src["fault_type"].value_counts().to_string())
    add("")
    add("[source] windows per group")
    add(src.groupby("group").size().to_string())
    add("")
    add(f"[target] windows={len(tgt)}  files={tgt['file_id'].nunique()}")
    add("")
    add("[folds] file-level stratified split (same file never crosses folds)")
    add(fold_tbl.to_string(index=False))
    add("")
    add("[column groups]")
    add("  meta        : " + ", ".join(META_COLS))
    add("  time  (t_*) : dimensionless ones are directly comparable; the raw")
    add("                dimensional ones (t_rms/t_std/t_var/t_peak/t_p2p/t_mean_abs)")
    add("                are NOT comparable - use the per-file normalized *_rel columns")
    add("  order (o_*) : TRANSFERABLE core features (speed/sampling-rate invariant)")
    add("  order theo  : o_theo_* only exists for source (needs known bearing geometry)")
    add("  freq  (f_*) : ABSOLUTE-frequency features, NOT transferable; source diagnosis only")
    add("")
    add("[how to use in task 2/3]")
    add("  transferable = [c for c in cols if (c.startswith('t_') and c not in")
    add("                  {'t_mean_abs','t_rms','t_std','t_var','t_peak','t_p2p'})")
    add("                  or (c.startswith('o_') and not c.startswith('o_theo_'))]")
    add("  never split random windows of one file across train/test: use splits.json")
    add("=" * 74)
    (C.OUT_DIR / "dataset_card.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="S_all")
    ap.add_argument("--win-sec-signal", type=float, default=C.WIN_SEC_SIGNAL)
    ap.add_argument("--win-sec-feature", type=float, default=C.WIN_SEC_FEATURE)
    ap.add_argument("--overlap", type=float, default=C.OVERLAP_TRAIN)
    ap.add_argument("--channel", default="DE")
    ap.add_argument("--band-mode", default="sk", choices=["sk", "fixed", "none"])
    ap.add_argument("--sk-band", action="store_true", default=True)
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save-windows", action="store_true")
    ap.add_argument("--save-timefreq", action="store_true")
    args = ap.parse_args(argv)

    want = set(args.subsets.split(","))
    src_recs = [r for r in dl.scan_source() if want & set(dl.subset_of(r))]
    tgt_recs = dl.scan_target()
    if args.limit:
        src_recs = src_recs[:args.limit]
        tgt_recs = tgt_recs[:max(2, args.limit // 8)]

    print(f"[build_dataset] source files={len(src_recs)}  target files={len(tgt_recs)}")

    rpm_map = {}
    rpm_csv = C.OUT_DIR / "target_rpm.csv"
    if rpm_csv.exists():
        t = pd.read_csv(rpm_csv)
        rpm_map = dict(zip(t["file_id"], t["fr_final"] * 60.0))
        print(f"[build_dataset] target rpm loaded from target_rpm.csv")
    else:
        print("[build_dataset] target_rpm.csv not found -> using prior 600 rpm")

    src = process(src_recs, args, desc="source")
    tgt = process(tgt_recs, args, rpm_map=rpm_map, desc="target")

    if src.empty:
        print("[build_dataset] no source features produced")
        return

    src = add_relative_dims(src)
    tgt = add_relative_dims(tgt)

    src.to_csv(C.OUT_DIR / "features_source.csv", index=False, encoding="utf-8")
    tgt.to_csv(C.OUT_DIR / "features_target.csv", index=False, encoding="utf-8")

    splits, files = make_splits(src, args.folds, C.SEED)
    (C.OUT_DIR / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    fold_tbl = _fold_table(files, splits)

    if args.save_windows:
        _save_windows(src_recs, args)
    if args.save_timefreq:
        _save_timefreq(src_recs, tgt_recs, args)

    write_card(src, tgt, splits, fold_tbl, args, t0)
    print(f"[build_dataset] source {src.shape}  target {tgt.shape}")
    print(fold_tbl.to_string(index=False))
    print("[build_dataset] wrote features_source.csv / features_target.csv / splits.json / dataset_card.txt")
    return src, tgt


def _save_windows(recs, args):
    for tag, rlist in (("source", recs),):
        arrs = []
        for rec in tqdm(rlist, desc=f"windows-{tag}", ncols=90, file=sys.stdout):
            x, fs, _ = dl.load_signal(rec, channel=args.channel)
            W, _ = pp.segment(x, fs, args.win_sec_signal, args.overlap)
            if W.shape[0]:
                arrs.append(W.astype(np.float32))
        if arrs:
            np.savez_compressed(C.OUT_DIR / f"windows_{tag}_signal.npz",
                                **{f"w{i}": a for i, a in enumerate(arrs)})
            print(f"  saved windows_{tag}_signal.npz ({len(arrs)} files)")


def _save_timefreq(src_recs, tgt_recs, args):
    import features_timefreq as tfq
    for tag, rlist in (("source", src_recs[:40]), ("target", tgt_recs)):
        arrs, ids = [], []
        for rec in rlist:
            x, fs, _ = dl.load_signal(rec, channel=args.channel)
            W, _ = pp.segment(x, fs, C.WIN_SEC_FEATURE, 0.0)
            if W.shape[0] == 0:
                continue
            arrs.append(tfq.batch_timefreq(W[:8], fs))
            ids.append(rec["file_id"])
        if arrs:
            np.savez_compressed(C.OUT_DIR / f"timefreq_{tag}.npz",
                                images=np.concatenate(arrs, axis=0),
                                file_ids=np.array(ids))
            print(f"  saved timefreq_{tag}.npz ({sum(a.shape[0] for a in arrs)} images)")


if __name__ == "__main__":
    main()
