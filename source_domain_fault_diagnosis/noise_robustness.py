import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import config as C
import data_io as dio
import feature_prep as fp
import models as M

DRIVER = r'''
import json, sys
import numpy as np
sys.path.insert(0, r"{task1}")
import config as T1
import data_loader as dl
import preprocess as pp
import spectral_kurtosis as sk
import features_time as ft
import features_freq as ff
import features_order as fo

snr_list = {snrs}
out_dir = r"{out_dir}"
recs = dl.scan_source()
rows = []
for snr in snr_list:
    for rec in recs:
        x, fs, rpm = dl.load_signal(rec)
        if not np.isfinite(rpm):
            rpm = T1.DEFAULT_RPM
        if snr is not None:
            p = np.mean(x ** 2)
            npow = p / (10 ** (snr / 10.0))
            x = x + np.random.RandomState(abs(hash(rec["file_id"])) % 2**31).normal(
                0, np.sqrt(npow), size=x.shape)
        W, _ = pp.segment(x, fs, T1.WIN_SEC_FEATURE, 0.5)
        if W.shape[0] == 0:
            continue
        band = sk.best_band(W[0], fs)
        for w in W:
            r = {{"file_id": rec["file_id"], "group": rec["group"], "fs": fs,
                 "fault_type": rec["fault_type"], "load": rec["load"]}}
            r.update(ft.time_features(w))
            r.update(ff.freq_features(w, fs, band, rec["bearing"], rpm))
            r.update(fo.order_features(w, fs, rpm / 60.0, band, rec["bearing"]))
            rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in ["t_mean_abs", "t_rms", "t_std", "t_var", "t_peak", "t_p2p"]:
        if c in df.columns:
            med = df.groupby("file_id")[c].transform("median")
            df[c + "_rel"] = df[c] / (med.abs() + 1e-15)
    df.to_csv(out_dir + "/noisy_features_snr%s.csv" % snr, index=False)
    rows = []
    print("snr", snr, "done", flush=True)
'''


def build_noisy_features(snrs):
    code = DRIVER.format(task1=str(C.TASK1_DIR).replace("\\", "\\\\"),
                         out_dir=str(C.RES_DIR).replace("\\", "\\\\"),
                         snrs=json.dumps(snrs))
    print(f"[noise] extracting noisy features for SNR={snrs} (subprocess in task-1 dir)")
    logp = C.RES_DIR / "noise_extract.log"
    with open(logp, "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-c", code], cwd=str(C.TASK1_DIR),
                           stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(logp.read_text(encoding="utf-8", errors="replace")[-2500:])
        raise RuntimeError("noisy feature extraction failed")
    print(logp.read_text(encoding="utf-8", errors="replace").strip())


def evaluate(snrs, fset="F2", model="svm", protocol="P1"):
    splits = dio.load_splits()
    base = dio.load_source_features()
    labels = base.groupby("file_id")["fault_type"].first()
    rows = []
    for snr in [None] + list(snrs):
        path = (C.SRC_FEATURES if snr is None else
                C.RES_DIR / f"noisy_features_snr{snr}.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["fault_type"] = df["file_id"].map(labels)
        cols = dio.feature_columns(df, fset)
        classes = C.PROTOCOL_CLASSES[protocol]
        f1s, accs = [], []
        for fold, split in splits[protocol].items():
            tr, te = dio.group_split(df, split)
            Xtr, ytr, _ = dio.build_xy(tr, cols, classes)
            Xte, yte, _ = dio.build_xy(te, cols, classes)
            pipe = fp.FeaturePipeline()
            Xtr2, Xte2 = pipe.fit_transform(Xtr, Xte, columns=cols)
            clf = M.make_model(model, len(classes), C.SEEDS[0],
                               fp.compute_class_weight(ytr))
            y_pred, _, _ = M.fit_predict(clf, Xtr2, ytr, Xte2)
            m = M.metrics(yte, y_pred, np.unique(yte))
            f1s.append(m["macro_f1"])
            accs.append(m["accuracy"])
        rows.append({"snr_db": "clean" if snr is None else snr,
                     "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                     "accuracy": float(np.mean(accs)), "protocol": protocol,
                     "feature_set": fset, "model": model})
        print(f"  SNR={rows[-1]['snr_db']}: macroF1={rows[-1]['macro_f1']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "noise_robustness.csv", index=False, encoding="utf-8")
    _plot(out)
    return out


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["snr_db"] != "clean"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar([float(v) for v in d["snr_db"]], d["macro_f1"],
                yerr=d["macro_f1_std"], marker="o", capsize=4, label="noisy")
    cl = df[df["snr_db"] == "clean"]
    if len(cl):
        ax.axhline(float(cl["macro_f1"].iloc[0]), color="r", ls="--",
                   label="clean baseline")
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("macro-F1")
    ax.set_title("Noise robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "noise_curve.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default="20,10,5,0")
    ap.add_argument("--feature-set", default="F2")
    ap.add_argument("--model", default="svm")
    ap.add_argument("--protocol", default="P1")
    args = ap.parse_args(argv)
    snrs = [int(s) for s in args.snrs.split(",")]
    build_noisy_features(snrs)
    out = evaluate(snrs, args.feature_set, args.model, args.protocol)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
