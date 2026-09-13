import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import preprocess as pp
import spectral_kurtosis as sk
from bearing_kinematics import bearing_orders
from features_order import envelope_order_spectrum

PRIMARY = {"B": "BSF", "IR": "BPFI", "OR": "BPFO", "N": "FTF"}
PASS_TOL = 0.05


def _eos(rec, band=None):
    x, fs, rpm = dl.load_signal(rec)
    W, _ = pp.segment(x, fs, C.WIN_SEC_FEATURE, 0.0)
    if W.shape[0] == 0:
        return None
    b = band if band is not None else sk.best_band(W[0], fs)
    orders, E = envelope_order_spectrum(W[0], fs, rpm / 60.0, b)
    if orders is None:
        return None
    return dict(orders=orders, E=E, rpm=rpm, fs=fs, band=b,
                res=float(orders[1] - orders[0]))


def find_pairs(src, per_type=3):
    idx = {}
    for r in src:
        if r["group"] not in ("12k_DE", "48k_DE") or r["fault_type"] == "N":
            continue
        key = (r["fault_type"], r["fault_size"], r["load"])
        idx.setdefault(key, {})[r["group"]] = r
    out = []
    for ft in ("B", "IR", "OR"):
        sel = [(k, v) for k, v in idx.items() if k[0] == ft and "12k_DE" in v and "48k_DE" in v]
        sel.sort(key=lambda kv: (kv[0][1] or "", kv[0][2]))
        out += [(k, v["12k_DE"], v["48k_DE"]) for k, v in sel[:per_type]]
    return out


def peak_near(orders, E, center, tol):
    m = (orders >= center * (1 - tol)) & (orders <= center * (1 + tol))
    if not m.any():
        return np.nan
    o, e = orders[m], E[m]
    return float(o[int(np.argmax(e))])


def _tol(d, center):
    return float(max(0.08, 3.0 * d["res"] / max(center, 1e-9)))


def check_h1(src):
    rows = []
    for key, r12, r48 in find_pairs(src, per_type=3):
        a = _eos(r12)
        b = _eos(r48)
        if a is None or b is None:
            continue
        theo = bearing_orders(r12["bearing"])
        k = PRIMARY[r12["fault_type"]]
        c = theo[k]
        o12 = peak_near(a["orders"], a["E"], c, _tol(a, c))
        o48 = peak_near(b["orders"], b["E"], c, _tol(b, c))
        rel = abs(o12 - o48) / c if np.isfinite(o12) and np.isfinite(o48) else np.nan
        rows.append(dict(pair=f"{key[0]}{key[1]}_load{key[2]}", fault=r12["fault_type"],
                         key_order=k, theo=round(c, 3),
                         order_12k=round(o12, 3), order_48k=round(o48, 3),
                         rel_gap_12k_48k=round(rel, 4),
                         dev_from_theory_12k=round(abs(o12 - c) / c, 4) if np.isfinite(o12) else np.nan,
                         dev_from_theory_48k=round(abs(o48 - c) / c, 4) if np.isfinite(o48) else np.nan))
        _plot(a, b, c, r12, r48, key, k)
    return pd.DataFrame(rows)


def check_speed_variation(src):
    rows = []
    by = {}
    for r in src:
        if r["group"] == "12k_DE" and r["fault_type"] in ("B", "IR", "OR"):
            by.setdefault((r["fault_type"], r["fault_size"]), []).append(r)
    for k, rs in by.items():
        if len(rs) < 3:
            continue
        rpms = [dl.read_rpm(r) for r in rs]
        order = np.argsort(rpms)
        ra, rb = rs[int(order[0])], rs[int(order[-1])]
        a, b = _eos(ra), _eos(rb)
        if a is None or b is None:
            continue
        theo = bearing_orders(ra["bearing"])
        key = PRIMARY[ra["fault_type"]]
        c = theo[key]
        oa = peak_near(a["orders"], a["E"], c, _tol(a, c))
        ob = peak_near(b["orders"], b["E"], c, _tol(b, c))
        rows.append(dict(group=f"{k[0]}{k[1]}", order_key=key,
                         rpm_low=round(a["rpm"], 1), rpm_high=round(b["rpm"], 1),
                         order_low=round(oa, 3), order_high=round(ob, 3),
                         theo=round(c, 3),
                         rel_gap=round(abs(oa - ob) / c, 4)))
    return pd.DataFrame(rows)


def _plot(a, b, theo, r12, r48, key, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for d, rec, c in ((a, r12, "tab:blue"), (b, r48, "tab:orange")):
        norm = d["E"] / (d["E"][(d["orders"] >= 0.5) & (d["orders"] <= C.N_ORDERS)].sum() + 1e-15)
        ax.plot(d["orders"], norm, lw=0.8, color=c,
                label=f"{rec['group']} {d['fs']//1000}kHz {d['rpm']:.0f}rpm")
    ax.axvline(theo, color="k", ls="--", lw=1, label=f"theo {k}={theo:.2f}")
    ax.set_xlim(0, C.N_ORDERS)
    ax.set_xlabel("order")
    ax.set_ylabel("normalized envelope order spectrum")
    ax.set_title(f"H1 order alignment: {key[0]}{key[1]} load{key[2]}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"verify_order_align_{key[0]}{key[1]}_load{key[2]}.png"
    fig.savefig(C.FIG_DIR / name, dpi=130)
    plt.close(fig)


def check_h2():
    p = C.OUT_DIR / "target_rpm.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    rel_std = 100.0 * df["rpm_final"].std() / df["rpm_final"].mean()
    return dict(n=len(df), mean_rpm=float(df["rpm_final"].mean()),
                std_rpm=float(df["rpm_final"].std()), rel_std_pct=float(rel_std),
                methods=df["method"].value_counts().to_dict())


def main():
    src = dl.scan_source()
    h1 = check_h1(src)
    h1.to_csv(C.OUT_DIR / "verify_order_align.csv", index=False, encoding="utf-8")
    sv = check_speed_variation(src)
    sv.to_csv(C.OUT_DIR / "verify_speed_variation.csv", index=False, encoding="utf-8")
    h2 = check_h2()

    lines = []
    add = lines.append
    add("=" * 74)
    add("GATE REPORT - order-domain verification (H1 / H2)")
    add("=" * 74)
    add("")
    add("[H1] same fault, different sampling rate / speed -> order peaks aligned?")
    add(h1.to_string(index=False) if len(h1) else "  (no comparable pairs found)")
    if len(h1):
        med = h1.groupby("fault")["rel_gap_12k_48k"].median()
        add("  median rel gap per fault type:")
        add(med.round(4).to_string())
        ok = h1["rel_gap_12k_48k"] <= PASS_TOL
        add(f"  pairs within {PASS_TOL:.0%}: {ok.mean():.0%}   "
            f"overall median rel gap = {h1['rel_gap_12k_48k'].median():.4f}")
        add(f"  H1 verdict: {'PASS' if h1['rel_gap_12k_48k'].median() <= PASS_TOL else 'FAIL'}"
            "   (verdict uses the median; BSF is the weakest order and is the main outlier)")
    add("")
    add("[H1b] same fault, different rotational speed (load variation)")
    add(sv.to_string(index=False) if len(sv) else "  (none)")
    if len(sv):
        add(f"  median rel gap = {sv['rel_gap'].median():.4f}")
    add("")
    add("[H2] target rpm estimate stability")
    if h2:
        add(f"  n={h2['n']}  mean={h2['mean_rpm']:.2f} rpm  std={h2['std_rpm']:.3f}  "
            f"rel_std={h2['rel_std_pct']:.2f}%  methods={h2['methods']}")
        add(f"  H2 verdict: {'PASS' if h2['rel_std_pct'] < 3.0 else 'FAIL'}")
    else:
        add("  target_rpm.csv not found - run rpm_estimator first")
    add("")
    add("[note] order resolution = 1 / revolutions-in-window.")
    add("  source 0.5 s @1800 rpm -> ~15 rev -> 0.067 order")
    add("  target 0.5 s @ 600 rpm -> ~ 5 rev -> 0.200 order (coarser; +-3% of order")
    add("  3.58 is +-0.107, i.e. about one bin - top-K peak features are more robust)")
    add("")
    add("[fallback if H1 fails] use time-domain dimensionless features + relative")
    add("  band-energy ratios instead of pure order features.")
    add("=" * 74)
    text = "\n".join(lines)
    (C.OUT_DIR / "verify_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return h1, sv, h2


if __name__ == "__main__":
    main()
