import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

import config as C

_NAME_RE = re.compile(
    r"^(?P<ft>B|IR|OR|N)(?P<size>\d{3})?(?:@(?P<pos>\d+))?_(?P<load>\d+)"
    r"(?:_\((?P<rpm>\d+)rpm\))?\.mat$"
)


def parse_filename(name):
    m = _NAME_RE.match(name)
    if m is None:
        return None
    g = m.groupdict()
    return dict(
        fault_type=g["ft"],
        fault_size=g["size"],
        or_position=int(g["pos"]) if g["pos"] else None,
        load=int(g["load"]),
        rpm_filename=float(g["rpm"]) if g["rpm"] else None,
    )


def _probe(path):
    names = []
    shapes = {}
    for nm, shp, _dt in whosmat(str(path)):
        names.append(nm)
        shapes[nm] = tuple(shp)
    return names, shapes


def _pick(names, suffix):
    for nm in names:
        if nm.upper().endswith(suffix):
            return nm
    return None


def _time_channels(names):
    out = []
    for nm in names:
        for ch in ("DE", "FE", "BA"):
            if nm.upper().endswith("_" + ch + "_TIME"):
                out.append(ch)
                break
    return out


def _record(path, group, domain):
    names, shapes = _probe(path)
    chans = _time_channels(names)
    de = _pick(names, "_DE_TIME") or _pick(names, "_FE_TIME") or _pick(names, "_BA_TIME")
    if de is None:
        time_keys = [n for n in names if n.upper().endswith("_TIME")]
        de = time_keys[0] if time_keys else names[0]
    meta = parse_filename(path.name) or dict(
        fault_type=None, fault_size=None, or_position=None, load=None, rpm_filename=None
    )
    return dict(
        file_id=str(path.relative_to(C.SRC_DIR if domain == "src" else C.TGT_DIR)).replace("\\", "/")[:-4],
        path=path,
        domain=domain,
        group=group,
        fs=C.GROUPS[group]["fs"],
        bearing=C.GROUPS[group]["bearing"],
        channels=chans,
        var_names=names,
        signal_var=de,
        rpm_var=_pick(names, "RPM"),
        n_points=int(shapes[de][0]),
        **meta,
    )


def scan_source():
    recs = []
    for sub in sorted(p for p in C.SRC_DIR.iterdir() if p.is_dir()):
        group = sub.name.replace("_data", "").replace("kHz_", "k_")
        for f in sorted(sub.rglob("*.mat")):
            recs.append(_record(f, group, "src"))
    return recs


def scan_target():
    recs = []
    for f in sorted(C.TGT_DIR.glob("*.mat")):
        names, shapes = _probe(f)
        key = next((n for n in names if not n.startswith("__")), names[0])
        recs.append(dict(
            file_id=f.stem,
            path=f,
            domain="tgt",
            group="target",
            fs=C.FS_TGT,
            bearing=None,
            channels=["single"],
            var_names=names,
            signal_var=key,
            rpm_var=None,
            n_points=int(shapes[key][0]),
            fault_type=None,
            fault_size=None,
            or_position=None,
            load=None,
            rpm_filename=None,
        ))
    return recs


def load_signal(rec, channel="DE"):
    key = _pick(rec["var_names"], "_" + channel + "_TIME")
    if key is None:
        key = rec["signal_var"]
    wanted = [key] + ([rec["rpm_var"]] if rec.get("rpm_var") else [])
    md = loadmat(str(rec["path"]), variable_names=wanted)
    x = np.asarray(md[key]).ravel().astype(np.float64)
    rpm = np.nan
    if rec.get("rpm_var") and rec["rpm_var"] in md:
        rpm = float(np.asarray(md[rec["rpm_var"]]).ravel()[0])
    if not np.isfinite(rpm) and rec.get("rpm_filename"):
        rpm = float(rec["rpm_filename"])
    return x, rec["fs"], rpm


def resolve_rpm(rec, x=None):
    _, _, rpm = load_signal(rec)
    if not np.isfinite(rpm):
        rpm = C.DEFAULT_RPM
    return rpm


def read_rpm(rec):
    if rec.get("rpm_var"):
        md = loadmat(str(rec["path"]), variable_names=[rec["rpm_var"]])
        v = float(np.asarray(md[rec["rpm_var"]]).ravel()[0])
        if np.isfinite(v):
            return v
    if rec.get("rpm_filename"):
        return float(rec["rpm_filename"])
    return C.DEFAULT_RPM


def record_table(recs):
    cols = ["file_id", "domain", "group", "fs", "n_points", "channels", "bearing",
            "fault_type", "fault_size", "or_position", "load", "rpm_filename"]
    rows = []
    for r in recs:
        row = {c: r.get(c) for c in cols}
        row["channels"] = ",".join(r.get("channels") or [])
        row["duration_s"] = round(r["n_points"] / r["fs"], 4)
        row["has_rpm_var"] = bool(r.get("rpm_var"))
        rows.append(row)
    return rows


def subset_of(rec):
    tags = []
    g = rec["group"]
    if g in ("12k_DE", "12k_FE", "48k_DE", "48k_Normal"):
        tags.append("S_all")
    if g in ("48k_DE", "48k_Normal"):
        tags.append("S_48k")
    if g == "12k_DE":
        tags.append("S_12k_DE")
    return tags
