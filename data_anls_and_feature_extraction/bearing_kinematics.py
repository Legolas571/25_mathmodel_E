import numpy as np
import pandas as pd

import config as C


def fault_orders(N, d, D):
    r = d / D
    return {
        "BPFO": (N / 2.0) * (1 - r),
        "BPFI": (N / 2.0) * (1 + r),
        "BSF": (D / (2.0 * d)) * (1 - r ** 2),
        "FTF": 0.5 * (1 - r),
    }


def fault_freqs(rpm, N, d, D):
    fr = rpm / 60.0
    o = fault_orders(N, d, D)
    return {k: v * fr for k, v in o.items()}


def bearing_orders(name):
    p = C.BEARINGS[name]
    return fault_orders(p["N"], p["d"], p["D"])


def bearing_freqs(name, rpm):
    p = C.BEARINGS[name]
    return fault_freqs(rpm, p["N"], p["d"], p["D"])


def table():
    rows = []
    for name in C.BEARINGS:
        p = C.BEARINGS[name]
        o = bearing_orders(name)
        row = dict(bearing=name, N=p["N"], d_in=p["d"], D_in=p["D"])
        row.update({f"order_{k}": round(v, 5) for k, v in o.items()})
        row.update({f"freq_1797rpm_{k}_Hz": round(v * 1797 / 60.0, 3) for k, v in o.items()})
        row.update({f"freq_600rpm_{k}_Hz": round(v * 600 / 60.0, 3) for k, v in o.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = table()
    df.to_csv(C.OUT_DIR / "fault_orders.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False))
    print("[bearing_kinematics] wrote fault_orders.csv")
    return df


if __name__ == "__main__":
    main()
