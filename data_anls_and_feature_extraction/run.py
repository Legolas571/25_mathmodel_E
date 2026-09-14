import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config as C


def _step(title, fn):
    print("\n" + "=" * 70)
    print(f"[run] {title}")
    print("=" * 70)
    t0 = time.time()
    try:
        out = fn()
        print(f"[run] {title} done in {time.time()-t0:.1f}s")
        return True
    except Exception:
        print(f"[run] {title} FAILED after {time.time()-t0:.1f}s")
        traceback.print_exc()
        return False


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    quick = "--quick" in argv
    t0 = time.time()
    log = []

    stale = sorted(C.FIG_DIR.glob("*.png"))
    for p in stale:
        p.unlink()
    if stale:
        print(f"[run] cleared {len(stale)} stale figure(s) from figures/")

    import data_audit
    import bearing_kinematics
    import features_freq
    import features_timefreq
    import rpm_estimator
    import verify_order_domain
    import build_dataset
    import analysis_report
    import data_loader

    steps = [
        ("1/7 data audit", lambda: data_audit.main()),
        ("2/7 bearing kinematics", lambda: bearing_kinematics.main()),
        ("3/7 target rpm estimation", lambda: rpm_estimator.main()),
        ("4/7 gate: order-domain verification", lambda: verify_order_domain.main()),
        ("5/7 dataset build (features)", lambda: build_dataset.main(argv)),
        ("6/7 mechanism + time-frequency figures", lambda: (
            features_freq.verify_mechanism(data_loader.scan_source(), n_files=4),
            features_timefreq.main())),
        ("7/7 feature analysis report", lambda: analysis_report.main()),
    ]

    if quick:
        steps = steps[:4]

    for title, fn in steps:
        ok = _step(title, fn)
        log.append(f"{title}: {'OK' if ok else 'FAILED'}")
        if not ok and title.startswith(("3/7", "5/7")):
            print("[run] critical step failed - stopping")
            break

    text = ["=" * 70, "TASK-1 PIPELINE RUN LOG", "=" * 70,
            f"python {sys.version.split()[0]} | numpy {__import__('numpy').__version__} | "
            f"scipy {__import__('scipy').__version__} | seed {C.SEED}",
            f"total elapsed {time.time()-t0:.1f}s", ""]
    text += log
    (C.OUT_DIR / "run_log.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n" + "\n".join(text))
    return log


if __name__ == "__main__":
    main()
