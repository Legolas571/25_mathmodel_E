import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config as C


def _step(title, fn):
    print("\n" + "=" * 72)
    print(f"[run] {title}")
    print("=" * 72)
    t0 = time.time()
    try:
        fn()
        print(f"[run] {title} done in {time.time()-t0:.1f}s")
        return True
    except Exception:
        print(f"[run] {title} FAILED after {time.time()-t0:.1f}s")
        traceback.print_exc()
        return False


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    quick = "--quick" in argv
    deep = "--deep" in argv
    t0 = time.time()
    log = []

    for p in sorted(C.FIG_DIR.glob("*.png")):
        p.unlink()

    import baseline
    import proxy_eval
    import file_level
    import target_infer
    import consensus_label
    import relabel
    import visualize
    import methods_deep

    if quick:
        steps = [
            ("1/4 baseline (M0 + domain shift)", lambda: baseline.main()),
            ("2/4 proxy eval (A_P4, M0/M2/M5)", lambda: proxy_eval.main(
                ["--proxies", "A_P4", "--methods", "M0,M2,M5", "--seeds", "0",
                 "--tag", "proxy", "--fresh"])),
            ("3/4 target inference", lambda: target_infer.main(
                ["--methods", "M0,M2,M5", "--seeds", "0"])),
            ("4/4 consensus labelling", lambda: consensus_label.main([])),
        ]
    else:
        steps = [
            ("1/6 baseline (M0 + domain shift)", lambda: baseline.main()),
            ("2/6 proxy eval: shallow + pseudo methods",
             lambda: proxy_eval.main(["--proxies", ",".join(C.PROXY_ORDER),
                                      "--methods", ",".join(C.METHODS_ALL),
                                      "--seeds", "0,1", "--tag", "proxy", "--fresh"])),
            ("3/7 proxy eval: deep methods (DANN / LMMD) on A_P3 + A_P4",
             lambda: proxy_eval.main(["--proxies", "A_P3,A_P4",
                                      "--methods", ",".join(C.METHODS_DEEP),
                                      "--seeds", "0", "--tag", "proxy_deep", "--fresh"])),
            ("4/7 file-level methods on proxies (selects the labelling method)",
             lambda: file_level.proxy_main(
                 ["--proxies", ",".join(C.PROXY_ORDER),
                  "--methods", ",".join(file_level.FILE_METHODS),
                  "--seeds", "0,1,2", "--tag", "proxy_file"])),
            ("5/7 window-level target inference (reference)",
             lambda: target_infer.main([])),
            ("6/7 FINAL labelling: file-level + class-existence constraint",
             lambda: relabel.main([])),
            ("7/7 visualisation", lambda: visualize.main()),
        ]

    for title, fn in steps:
        ok = _step(title, fn)
        log.append(f"{title}: {'OK' if ok else 'FAILED'}")

    text = ["=" * 72, "TASK-3 PIPELINE RUN LOG", "=" * 72,
            f"python {sys.version.split()[0]} | numpy {__import__('numpy').__version__} | "
            f"scipy {__import__('scipy').__version__} | seeds {C.SEEDS}",
            f"total elapsed {(time.time()-t0)/60:.1f} min", ""] + log
    (C.RES_DIR / "run_log.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n" + "\n".join(text))
    return log


if __name__ == "__main__":
    main()
