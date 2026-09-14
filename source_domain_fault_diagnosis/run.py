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
    skip_cnn = "--skip-cnn" in argv
    skip_noise = "--skip-noise" in argv
    t0 = time.time()
    log = []

    for p in sorted(C.FIG_DIR.glob("*.png")):
        p.unlink()

    import make_splits
    import train_eval
    import report
    import explain
    import model_cnn1d
    import noise_robustness

    if quick:
        steps = [
            ("1/3 make splits", lambda: make_splits.main()),
            ("2/3 quick train/eval",
             lambda: train_eval.main(["--protocols", "P1", "--feature-sets", "F2",
                                      "--models", "lr,svm", "--seeds", "0",
                                      "--tag", "quick", "--fresh"])),
            ("3/3 report", lambda: report.main()),
        ]
    else:
        steps = [
            ("1/6 make splits", lambda: make_splits.main()),
            ("2/6 stage1: protocols x feature sets x sklearn models",
             lambda: train_eval.main(["--tag", "stage1",
                                      "--protocols", ",".join(C.PROTOCOLS),
                                      "--feature-sets", ",".join(C.MAIN_FEATURE_SETS),
                                      "--models", ",".join(C.MAIN_MODELS),
                                      "--seeds", ",".join(map(str, C.SEEDS))])),
            ("3/6 stage2: MLP + soft-voting ensemble",
             lambda: train_eval.main(["--tag", "stage2",
                                      "--protocols", "P1,P2,P3",
                                      "--feature-sets", "F2,F3",
                                      "--models", "mlp,vote",
                                      "--seeds", "0"])),
            ("4/6 explain: feature-group ablation", lambda: explain.main()),
            ("5/6 M5: raw-signal 1D-CNN (WDCNN)",
             lambda: model_cnn1d.main(["--protocols", "P1,P2", "--seeds", "0",
                                       "--epochs", "40"])),
            ("6/6 report: summary + figures", lambda: report.main()),
        ]
        if not skip_noise:
            import noise_robustness
            steps.append(("7/7 noise robustness (SNR 20/10/5/0 dB)",
                          lambda: noise_robustness.main(["--snrs", "20,10,5,0"])))

    for title, fn in steps:
        ok = _step(title, fn)
        log.append(f"{title}: {'OK' if ok else 'FAILED'}")

    text = ["=" * 72, "TASK-2 PIPELINE RUN LOG", "=" * 72,
            f"python {sys.version.split()[0]} | numpy {__import__('numpy').__version__} | "
            f"scipy {__import__('scipy').__version__} | seeds {C.SEEDS}",
            f"total elapsed {(time.time()-t0)/60:.1f} min", ""] + log
    (C.RES_DIR / "run_log.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n" + "\n".join(text))
    return log


if __name__ == "__main__":
    main()
