"""Fixed-threshold vs self-calibrating detector.

  1. Tune the adaptive detector on the synthetic TRAIN servers only
     (objective as in experiment.py: F1 of shift starts, then mean delay).
  2. Evaluate fixed (Shubham's tuned config) and adaptive on the synthetic TEST servers.
  3. Real-trace checks on the CloudPhysics sample (not used for tuning):
       a. flagged fraction and number of starts on the raw trace (no ground truth),
       b. splices of real segments: B is appended to A with its keys moved to a new
          namespace (a real working-set change, ground truth = the splice point) vs a
          control where B is simply the next real segment (no injected shift).

  python scripts/fetch_sample_trace.py
  python -u scripts/detector_eval.py --out results/detector_eval
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.detector import DetectorConfig, ShiftDetector, run_detector, score_detection  # noqa: E402
from shiftcache.detector_adaptive import AdaptiveConfig, detect_from_overlaps  # noqa: E402
from shiftcache.realdata import load_oracle_general  # noqa: E402
from scripts.experiment import bootstrap_ci  # noqa: E402
from scripts.workloads import TEST_SEEDS, TRAIN_SEEDS, random_server  # noqa: E402

FIXED = DetectorConfig(window=1000, start_thresh=0.8, end_thresh=0.8, persist=2, ref_windows=5,
                       min_hold=50, max_hold=150)   # tuned by experiment.py on synthetic TRAIN
TOL = 20_000
_ov_cache = {}


def overlaps(name, keys, window, ref):
    k = (name, window, ref)
    if k not in _ov_cache:
        d = ShiftDetector(DetectorConfig(window=window, ref_windows=ref))
        for t, key in enumerate(keys.tolist()):
            d.update(key, t)
        _ov_cache[k] = d.overlaps
    return _ov_cache[k]


def flagged_fraction(starts, ends, n):
    on, total, ends = 0, 0, list(ends)
    for s in starts:
        e = next((x for x in ends if x > s), n)
        total += min(e, n) - s
    return total / n


def adaptive_run(name, keys, cfg):
    st, en = detect_from_overlaps(overlaps(name, keys, cfg.window, cfg.ref_windows), cfg)
    return st, en


def fixed_run(keys):
    mask, d = run_detector(keys, FIXED)
    return list(d.starts), list(d.ends)


def tune(train, tol):
    grid = itertools.product([1000, 2000], [5, 10], [3, 4, 5, 6, 8], [1.0, 2.0], [2, 3, 4])
    best = None
    for w, r, zs, ze, per in grid:
        cfg = AdaptiveConfig(window=w, ref_windows=r, z_start=zs, z_end=ze, persist=per)
        tp = fp = fn = 0
        delays = []
        for tr in train:
            st, _ = adaptive_run(tr.name, tr.keys, cfg)
            s = score_detection(st, tr.transitions, tol)
            tp, fp, fn = tp + s["tp"], fp + s["fp"], fn + s["fn"]
            delays += s["delays"]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        key = (round(f1, 4), -np.mean(delays) if delays else -1e9)
        if best is None or key > best[0]:
            best = (key, cfg, tp, fp, fn)
    (f1, negd), cfg, tp, fp, fn = best
    print(f"adaptive tuned on train: {cfg}\n  F1={f1} tp={tp} fp={fp} fn={fn} mean delay={-negd:.0f}",
          flush=True)
    return cfg


def synthetic_eval(test, cfg):
    rows = []
    for tr in test:
        for label, (st, en) in [("fixed", fixed_run(tr.keys)),
                                ("adaptive", adaptive_run(tr.name, tr.keys, cfg))]:
            s = score_detection(st, tr.transitions, TOL)
            rows.append({"trace": tr.name, "detector": label, "tp": s["tp"], "fp": s["fp"],
                         "fn": s["fn"], "delay": np.mean(s["delays"]) if s["delays"] else np.nan,
                         "flagged": flagged_fraction(st, en, len(tr))})
    return pd.DataFrame(rows)


def splices(keys, L=50_000, offsets=(0, 6_000, 12_000)):
    """(name, trace keys, truth) for shift splices and no-shift controls."""
    out = []
    for o in offsets:
        a, b = keys[o:o + L], keys[o + L:o + 2 * L]
        out.append((f"shift@{o}", np.concatenate([a, b + (1 << 40)]), [L]))
        out.append((f"control@{o}", np.concatenate([a, b]), []))
    return out


def real_eval(keys, cfg):
    rows = [{"what": "raw CloudPhysics sample", "detector": lab, "starts": len(st),
             "flagged": flagged_fraction(st, en, len(keys)), "detected": "", "delay": np.nan}
            for lab, (st, en) in [("fixed", fixed_run(keys)),
                                  ("adaptive", adaptive_run("cp_raw", keys, cfg))]]
    for name, k, truth in splices(keys):
        for lab, (st, en) in [("fixed", fixed_run(k)), ("adaptive", adaptive_run("cp_" + name, k, cfg))]:
            s = score_detection(st, truth, TOL)
            rows.append({"what": name, "detector": lab, "starts": len(st),
                         "flagged": flagged_fraction(st, en, len(k)),
                         "detected": ("yes" if s["tp"] else "NO") if truth else "n/a",
                         "delay": np.mean(s["delays"]) if s["delays"] else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/detector_eval")
    ap.add_argument("--trace", default="data/cloudphysics_sample.oracleGeneral.bin")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 200)

    train = [random_server(s) for s in TRAIN_SEEDS]
    cfg = tune(train, TOL)
    del train
    test = [random_server(s) for s in TEST_SEEDS]
    syn = synthetic_eval(test, cfg)
    syn.to_csv(os.path.join(a.out, "synthetic_test.csv"), index=False)
    print("\n=== synthetic held-out servers (8) ===")
    for lab in ("fixed", "adaptive"):
        d = syn[syn.detector == lab]
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        print(f"{lab:<9} precision {tp / max(tp + fp, 1):.2f}  recall {tp / max(tp + fn, 1):.2f}  "
              f"mean delay {d.delay.mean():,.0f}  flagged {100 * d.flagged.mean():.1f}% of accesses  "
              f"(tp {tp} fp {fp} fn {fn})")

    real = load_oracle_general(a.trace)
    rr = real_eval(real.keys, cfg)
    rr.to_csv(os.path.join(a.out, "real_cloudphysics.csv"), index=False)
    print("\n=== real CloudPhysics sample (113,872 requests; NOT used for tuning) ===")
    print(rr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
