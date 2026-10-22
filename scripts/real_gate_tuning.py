"""Tune the shift gate for HIT RATE (not shift-detection F1) on real MSR traces.

Why: with F1-tuned detectors the gated controller lost to ARC on real traces and the detector's
windows were worse places to act than random ones (README). Detection score is the wrong objective.

Proxy (label-free, cheap): with whole-trace hit arrays of ARC and LRU,
    gain(G) = sum over accesses in gate G of (LRU_hit - ARC_hit) / n * 100   [points]
i.e. "what if we behaved like LRU inside the gate". It ignores state carry-over at gate edges, so it
is validated against the real LRU-gated simulation (correlation on TRAIN and TEST pairs) before use.

Protocol (same split as real_experiment.py): tune on TRAIN splices + natural TRAIN volumes only, then
evaluate on TEST splices + natural TEST volumes with real simulation of LRU-gated ARC.
Also reports the CEILING of LRU-like switching: per 1000-access window, gate exactly when LRU beats ARC.

  python -u scripts/real_gate_tuning.py --dir <folder with msr_*.zst> --out results/real
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate  # noqa: E402
from shiftcache.detector import DetectorConfig  # noqa: E402
from shiftcache.detector_adaptive import AdaptiveConfig, detect_from_overlaps  # noqa: E402
from shiftcache.policies import ARC, LRU  # noqa: E402
from scripts.lru_gate_control import LRUGatedARC  # noqa: E402
from scripts.real_experiment import (cluster_ci, detect_fixed_from_overlaps, load_vols, mask_from,  # noqa: E402
                                     natural, overlaps, splice)

TRAIN_V, TEST_V = ["hm_0", "prn_0", "prn_1", "src1_1"], ["proj_0", "proj_4", "prxy_0", "usr_2"]
OLD_FIXED = DetectorConfig(window=1000, start_thresh=0.5, end_thresh=0.5, persist=4, ref_windows=5,
                           min_hold=50, max_hold=150)   # the F1-tuned gate used in the previous run


def pairs(rng, names, k):
    allp = [(x, y) for x in names for y in names if x != y]
    return [allp[i] for i in sorted(rng.permutation(len(allp))[:k])]


def hits(tr, cap):
    return simulate(tr, ARC(cap)).hits, simulate(tr, LRU(cap)).hits


def proxy(mask, h_arc, h_lru):
    return float(((h_lru[mask].astype(np.int8) - h_arc[mask].astype(np.int8)).sum()) / len(mask) * 100)


def ceiling(h_arc, h_lru, w=1000):
    n = (len(h_arc) // w) * w
    d = (h_lru[:n].astype(np.int8) - h_arc[:n].astype(np.int8)).reshape(-1, w).sum(axis=1)
    return float(np.clip(d, 0, None).sum() / len(h_arc) * 100), float(d.sum() / len(h_arc) * 100)


def gate_mask(kind, cfg, tr):
    ov = overlaps(tr, cfg.window, cfg.ref_windows)
    st, en = (detect_fixed_from_overlaps if kind == "fixed" else detect_from_overlaps)(ov, cfg)
    return mask_from(st, en, len(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--side", type=int, default=500_000)
    ap.add_argument("--natural-n", type=int, default=1_500_000)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    T0 = time.time()
    rng = np.random.default_rng(0)
    tr_pairs, te_pairs = pairs(rng, TRAIN_V, 10), pairs(rng, TEST_V, 12)   # same draws as real_experiment.py
    vols = load_vols(a.dir, TRAIN_V + TEST_V, max(a.side, a.natural_n))
    train = [splice(vols, x, y, a.side) for x, y in tr_pairs] + [natural(vols, v, a.natural_n) for v in TRAIN_V]
    test_sp = [splice(vols, x, y, a.side) for x, y in te_pairs]
    test_nat = [natural(vols, v, a.natural_n) for v in TEST_V]
    cap = lambda tr: max(1, int(tr.footprint * a.fraction))  # noqa: E731
    H = {}
    for tr in train + test_sp + test_nat:
        H[(tr.name, len(tr))] = hits(tr, cap(tr))
    print(f"whole-trace ARC/LRU hit arrays done ({(time.time() - T0) / 60:.1f} min)", flush=True)

    # ceiling of LRU-like switching
    print("\n=== ceiling: gate exactly the 1000-access windows where LRU beats ARC (points, free of detector) ===")
    ceil = []
    for grp, trs in [("train", train), ("test splices", test_sp), ("test natural", test_nat)]:
        c = [ceiling(*H[(t.name, len(t))])[0] for t in trs]
        ceil.append((grp, np.mean(c)))
        print(f"  {grp:<13} mean ceiling {np.mean(c):+.2f}   (always-LRU, no gate: "
              f"{np.mean([ceiling(*H[(t.name, len(t))])[1] for t in trs]):+.2f})", flush=True)

    # tune on TRAIN by the proxy
    grid = ([("fixed", DetectorConfig(window=1000, start_thresh=th, end_thresh=th, persist=p, ref_windows=r))
             for th, p, r in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 3, 4], [5, 10, 20])] +
            [("adaptive", AdaptiveConfig(window=w, ref_windows=r, z_start=zs, z_end=ze, persist=p))
             for w, r, zs, ze, p in itertools.product([1000, 2000], [5, 10], [2, 3, 4, 5, 6], [1.0, 2.0], [2, 3, 4])])
    res = []
    for kind, cfg in grid:
        g = [proxy(gate_mask(kind, cfg, t), *H[(t.name, len(t))]) for t in train]
        res.append((float(np.mean(g)), float(np.mean(np.array(g) > 0)), kind, cfg))
    res.sort(key=lambda r: -r[0])
    print(f"\n=== proxy tuning on TRAIN ({len(train)} traces, {len(grid)} configs, never-gate = 0.00) ===")
    for sc, frac_pos, kind, cfg in res[:5]:
        print(f"  {sc:+.3f} pts  ({frac_pos:.0%} of train traces > 0)  {kind}: {cfg}")
    best_sc, _, kind, cfg = res[0]
    if best_sc <= 0:
        print("  no configuration beats never-gating on TRAIN; selecting the best anyway for reporting")

    # proxy vs real simulation (validation of the proxy), TEST splices
    rows = []
    for t in test_sp + test_nat:
        h_arc, h_lru = H[(t.name, len(t))]
        c = cap(t)
        for lab, (k_, c_) in [("F1-tuned fixed (previous gate)", ("fixed", OLD_FIXED)), ("hit-rate-tuned", (kind, cfg))]:
            m = gate_mask(k_, c_, t)
            real = simulate(t, LRUGatedARC(c, k=a.k, gate=m)).hit_rate - h_arc.mean()
            rows.append({"trace": t.name, "kind": "splice" if ">" in t.name else "natural", "gate": lab,
                         "proxy": proxy(m, h_arc, h_lru), "real": real * 100, "gate_frac": float(m.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "gate_tuning_test.csv"), index=False)
    print("\n=== TEST: LRU-gated ARC with each gate, real simulation, points vs ARC ===")
    for lab in df.gate.unique():
        for kd in ("splice", "natural"):
            s = df[(df.gate == lab) & (df.kind == kd)]
            if kd == "splice":
                s = s.assign(dst=s.trace.str.split(">").str[1])
                m, lo, hi, _ = cluster_ci(s, "real", "dst")
            else:
                m, lo, hi, _ = cluster_ci(s, "real", "trace")
            print(f"  {lab:<32} {kd:<8} overall {m:+.2f} [{lo:+.2f},{hi:+.2f}]  gate on {s.gate_frac.mean():.0%}")
    r = np.corrcoef(df.proxy, df.real)[0, 1]
    print(f"\nproxy vs real simulation on TEST (both gates, {len(df)} runs): correlation {r:.2f}, "
          f"mean abs error {np.abs(df.proxy - df.real).mean():.2f} pts")
    print(f"\ntotal {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
