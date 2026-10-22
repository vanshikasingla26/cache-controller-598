"""Real-data experiment on MSR-Cambridge traces (oracleGeneral .zst files).

Design (fixed BEFORE looking at any result):
  * Slice: first --side requests of each volume.
  * Split by server FAMILY so related volumes never sit on both sides:
      TRAIN  hm_0, prn_0, prn_1, src1_0, src1_1, web_2
      TEST   proj_0, proj_1, proj_2, proj_4, prxy_0, usr_1, usr_2
  * Known change points: a spliced trace is volume A's slice followed by volume B's slice
    (keys of B moved to a disjoint namespace). The true shift is the splice index. This is an
    artificial but real-data shift: both sides are real traffic with different statistics.
  * Detectors (fixed-threshold and self-calibrating) are tuned on TRAIN splices only.
  * The re-ranker is trained on rows collected inside detector-flagged windows of TRAIN splices.
  * TEST evaluation: (a) TEST splices with known shift, (b) natural (unspliced) TEST volumes.
  * Confidence intervals resample whole volumes (pairs sharing a volume are not independent).
  * Cache = --fraction of the trace's unique objects; unit-size objects.

  python -u scripts/real_experiment.py --dir /path/to/msr_zst --out results/real
"""
import argparse
import gc
import glob
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.detector import DetectorConfig, ShiftDetector, run_detector, score_detection  # noqa: E402
from shiftcache.detector_adaptive import AdaptiveConfig, detect_from_overlaps, run_adaptive  # noqa: E402
from shiftcache.models import fit_models  # noqa: E402
from shiftcache.policies import ARC, LRU, ControlledARC, next_use  # noqa: E402
from shiftcache.realdata import load_oracle_general  # noqa: E402
from shiftcache.trace import Trace  # noqa: E402
from scripts.experiment import random_gate  # noqa: E402
from scripts.lru_gate_control import LRUGatedARC  # noqa: E402

TRAIN_VOLS = ["hm_0", "prn_0", "prn_1", "src1_0", "src1_1", "web_2"]
TEST_VOLS = ["proj_0", "proj_1", "proj_2", "proj_4", "prxy_0", "usr_1", "usr_2"]
OFFSET = 1 << 40
SYNTH_FIXED = DetectorConfig(window=1000, start_thresh=0.8, end_thresh=0.8, persist=2,
                             ref_windows=5, min_hold=50, max_hold=150)


# ------------------------------------------------------------------ data
def load_vols(d, names, n):
    out = {}
    for nm in names:
        p = os.path.join(d, f"msr_{nm}_oracleGeneral.zst")
        k = load_oracle_general(p, max_requests=n).keys
        out[nm] = k
    return out


def splice(vols, a, b, side):
    ka, kb = vols[a][:side], vols[b][:side]
    keys = np.concatenate([ka, kb + OFFSET])
    n = len(keys)
    return Trace(keys, np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool),
                 name=f"{a}>{b}", transitions=[len(ka)])


def natural(vols, a, n):
    k = vols[a][:n]
    return Trace(k, np.arange(len(k), dtype=np.int64), np.zeros(len(k), dtype=bool), name=a)


# ------------------------------------------------------------------ detectors from overlaps
_OV = {}


def overlaps(tr, window, ref):
    key = (tr.name, len(tr), window, ref)
    if key not in _OV:
        d = ShiftDetector(DetectorConfig(window=window, ref_windows=ref))
        for t, k in enumerate(tr.keys.tolist()):
            d.update(k, t)
        _OV[key] = d.overlaps
    return _OV[key]


def detect_fixed_from_overlaps(ov, cfg):
    """Same state machine as ShiftDetector._close_window, run on a precomputed overlap series."""
    starts, ends, in_t, low, high, held = [], [], False, 0, 0, 0
    for i, o in enumerate(ov):
        t_next = (i + 2) * cfg.window
        if not in_t:
            low = low + 1 if o < cfg.start_thresh else 0
            if low >= cfg.persist:
                in_t, held, high = True, 0, 0
                starts.append(t_next)
        else:
            held += 1
            high = high + 1 if o >= cfg.end_thresh else 0
            if (held >= cfg.min_hold and high >= cfg.persist) or held >= cfg.max_hold:
                in_t, low = False, 0
                ends.append(t_next)
    return starts, ends


def mask_from(starts, ends, n):
    m = np.zeros(n, dtype=bool)
    ends = list(ends)
    for s in starts:
        e = next((x for x in ends if x > s), n)
        m[s:min(e, n)] = True
    return m


def run_detector_cfg(kind, cfg, tr):
    if kind == "fixed":
        st, en = detect_fixed_from_overlaps(overlaps(tr, cfg.window, cfg.ref_windows), cfg)
    else:
        st, en = detect_from_overlaps(overlaps(tr, cfg.window, cfg.ref_windows), cfg)
    return st, en


def tune(kind, train, tol):
    if kind == "fixed":
        grid = [DetectorConfig(window=1000, start_thresh=th, end_thresh=th, persist=p, ref_windows=r)
                for th, p, r in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 3, 4], [5, 10, 20])]
    else:
        grid = [AdaptiveConfig(window=w, ref_windows=r, z_start=zs, z_end=ze, persist=p)
                for w, r, zs, ze, p in itertools.product([1000, 2000], [5, 10], [2, 3, 4, 5, 6], [1.0, 2.0], [2, 3, 4])]
    best = None
    for cfg in grid:
        tp = fp = fn = 0
        delays = []
        for tr in train:
            st, _ = run_detector_cfg(kind, cfg, tr)
            s = score_detection(st, tr.transitions, tol)
            tp, fp, fn = tp + s["tp"], fp + s["fp"], fn + s["fn"]
            delays += s["delays"]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        key = (round(f1, 4), -np.mean(delays) if delays else -1e9)
        if best is None or key > best[0]:
            best = (key, cfg, tp, fp, fn)
    (f1, nd), cfg, tp, fp, fn = best
    print(f"  {kind} detector tuned on TRAIN splices: {cfg}\n    F1={f1} tp={tp} fp={fp} fn={fn} "
          f"mean delay={-nd:.0f}", flush=True)
    return cfg, f1


def cluster_ci(df, col, group, B=10_000, seed=0):
    x = df.groupby(group)[col].mean().values
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(x)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--side", type=int, default=1_000_000, help="requests per side of a splice")
    ap.add_argument("--natural-n", type=int, default=2_000_000)
    ap.add_argument("--train-pairs", type=int, default=12)
    ap.add_argument("--test-pairs", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--win", type=int, default=50_000)
    ap.add_argument("--tol", type=int, default=20_000)
    ap.add_argument("--collect-every", type=int, default=4)
    ap.add_argument("--skip-always", action="store_true")
    ap.add_argument("--skip-splices", action="store_true",
                    help="skip stage [3] (spliced TEST evaluation); reuse real_splice_raw.csv")
    ap.add_argument("--max-onehit", type=float, default=0.90,
                    help="drop volumes whose slice has a larger one-hit-object share (pure scans)")
    ap.add_argument("--train-vols", nargs="*", default=TRAIN_VOLS)
    ap.add_argument("--test-vols", nargs="*", default=TEST_VOLS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    T0 = time.time()

    n_load = max(a.side, a.natural_n)
    vols = load_vols(a.dir, list(a.train_vols) + list(a.test_vols), n_load)
    print("loaded:", {k: len(v) for k, v in vols.items()}, flush=True)
    # exclusion rule (descriptive statistic of the slice, decided before any result)
    dropped = {}
    for nm in list(vols):
        for n_ in {a.side, a.natural_n}:
            _, cnt = np.unique(vols[nm][:n_], return_counts=True)
            oh = float(np.mean(cnt == 1))
            if oh > a.max_onehit:
                dropped[nm] = round(oh, 3)
    a.train_vols = [v for v in a.train_vols if v not in dropped]
    a.test_vols = [v for v in a.test_vols if v not in dropped]
    print("excluded (one-hit share above %.0f%%):" % (100 * a.max_onehit), dropped,
          "\n  train vols:", a.train_vols, "\n  test vols:", a.test_vols, flush=True)

    def pick(names, k):
        allp = [(x, y) for x in names for y in names if x != y]
        idx = rng.permutation(len(allp))[:k]
        return [allp[i] for i in sorted(idx)]

    train_pairs, test_pairs = pick(a.train_vols, a.train_pairs), pick(a.test_vols, a.test_pairs)
    print("train pairs:", train_pairs, "\ntest pairs:", test_pairs, flush=True)
    train = [splice(vols, x, y, a.side) for x, y in train_pairs]
    test = [splice(vols, x, y, a.side) for x, y in test_pairs]

    # 1. detectors tuned on TRAIN splices
    print("\n[1] tuning detectors", flush=True)
    cfg_f, f1_f = tune("fixed", train, a.tol)
    cfg_a, f1_a = tune("adaptive", train, a.tol)
    kind, cfg = ("adaptive", cfg_a) if f1_a >= f1_f else ("fixed", cfg_f)
    print(f"  -> using the {kind} detector as the gate (higher TRAIN F1)", flush=True)

    def gate_for(tr):
        st, en = run_detector_cfg(kind, cfg, tr)
        return mask_from(st, en, len(tr)), st, en

    # 2. detector evaluation on TEST splices (known shift)
    det_rows = []
    for tr in test:
        n = len(tr)
        for lab, (k_, c_) in [("fixed (synthetic-tuned)", ("fixed", SYNTH_FIXED)),
                              ("fixed (real-tuned)", ("fixed", cfg_f)),
                              ("adaptive (real-tuned)", ("adaptive", cfg_a))]:
            st, en = run_detector_cfg(k_, c_, tr)
            s = score_detection(st, tr.transitions, a.tol)
            det_rows.append({"pair": tr.name, "dst": tr.name.split(">")[1], "detector": lab,
                             "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                             "delay": np.mean(s["delays"]) if s["delays"] else np.nan,
                             "flagged": mask_from(st, en, n).mean()})
    ddf = pd.DataFrame(det_rows)
    ddf.to_csv(os.path.join(a.out, "real_detector_test.csv"), index=False)
    print("\n=== detectors on TEST splices (known shift at the splice) ===", flush=True)
    for lab in ddf.detector.unique():
        d = ddf[ddf.detector == lab]
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        print(f"  {lab:<26} recall {tp / max(tp + fn, 1):.2f}  precision {tp / max(tp + fp, 1):.2f}  "
              f"delay {d.delay.mean():,.0f}  flagged {100 * d.flagged.mean():.0f}% of accesses "
              f"(tp {tp}, fp {fp}, fn {fn} over {len(d)} splices)", flush=True)

    # 3. training rows on TRAIN splices (detector-gated windows)
    print("\n[2] collecting re-ranker training rows", flush=True)
    X, y = [], []
    for tr in train:
        cap = max(1, int(tr.footprint * a.fraction))
        mask, _, _ = gate_for(tr)
        if mask.sum() == 0:
            continue
        c = ControlledARC(cap, k=a.k, gate=mask, collect=True, nu=next_use(tr.keys))
        c.collect_every = a.collect_every
        simulate(tr, c)
        if len(c.labels):
            X.append(c.rows)
            y.append(c.labels)
        del c
    X, y = np.vstack(X), np.concatenate(y)
    print(f"  {len(y):,} training rows, positive rate {y.mean():.3f}", flush=True)
    models = fit_models(X, y, which=["lr", "gbt"])
    del X, y

    # 4. spliced TEST evaluation
    print("\n[3] TEST splices", flush=True)
    rows = []
    for tr in ([] if a.skip_splices else test):
        n, cap = len(tr), max(1, int(tr.footprint * a.fraction))
        gt = transition_mask(n, tr.transitions, a.win)
        det, _, _ = gate_for(tr)
        rnd = random_gate(n, det, rng)
        variants = [("ARC", lambda: ARC(cap)), ("LRU", lambda: LRU(cap)),
                    ("LRU-gated[det]", lambda: LRUGatedARC(cap, k=a.k, gate=det)),
                    ("LRU-gated[gt]", lambda: LRUGatedARC(cap, k=a.k, gate=gt))]
        for m in ("lr", "gbt"):
            for gname, gate in [("det", det), ("gt", gt), ("random", rnd)] + ([] if (a.skip_always or m == "lr") else [("always", "always")]):
                variants.append((f"{m}[{gname}]", lambda mm=models[m], g=gate: ControlledARC(
                    cap, k=a.k, gate=g, model=mm)))
        for name, make in variants:
            t0 = time.time()
            r = simulate(tr, make())
            rows.append({"pair": tr.name, "dst": tr.name.split(">")[1], "policy": name,
                         "overall_hr": r.hit_rate, **split_hit_rates(r, gt), "wall_s": time.time() - t0,
                         "gate_frac": float(det.mean())})
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "real_splice_raw.partial.csv"), index=False)
        print(f"  {tr.name}: done ({(time.time() - T0) / 60:.1f} min elapsed, "
              f"detector gates {det.mean():.0%} of accesses)", flush=True)
    if not a.skip_splices:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "real_splice_raw.csv"), index=False)

    # 5. natural (unspliced) TEST volumes
    print("\n[4] natural TEST volumes", flush=True)
    nrows = []
    for v in a.test_vols:
        tr = natural(vols, v, a.natural_n)
        n, cap = len(tr), max(1, int(tr.footprint * a.fraction))
        det, st, _ = gate_for(tr)
        variants = [("ARC", lambda: ARC(cap)), ("LRU-gated[det]", lambda: LRUGatedARC(cap, k=a.k, gate=det)),
                    ("gbt[det]", lambda: ControlledARC(cap, k=a.k, gate=det, model=models["gbt"])),
                    ("lr[det]", lambda: ControlledARC(cap, k=a.k, gate=det, model=models["lr"]))]
        if not a.skip_always:
            variants.append(("gbt[always]", lambda: ControlledARC(cap, k=a.k, gate="always", model=models["gbt"])))
        for name, make in variants:
            pol = make()
            r = simulate(tr, pol)
            nrows.append({"vol": v, "policy": name, "overall_hr": r.hit_rate, "gate_frac": float(det.mean()),
                          "n_starts": len(st)})
            del pol, r
            gc.collect()
        pd.DataFrame(nrows).to_csv(os.path.join(a.out, "real_natural_raw.partial.csv"), index=False)
        print(f"  {v}: gate on {det.mean():.0%} of accesses, {len(st)} starts", flush=True)
        del tr, det
        gc.collect()
    pd.DataFrame(nrows).to_csv(os.path.join(a.out, "real_natural_raw.csv"), index=False)

    # 6. summaries (standalone: python scripts/summarize_real.py --out results/real)
    from scripts.summarize_real import summarize
    print("\n" + summarize(a.out))
    print(f"\ntotal {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
