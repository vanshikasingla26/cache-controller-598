"""End-to-end pipeline (RQ1-RQ5 on one workload family; RQ6 = held-out servers).

  1. Tune the detector on TRAIN servers only (F1 of shift starts, then delay).
  2. Collect re-ranker training rows on TRAIN servers inside detector-flagged
     windows (ARC makes the actual decisions, so rows reflect ARC's state).
  3. Train logistic regression and gradient-boosted trees.
  4. On each held-out TEST server, compare: ARC, LRU, oracle ceilings, and the
     model gated always / randomly / by ground truth / by our detector.
  5. Bootstrap 95% CI across test servers for the detector-gated gain vs ARC.

Transition vs steady split for EVALUATION uses ground truth (first --win
accesses after each true shift), never the detector.

  python scripts/experiment.py                  # synthetic servers
  python scripts/experiment.py --quick          # 3 train / 3 test, smoke run
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.detector import (DetectorConfig, arc_adaptation_delay,  # noqa: E402
                                 run_detector, score_detection)
from shiftcache.models import FastLR, fit_models  # noqa: E402
from shiftcache.policies import (ARC, LRU, ControlledARC, OracleCrossListARC,  # noqa: E402
                                 next_use)
from scripts.workloads import TEST_SEEDS, TRAIN_SEEDS, random_server  # noqa: E402


def tune_detector(traces, tol):
    grid = itertools.product([0.5, 0.6, 0.7, 0.8], [2, 3, 4], [5, 10, 20])
    best = None
    for th, per, ref in grid:
        cfg = DetectorConfig(start_thresh=th, persist=per, ref_windows=ref)
        tp = fp = fn = 0
        delays = []
        for tr in traces:
            _, d = run_detector(tr.keys, cfg)
            s = score_detection(d.starts, tr.transitions, tol)
            tp, fp, fn = tp + s["tp"], fp + s["fp"], fn + s["fn"]
            delays += s["delays"]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        key = (round(f1, 4), -np.mean(delays) if delays else -1e9)
        if best is None or key > best[0]:
            best = (key, cfg, tp, fp, fn)
    (f1, negd), cfg, tp, fp, fn = best
    print(f"detector tuned on train: {cfg}  F1={f1} tp={tp} fp={fp} fn={fn} mean delay={-negd:.0f}")
    return cfg


def random_gate(n, det_mask, rng):
    """Same number and lengths of 'on' intervals as the detector, placed at random."""
    edges = np.flatnonzero(np.diff(np.r_[0, det_mask.astype(np.int8), 0]))
    lens = edges[1::2] - edges[::2]
    m = np.zeros(n, dtype=bool)
    for L in lens:
        s = int(rng.integers(0, max(n - L, 1)))
        m[s:s + L] = True
    return m


def bootstrap_ci(x, B=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--win", type=int, default=50_000, help="ground-truth transition window")
    ap.add_argument("--tol", type=int, default=20_000, help="detection match tolerance")
    ap.add_argument("--w-max", type=float, default=1.0)
    ap.add_argument("--models", nargs="+", default=["lr", "gbt"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-always", nargs="*", default=[], help="models to skip always-on for (slow)")
    ap.add_argument("--collect-every", type=int, default=4, help="subsample gated decisions for rows")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr_seeds, te_seeds = (TRAIN_SEEDS[:3], TEST_SEEDS[:3]) if a.quick else (TRAIN_SEEDS, TEST_SEEDS)
    train = [random_server(s) for s in tr_seeds]
    test = [random_server(s) for s in te_seeds]

    # 1. detector
    cfg = tune_detector(train, a.tol)

    # 2. training rows from detector-flagged windows on train servers
    X, y, g = [], [], []
    for tr in train:
        cap = max(1, int(tr.footprint * a.fraction))
        mask, _ = run_detector(tr.keys, cfg)
        c = ControlledARC(cap, k=a.k, gate=mask, collect=True, nu=next_use(tr.keys))
        c.collect_every = a.collect_every
        simulate(tr, c)
        X.append(c.rows)
        y.append(c.labels)
        g += [f"{tr.name}:{i}" for i in set(c.groups.tolist())]
        del c
    X, y = np.vstack(X), np.concatenate(y)
    print(f"training rows: {len(y):,} from {len(set(g)):,} eviction decisions, positive rate {y.mean():.3f}")

    # 3. models
    models = fit_models(X, y, which=a.models)
    del X, y, train

    # 4. evaluation
    rows, det_rows = [], []
    rng = np.random.default_rng(0)
    for tr in test:
        cap = max(1, int(tr.footprint * a.fraction))
        nu = next_use(tr.keys)
        n = len(tr)
        gt = transition_mask(n, tr.transitions, a.win)
        t0 = time.perf_counter()
        det, d = run_detector(tr.keys, cfg)
        det_us = 1e6 * (time.perf_counter() - t0) / n
        rnd = random_gate(n, det, rng)
        s = score_detection(d.starts, tr.transitions, a.tol)

        # held-out AUC of each model on this server's transition rows
        c = ControlledARC(cap, k=a.k, gate=det, collect=True, nu=nu)
        c.collect_every = a.collect_every
        arc_res = simulate(tr, ARC(cap))
        simulate(tr, c)
        auc = {}
        if len(c.labels):
            from sklearn.metrics import roc_auc_score
            Xt, yt = c.rows, c.labels
            if 0 < yt.mean() < 1:
                auc = {m: roc_auc_score(yt, mdl.predict_proba(Xt)[:, 1]) for m, mdl in models.items()}

        for ad in arc_adaptation_delay(arc_res.p_trace, 1000, tr.transitions, n, cap):
            match = next((dd for dd in d.starts if ad["shift"] <= dd <= ad["shift"] + a.tol), None)
            det_rows.append({"trace": tr.name, **ad,
                             "detector_delay": None if match is None else match - ad["shift"]})

        del c
        variants = [("LRU", lambda: LRU(cap)), ("ARC", lambda: ARC(cap)),
                    ("Oracle-X[gt]", lambda: OracleCrossListARC(cap, nu, k=a.k, active=gt)),
                    ("Oracle-X[det]", lambda: OracleCrossListARC(cap, nu, k=a.k, active=det))]
        for m, mdl in models.items():
            for gname, gate in [("always", "always"), ("random", rnd), ("gt", gt), ("det", det)]:
                if gname == "always" and m in a.skip_always:
                    continue
                variants.append((f"{m}[{gname}]", lambda mdl=mdl, gate=gate: ControlledARC(
                    cap, k=a.k, gate=gate, model=mdl, w_max=a.w_max)))
        for name, make in variants:
            pol = make()
            r = simulate(tr, pol)
            row = {"trace": tr.name, "policy": name, "overall_hr": r.hit_rate,
                   **split_hit_rates(r, gt), "latency_us": r.mean_latency_us(), "wall_s": r.wall_s}
            if isinstance(pol, ControlledARC):
                row.update(pol.stats())
                row["calls_per_evict"] = pol.n_model_calls / max(1, int((~r.hits).sum()))
                row["auc"] = auc.get(name.split("[")[0])
            rows.append(row)
            del pol, r
        rows.append({"trace": tr.name, "policy": "_detector", "precision": s["precision"],
                     "recall": s["recall"], "flag_frac": det.mean(), "det_us_per_access": det_us})
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "experiment_raw.partial.csv"), index=False)
        print(f"  {tr.name}: done  (detector P={s['precision']:.2f} R={s['recall']:.2f}, "
              f"flags {det.mean():.1%} of accesses)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "experiment_raw.csv"), index=False)
    pd.DataFrame(det_rows).to_csv(os.path.join(a.out, "detection_vs_arc.csv"), index=False)

    pol = df[df.policy != "_detector"].copy()
    arc = pol[pol.policy == "ARC"].set_index("trace")
    for col in ("transition_hr", "steady_hr", "overall_hr"):
        pol[f"d_{col}"] = pol.apply(lambda r: r[col] - arc.loc[r.trace, col], axis=1)

    pd.set_option("display.width", 200)
    summ = pol.groupby("policy", sort=False)[
        ["overall_hr", "transition_hr", "steady_hr", "d_transition_hr", "d_steady_hr", "d_overall_hr"]
    ].mean()
    extra = pol.groupby("policy", sort=False)[["calls_per_evict", "us_per_call", "auc"]].mean()
    print("\n=== mean over held-out servers (hit rates; d_ = minus ARC, in points x100) ===")
    out = summ.join(extra)
    for c in out.columns:
        if c.startswith("d_"):
            out[c] = out[c] * 100
    print(out.round(3).to_string())

    print("\n=== 95% bootstrap CI across held-out servers (points) ===")
    for p in [x for x in pol.policy.unique() if x not in ("ARC",)]:
        sub = pol[pol.policy == p]
        m, lo, hi = bootstrap_ci(sub.d_transition_hr * 100)
        m2, lo2, hi2 = bootstrap_ci(sub.d_steady_hr * 100)
        print(f"  {p:<16} transition {m:+.2f} [{lo:+.2f}, {hi:+.2f}]   steady {m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]")

    det = df[df.policy == "_detector"]
    dr = pd.DataFrame(det_rows)
    print(f"\n=== detection (held-out): precision {det.precision.mean():.2f}, recall {det.recall.mean():.2f}, "
          f"{det.det_us_per_access.mean():.2f} us/access")
    reacted = dr[dr.t_half.notna()]
    print(f"  detector delay median {dr.detector_delay.median():,.0f}; ARC p half-swing median "
          f"{reacted.t_half.median():,.0f} ({len(reacted)}/{len(dr)} shifts where p moved >=5% of c)")
    both = reacted[reacted.detector_delay.notna()]
    print(f"  detector earlier than ARC half-swing on {int((both.detector_delay < both.t_half).sum())}/{len(both)}")
    out.to_csv(os.path.join(a.out, "experiment_summary.csv"))


if __name__ == "__main__":
    main()
