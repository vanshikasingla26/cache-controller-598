"""Always-on vs detector-triggered learned re-ranker on REAL MSR splices, with overhead.

Closes two proposal-feedback points on real data: (1) compare always-on re-ranking with the triggered
version, (2) report overhead (inference latency per call, model calls per eviction, wall time per access,
model size) and false alarms. Same protocol as real_experiment.py: detectors tuned and the re-ranker trained
on TRAIN splices only (same seeded pairs); the trained models are saved to results/real/models_real.pkl.
Evaluation: one TEST splice per destination volume (always-on is slow, so 4 pairs).

  python -u scripts/real_always_overhead.py --dir data/msr --out results/real
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.detector import score_detection  # noqa: E402
from shiftcache.models import fit_models  # noqa: E402
from shiftcache.policies import ARC, ControlledARC, next_use  # noqa: E402
from scripts.lru_gate_control import LRUGatedARC  # noqa: E402
from scripts.real_experiment import load_vols, mask_from, run_detector_cfg, splice, tune  # noqa: E402

TRAIN_V, TEST_V = ["hm_0", "prn_0", "prn_1", "src1_1"], ["proj_0", "proj_4", "prxy_0", "usr_2"]


def pairs(rng, names, k):
    allp = [(x, y) for x in names for y in names if x != y]
    return [allp[i] for i in sorted(rng.permutation(len(allp))[:k])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/msr")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--side", type=int, default=500_000)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--win", type=int, default=50_000)
    ap.add_argument("--tol", type=int, default=20_000)
    a = ap.parse_args()
    T0 = time.time()
    rng = np.random.default_rng(0)
    tr_pairs, te_pairs = pairs(rng, TRAIN_V, 10), pairs(rng, TEST_V, 12)   # same draws as real_experiment.py
    vols = load_vols(a.dir, TRAIN_V + TEST_V, a.side)
    train = [splice(vols, x, y, a.side) for x, y in tr_pairs]
    cfg_f, f1_f = tune("fixed", train, a.tol)
    cfg_a, f1_a = tune("adaptive", train, a.tol)
    kind, cfg = ("adaptive", cfg_a) if f1_a >= f1_f else ("fixed", cfg_f)
    print(f"gate: {kind} (same rule as real_experiment.py)", flush=True)

    mpath = os.path.join(a.out, "models_real.pkl")
    if os.path.exists(mpath):                      # resume: reuse the saved re-ranker
        with open(mpath, "rb") as fh:
            blob = fh.read()
        models, train_s, y = pickle.loads(blob), float("nan"), np.zeros(0)
        print(f"loaded saved model {mpath}", flush=True)
    X, y_list = [], []
    for tr in ([] if os.path.exists(mpath) else train):
        cap = max(1, int(tr.footprint * a.fraction))
        st, en = run_detector_cfg(kind, cfg, tr)
        mask = mask_from(st, en, len(tr))
        if not mask.any():
            continue
        c = ControlledARC(cap, k=a.k, gate=mask, collect=True, nu=next_use(tr.keys))
        c.collect_every = 4
        simulate(tr, c)
        if len(c.labels):
            X.append(c.rows)
            y_list.append(c.labels)
    if X:
        X, y = np.vstack(X), np.concatenate(y_list)
        t = time.perf_counter()
        models = fit_models(X, y, which=["gbt"])
        train_s = time.perf_counter() - t
        blob = pickle.dumps(models)
        with open(mpath, "wb") as fh:
            fh.write(blob)
    print(f"trained gbt on {len(y):,} rows in {train_s:.1f} s; model size {len(blob) / 1e6:.2f} MB "
          f"({(time.time() - T0) / 60:.1f} min)", flush=True)

    seen, test_pairs = set(), []
    for x_, y_ in te_pairs:                  # first pair for each destination volume
        if y_ not in seen:
            seen.add(y_)
            test_pairs.append((x_, y_))
    raw = os.path.join(a.out, "always_overhead_raw.csv")
    rows = pd.read_csv(raw).to_dict("records") if os.path.exists(raw) else []
    done = {(r["pair"], r["policy"]) for r in rows}
    for x_, y_ in test_pairs:
        tr = splice(vols, x_, y_, a.side)
        n, cap = len(tr), max(1, int(tr.footprint * a.fraction))
        gt = transition_mask(n, tr.transitions, a.win)
        st, en = run_detector_cfg(kind, cfg, tr)
        det = mask_from(st, en, n)
        sc = score_detection(st, tr.transitions, a.tol)
        for name, make in [("ARC", lambda: ARC(cap)),
                           ("LRU-gated[det]", lambda: LRUGatedARC(cap, k=a.k, gate=det)),
                           ("gbt[det]", lambda: ControlledARC(cap, k=a.k, gate=det, model=models["gbt"])),
                           ("gbt[always]", lambda: ControlledARC(cap, k=a.k, gate="always", model=models["gbt"]))]:
            if (tr.name, name) in done:
                continue
            pol = make()
            t = time.perf_counter()
            r = simulate(tr, pol)
            wall = time.perf_counter() - t
            row = {"pair": tr.name, "dst": y_, "policy": name, "overall_hr": r.hit_rate, **split_hit_rates(r, gt),
                   "wall_s": wall, "us_per_access": wall / n * 1e6, "gate_frac": float(det.mean()),
                   "false_starts": sc["fp"], "detected": sc["tp"]}
            if isinstance(pol, ControlledARC):
                row.update(pol.stats())
                row["calls_per_evict"] = pol.n_model_calls / max(1, int((~r.hits).sum()))
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "always_overhead_raw.csv"), index=False)
            print(f"  {tr.name} {name}: {wall:.0f} s ({(time.time() - T0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    arc = df[df.policy == "ARC"].set_index("pair")
    for c in ("transition_hr", "overall_hr"):
        df["d_" + c] = df.apply(lambda r: (r[c] - arc.loc[r.pair, c]) * 100, axis=1)
    keep = [c for c in ["d_transition_hr", "d_overall_hr", "us_per_access", "calls_per_evict", "us_per_call",
                        "gate_frac", "false_starts"] if c in df.columns]
    summ = df.groupby("policy", sort=False)[keep].mean().round(3)
    L = [f"REAL MSR splices, one per held-out destination volume ({len(test_pairs)} pairs); mean over pairs",
         summ.to_string(),
         f"\nre-ranker: GBT trained on {len(y):,} rows in {train_s:.1f} s; serialized model {len(blob) / 1e6:.2f} MB;"
         f" gate kind {kind}"]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "always_overhead_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
