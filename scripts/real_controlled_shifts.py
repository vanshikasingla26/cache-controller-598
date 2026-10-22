"""Controlled, one-dimension-at-a-time shifts injected into REAL MSR volumes (grader feedback).

For each volume, take its first --n requests; from the midpoint s onward apply ONE change with a
controlled magnitude (everything before s is untouched real traffic):
  popularity m : remap a fraction m of the post-shift hot set (blocks covering 50% of post-shift
                 accesses) to fresh block ids -> the hot set changes, the access structure stays real
  scan q       : a fraction q of post-shift requests becomes a sequential scan of fresh blocks
  tenant r     : a fraction r of post-shift requests comes from another real volume (interleaved)
Ground truth = s. Per trace we report: ARC's adaptation (p half-swing), detector delay, avoidable misses
(extra MIN - ARC gap right after the shift), and hit-rate change of LRU-gated ARC with the true change
point (diagnostic) and with the detector (the real system), versus plain ARC.

No tuning here. The gate config is the hit-rate-tuned one chosen on TRAIN volumes by
real_gate_tuning.py. Results are reported separately for TRAIN and TEST volumes.

  python -u scripts/real_controlled_shifts.py --dir <folder with msr_*.zst> --out results/real
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.detector import DetectorConfig, arc_adaptation_delay, score_detection  # noqa: E402
from shiftcache.policies import ARC, LRU, Belady, next_use  # noqa: E402
from shiftcache.trace import Trace  # noqa: E402
from scripts.lru_gate_control import LRUGatedARC  # noqa: E402
from scripts.real_experiment import (detect_fixed_from_overlaps, load_vols, mask_from,  # noqa: E402
                                     overlaps)

TRAIN_V, TEST_V = ["hm_0", "prn_0", "prn_1", "src1_1"], ["proj_0", "proj_4", "prxy_0", "usr_2"]
# chosen on TRAIN by real_gate_tuning.py (best hit-rate proxy), see results/real/gate_tuning_run.log
GATE = DetectorConfig(window=1000, start_thresh=0.3, end_thresh=0.3, persist=3, ref_windows=10,
                      min_hold=50, max_hold=150)
FRESH = 1 << 41
SCENARIOS = [("popularity", 0.25), ("popularity", 0.5), ("popularity", 1.0),
             ("scan", 0.1), ("scan", 0.3), ("scan", 0.5),
             ("tenant", 0.25), ("tenant", 0.5)]


def transform(keys, s, dim, mag, other, seed):
    rng = np.random.default_rng(seed)
    pre, post = keys[:s], keys[s:].copy()
    m = len(post)
    if dim == "popularity":
        u, cnt = np.unique(post, return_counts=True)
        order = np.argsort(-cnt)
        cum = np.cumsum(cnt[order]) / cnt.sum()
        hot = u[order[: int(np.searchsorted(cum, 0.5)) + 1]]
        chosen = hot[rng.random(len(hot)) < mag] if mag < 1 else hot
        remap = {int(k): FRESH + i for i, k in enumerate(chosen.tolist())}
        post = np.array([remap.get(int(k), int(k)) for k in post.tolist()], dtype=np.int64)
    elif dim == "scan":
        is_scan = rng.random(m) < mag
        real = post[: m - int(is_scan.sum())]
        out = np.empty(m, dtype=np.int64)
        out[is_scan] = FRESH + np.arange(int(is_scan.sum()))
        out[~is_scan] = real
        post = out
    elif dim == "tenant":
        is_other = rng.random(m) < mag
        n_o = int(is_other.sum())
        out = np.empty(m, dtype=np.int64)
        out[is_other] = other[:n_o] + FRESH          # other volume, disjoint namespace
        out[~is_other] = post[: m - n_o]
        post = out
    return np.concatenate([pre, post])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--win", type=int, default=50_000)
    ap.add_argument("--tol", type=int, default=20_000)
    a = ap.parse_args()
    T0 = time.time()
    vols = load_vols(a.dir, TRAIN_V + TEST_V, a.n)
    s = a.n // 2
    rows = []
    for group, names in (("train", TRAIN_V), ("test", TEST_V)):
        for i, v in enumerate(names):
            other = vols[names[(i + 1) % len(names)]]       # another volume from the same side
            for j, (dim, mag) in enumerate(SCENARIOS):
                k = transform(vols[v], s, dim, mag, other, seed=1000 * i + j)
                n = len(k)
                tr = Trace(k, np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool),
                           name=f"{v}:{dim}{mag}", transitions=[s])
                cap = max(1, int(tr.footprint * a.fraction))
                gt = transition_mask(n, [s], a.win)
                st, en = detect_fixed_from_overlaps(overlaps(tr, GATE.window, GATE.ref_windows), GATE)
                det = mask_from(st, en, n)
                sc = score_detection(st, [s], a.tol)
                r_arc = simulate(tr, ARC(cap))
                ad = arc_adaptation_delay(r_arc.p_trace, 1000, [s], n, cap)[0]
                res = {"ARC": r_arc, "LRU": simulate(tr, LRU(cap)),
                       "MIN": simulate(tr, Belady(cap, next_use(k))),
                       "LRU-gated[gt]": simulate(tr, LRUGatedARC(cap, k=8, gate=gt)),
                       "LRU-gated[det]": simulate(tr, LRUGatedARC(cap, k=8, gate=det))}
                base = split_hit_rates(r_arc, gt)
                gap_t = split_hit_rates(res["MIN"], gt)["transition_hr"] - base["transition_hr"]
                gap_s = split_hit_rates(res["MIN"], gt)["steady_hr"] - base["steady_hr"]
                row = {"group": group, "vol": v, "dim": dim, "mag": mag,
                       "arc_t_half": ad["t_half"], "arc_peak_frac": ad["peak_frac"],
                       "detected": sc["tp"], "det_delay": sc["delays"][0] if sc["delays"] else np.nan,
                       "gate_frac": float(det.mean()), "false_starts": sc["fp"],
                       "arc_transition_hr": base["transition_hr"], "arc_overall_hr": r_arc.hit_rate,
                       "avoidable_misses": (gap_t - gap_s) * a.win}
                for name, r in res.items():
                    if name == "ARC":
                        continue
                    sp = split_hit_rates(r, gt)
                    row[f"dT_{name}"] = (sp["transition_hr"] - base["transition_hr"]) * 100
                    row[f"dO_{name}"] = (r.hit_rate - r_arc.hit_rate) * 100
                rows.append(row)
                del res, r_arc
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "controlled_shifts_raw.partial.csv"), index=False)
            print(f"  {group} {v} done ({(time.time() - T0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "controlled_shifts_raw.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["arc_t_half", "detected", "det_delay", "gate_frac", "avoidable_misses",
            "dT_LRU-gated[gt]", "dO_LRU-gated[gt]", "dT_LRU-gated[det]", "dO_LRU-gated[det]", "dT_MIN"]
    for g in ("test", "train"):
        t = df[df.group == g].groupby(["dim", "mag"])[cols].mean().round(2)
        print(f"\n=== {g.upper()} volumes: mean over 4 volumes per scenario "
              "(dT = change in first 50k accesses after the shift, dO = whole trace; points vs ARC) ===")
        print(t.to_string())
    print(f"\ntotal {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
