"""Control experiment: is the learned re-ranker needed, or is a non-learned rule enough?

LRUGatedARC = ControlledARC with NO model. While the gate is on it evicts the
least-recently-used block among ARC's candidate set (the k tail items of T1 and
T2); otherwise it is plain ARC. Gates are the same as in experiment.py, and the
detector is tuned on the same TRAIN servers, so numbers are directly comparable
with experiment.py's gbt[det] / gbt[gt] rows (same test servers, same ARC).

  python -u scripts/lru_gate_control.py --out results/lru_gate
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.detector import run_detector  # noqa: E402
from shiftcache.policies import ARC, LRU, ControlledARC  # noqa: E402
from scripts.experiment import bootstrap_ci, tune_detector  # noqa: E402
from scripts.workloads import TEST_SEEDS, TRAIN_SEEDS, random_server  # noqa: E402


class LRUGatedARC(ControlledARC):
    """ARC that, while gated on, evicts the least recently used candidate."""

    def _decide(self, src, lists, t):
        cands = self._arc_order(src, lists)
        if len(cands) <= 1:
            return None
        self.n_decisions += 1
        lname, key, _ = min(cands, key=lambda c: self.last_t.get(c[1], -1))
        return lname, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--win", type=int, default=50_000)
    ap.add_argument("--tol", type=int, default=20_000)
    ap.add_argument("--out", default="results/lru_gate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = [random_server(s) for s in TRAIN_SEEDS]
    cfg = tune_detector(train, a.tol)
    del train

    rows = []
    for s in TEST_SEEDS:
        tr = random_server(s)
        cap = max(1, int(tr.footprint * a.fraction))
        n = len(tr)
        gt = transition_mask(n, tr.transitions, a.win)
        det, _ = run_detector(tr.keys, cfg)
        variants = [("ARC", lambda: ARC(cap)), ("LRU", lambda: LRU(cap)),
                    ("LRU-gated[gt]", lambda: LRUGatedARC(cap, k=a.k, gate=gt)),
                    ("LRU-gated[det]", lambda: LRUGatedARC(cap, k=a.k, gate=det))]
        for name, make in variants:
            r = simulate(tr, make())
            rows.append({"trace": tr.name, "policy": name, "overall_hr": r.hit_rate,
                         **split_hit_rates(r, gt)})
        print(f"  {tr.name} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "lru_gate_raw.csv"), index=False)
    arc = df[df.policy == "ARC"].set_index("trace")
    for c in ("transition_hr", "steady_hr", "overall_hr"):
        df["d_" + c] = df.apply(lambda r: (r[c] - arc.loc[r.trace, c]) * 100, axis=1)
    lines = ["policy, transition / steady / overall vs ARC in points, 95% bootstrap CI over "
             f"{len(TEST_SEEDS)} held-out servers"]
    for p in ["LRU", "LRU-gated[gt]", "LRU-gated[det]"]:
        sub = df[df.policy == p]
        t, st, o = (bootstrap_ci(sub["d_" + c]) for c in ("transition_hr", "steady_hr", "overall_hr"))
        lines.append(f"{p:<15} transition {t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]  "
                     f"steady {st[0]:+.2f} [{st[1]:+.2f},{st[2]:+.2f}]  "
                     f"overall {o[0]:+.2f} [{o[1]:+.2f},{o[2]:+.2f}]")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(a.out, "lru_gate_summary.txt"), "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
