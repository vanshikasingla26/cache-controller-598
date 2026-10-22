"""Headroom experiments (roadmap: 'measure the headroom first').

  1. Belady gap: (Belady - ARC) hit rate in transition windows vs steady state.
  2. Oracle tail re-ranker: ARC whose victim is the furthest-next-use item among
     the k LRU-end candidates. Run (a) always-on and (b) only inside transition
     windows, the latter bounds a controller that only fires during shifts.

Transition windows here come from GROUND TRUTH (synthetic / spliced traces),
never from our own detector, so the evaluation is not circular.

  python scripts/headroom.py --synthetic --fraction 0.05 --win 50000
  python scripts/headroom.py --splice data/web_0.csv.gz data/src1_2.csv.gz --fraction 0.05
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import load_msr, simulate, splice, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.policies import ARC, LRU, Belady, OracleTailARC, OracleCrossListARC, next_use  # noqa: E402
from scripts.demo_traces import demo_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--splice", nargs=2, metavar=("A", "B"), help="two MSR files to splice")
    ap.add_argument("--splice-len", type=int, default=1_000_000, help="block accesses taken from each")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--win", type=int, default=50_000, help="transition window length (accesses)")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.splice:
        A, B = (load_msr(p) for p in a.splice)
        tr = splice(A, B, min(a.splice_len, len(A)), min(a.splice_len, len(B)))
    else:
        tr = demo_trace()
    assert tr.transitions, "headroom needs ground-truth transitions"

    cap = max(1, int(tr.footprint * a.fraction))
    nu = next_use(tr.keys)
    mask = transition_mask(len(tr), tr.transitions, a.win)
    print(f"{tr.name}: {len(tr):,} accesses, cap={cap:,}, transitions at {tr.transitions}, "
          f"window={a.win:,} ({mask.mean():.1%} of trace)")

    pols = [LRU(cap), ARC(cap), Belady(cap, nu)]
    for k in a.ks:
        pols.append(OracleTailARC(cap, nu, k=k))
        pols.append(OracleTailARC(cap, nu, k=k, active=mask))
        pols.append(OracleCrossListARC(cap, nu, k=k))
        pols.append(OracleCrossListARC(cap, nu, k=k, active=mask))

    rows = []
    arc_split = None
    for p in pols:
        r = simulate(tr, p)
        s = {"policy": r.policy, "overall_hr": r.hit_rate, **split_hit_rates(r, mask)}
        if isinstance(p, OracleTailARC):
            s["overrides"] = p.n_overrides
        if isinstance(p, OracleCrossListARC):
            s["list_flips"] = p.n_list_flips
        if r.policy == "ARC":
            arc_split = s
        rows.append(s)

    df = pd.DataFrame(rows)
    for col in ("transition_hr", "steady_hr"):
        df[f"{col}_vs_ARC"] = df[col] - arc_split[col]
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))

    bel = df[df.policy == "Belady"].iloc[0]
    print("\nBelady gap  transition: {:+.4f}   steady: {:+.4f}".format(
        bel.transition_hr_vs_ARC, bel.steady_hr_vs_ARC))
    if bel.transition_hr_vs_ARC <= bel.steady_hr_vs_ARC:
        print("  !! gap does NOT widen in transitions on this trace: premise is weak here")
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"headroom_{tr.name}_f{a.fraction}.csv"), index=False)


if __name__ == "__main__":
    main()
