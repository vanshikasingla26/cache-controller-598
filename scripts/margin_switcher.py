"""Reactive switcher with a margin (the fallback rule from Progress Report 2), at one cache size.
Protocol fixed before results; leave-one-source-out (LOSO); same traces, windows, carry mechanics and candidates
as s3fifo_tuning.py / switcher_v2.py.

Rule: start with default S3-FIFO. At each window boundary take the candidate (8 S3-FIFO settings + 9 policies)
with the best hit rate on the window that just ended (shadow simulation, as the naive rule assumes) and move to
it only if it beat the current candidate on that window by more than a margin m (hit-rate points).
m = 0 is the naive "settings and policies" rule; a large m never switches (default S3-FIFO).

The margin is chosen per held-out source from MARGINS using only the other three sources, by an estimated score:
continuous hit rate of the planned candidate in each window minus the measured switching cost (carry, clamped at
zero, training traces only) for each switch. The chosen plan is then executed exactly on the held-out traces.

  python -u scripts/margin_switcher.py --fraction 0.01 --out results/real
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402
from scripts.policy_selection_headroom import boot  # noqa: E402
from scripts.s3fifo_tuning import NWIN, W, run_plan  # noqa: E402
from scripts.cost_aware_switcher import CANDS, DEFAULT, paths  # noqa: E402

MARGINS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)   # hit-rate points


def plan_for(cont_t, m):
    plan = [DEFAULT]
    for w in range(1, NWIN):
        best = max(CANDS, key=lambda c: cont_t[c][w - 1])
        lead = (cont_t[best][w - 1] - cont_t[plan[-1]][w - 1]) * 100
        plan.append(best if lead > m else plan[-1])
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    import libcachesim as lcs
    fr = f"{a.fraction:g}"
    T0 = time.time()
    tune = pd.read_csv(os.path.join(a.out, f"s3fifo_tuning_raw_f{fr}.csv")).set_index("trace")
    sw = pd.read_csv(os.path.join(a.out, f"switching_raw_f{fr}.csv")).set_index("trace")
    traces = list(tune.index)
    cont = {t: {c: np.array([tune.loc[t, f"cont_{c}_w{w + 1}"] for w in range(NWIN)]) for c in CANDS} for t in traces}
    src = {t: t.split(":")[0] for t in traces}
    P = paths()

    def switch_cost(ts):
        cs = []
        for t in ts:
            plan = sw.loc[t, "plan"].split(">")
            cs += [sw.loc[t, f"cont_{plan[w]}_w{w + 1}"] - sw.loc[t, f"carry_w{w + 1}"]
                   for w in range(1, NWIN) if plan[w] != plan[w - 1]]
        return max(0.0, float(np.mean(cs)))

    def est_score(t, plan, cost):
        return np.mean([cont[t][plan[w]][w] - cost * (plan[w] != plan[w - 1]) for w in range(1, NWIN)])

    rawp = os.path.join(a.out, f"margin_switcher_raw_f{fr}.csv")
    rows = pd.read_csv(rawp).to_dict("records") if os.path.exists(rawp) else []
    done = {r["trace"] for r in rows}
    info = []
    for held in sorted(set(src.values())):
        train = [t for t in traces if src[t] != held]
        test = [t for t in traces if src[t] == held]
        cost = switch_cost(train)
        sc = {m: np.mean([est_score(t, plan_for(cont[t], m), cost) for t in train]) for m in MARGINS}
        m_star = max(MARGINS, key=lambda m: (sc[m], m))   # ties go to the larger margin (fewer switches)
        info.append(f"held-out {held}: cost {cost * 100:.2f} pts, margin {m_star:g} pts; estimated train score "
                    + ", ".join(f"m={m:g}: {sc[m] * 100:.2f}" for m in MARGINS))
        for t in test:
            if t in done:
                continue
            plan = plan_for(cont[t], m_star)
            if all(p == DEFAULT for p in plan):
                hr = cont[t][DEFAULT]
            else:
                rec = read_records(P[t], W * NWIN)
                hr = run_plan(lcs, rec["id"].tolist(), int(tune.loc[t, "capacity"]), plan)
            r = {"trace": t, "source": held, "margin": m_star, "plan": ">".join(plan),
                 "switches": sum(plan[w] != plan[w - 1] for w in range(1, NWIN))}
            r.update({f"margin_w{w + 1}": hr[w] for w in range(NWIN)})
            rows.append(r)
            pd.DataFrame(rows).to_csv(rawp, index=False)
            print(f"  {t}: margin {m_star:g}, {r['switches']} switches ({(time.time() - T0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows).set_index("trace")
    ws = range(2, NWIN + 1)
    out = []
    for t in df.index:
        m_arc = np.mean([1 - cont[t]["ARC"][w - 1] for w in ws])
        red = lambda miss: (m_arc - miss) / m_arc * 100  # noqa: E731
        out.append({"trace": t, "source": src[t],
                    "S3FIFO_default": red(np.mean([1 - cont[t][DEFAULT][w - 1] for w in ws])),
                    "hybrid_m0": red(np.mean([1 - tune.loc[t, f"hybrid_w{w}"] for w in ws])),
                    "margin": red(np.mean([1 - df.loc[t, f"margin_w{w}"] for w in ws])),
                    "switches": df.loc[t, "switches"]})
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, f"margin_switcher_summary_f{fr}.csv"), index=False)
    L = [f"MARGIN SWITCHER, leave-one-source-out, cache {a.fraction:.0%}, windows 2-6; relative miss-ratio reduction "
         f"vs ARC (%), mean [95% CI over {len(s)} traces]"]
    for c in ("S3FIFO_default", "hybrid_m0", "margin"):
        m, lo, hi = boot(s[c].values)
        L.append(f"  {c:<15} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
    for c in ("hybrid_m0", "margin"):
        d = s[c] - s.S3FIFO_default
        m, lo, hi = boot(d.values)
        L.append(f"  {c} minus S3FIFO_default: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   better on {int((d > 0).sum())}/{len(s)}, "
                 f"worse on {int((d < 0).sum())}/{len(s)}")
    L.append(f"  switches per trace (margin rule): {s.switches.mean():.2f}")
    L += info
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, f"margin_switcher_summary_f{fr}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
