"""Summaries (with cluster-bootstrap CIs) from the raw CSVs written by real_experiment.py.

CIs resample whole DESTINATION volumes (spliced pairs) or whole volumes (natural), because pairs
that share a volume are not independent. With few volumes the intervals are wide on purpose.

  python scripts/summarize_real.py --out results/real
"""
import argparse
import os

import numpy as np
import pandas as pd


def cluster_ci(df, col, group, B=10_000, seed=0):
    x = df.groupby(group)[col].mean().values
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(x)


def summarize(out):
    L = []
    sp = os.path.join(out, "real_splice_raw.csv")
    if os.path.exists(sp):
        sdf = pd.read_csv(sp)
        arc = sdf[sdf.policy == "ARC"].set_index("pair")
        for c in ("transition_hr", "steady_hr", "overall_hr"):
            sdf["d_" + c] = sdf.apply(lambda r: (r[c] - arc.loc[r.pair, c]) * 100, axis=1)
        L.append(f"REAL spliced TEST pairs ({sdf.pair.nunique()} pairs, {sdf.dst.nunique()} destination "
                 "volumes). Points of hit rate vs ARC; [95% CI resampling destination volumes]")
        L.append(f"  gate on (detector) for {sdf.gate_frac.mean():.0%} of accesses on average")
        for p in [x for x in sdf.policy.unique() if x != "ARC"]:
            sub = sdf[sdf.policy == p]
            parts = []
            for c in ("transition_hr", "steady_hr", "overall_hr"):
                m, lo, hi, _ = cluster_ci(sub, "d_" + c, "dst")
                parts.append(f"{c[:-3]} {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
            L.append(f"  {p:<16} " + "   ".join(parts))
        L.append("  pairs where the detector-gated model beats ARC overall: "
                 f"{int((sdf[sdf.policy == 'gbt[det]'].d_overall_hr > 0).sum())}/{sdf.pair.nunique()}")
    npth = os.path.join(out, "real_natural_raw.csv")
    if not os.path.exists(npth):
        npth = os.path.join(out, "real_natural_raw.partial.csv")
    if os.path.exists(npth):
        ndf = pd.read_csv(npth)
        narc = ndf[ndf.policy == "ARC"].set_index("vol")
        ndf["d_overall"] = ndf.apply(lambda r: (r.overall_hr - narc.loc[r.vol, "overall_hr"]) * 100, axis=1)
        L.append(f"\nREAL natural TEST volumes ({ndf.vol.nunique()} volumes, no injected shift), overall hit "
                 "rate, points vs ARC; [95% CI over volumes]")
        for p in [x for x in ndf.policy.unique() if x != "ARC"]:
            sub = ndf[ndf.policy == p]
            m, lo, hi, k = cluster_ci(sub, "d_overall", "vol")
            L.append(f"  {p:<16} overall {m:+.3f} [{lo:+.3f},{hi:+.3f}]   gate on {sub.gate_frac.mean():.0%} of accesses")
        L.append("  per-volume ARC hit rate: " + ", ".join(f"{v} {narc.loc[v, 'overall_hr']:.3f}" for v in narc.index))
    txt = "\n".join(L)
    with open(os.path.join(out, "real_experiment_summary.txt"), "w") as f:
        f.write(txt + "\n")
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    print(summarize(ap.parse_args().out))
