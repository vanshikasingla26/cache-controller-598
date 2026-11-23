"""Two analyses from existing results (classmate feedback on the Nov 5 presentation):

1. Scan-exclusion sensitivity. Our fixed rule drops trace slices with > 90% one-hit objects (4 of 42 MSR
   slices). Re-run the non-learned leave-one-source-out selectors (always S3-FIFO, best fixed policy on
   the training sources, pick from first half, oracle) with all 42 traces kept, at 1/5/10% caches, and
   report the 4 scan traces on their own.
2. When does adaptation pay off? Per trace, relate cheap workload statistics (one-hit share, unique
   ratio, top-10% access share, sequential share, accesses per object; means over the trace's six
   500k-request windows) to (a) the room adaptation has over a fixed policy (oracle minus always
   S3-FIFO, 5% cache, LOSO), (b) the gain the simulation selector actually realizes (pick minus always
   S3-FIFO, 5%), and (c) the per-window room at 10% (best candidate per window minus default S3-FIFO).
   Spearman correlations over traces, plus the 5% results split into thirds of the strongest statistic.

  python scripts/workload_analysis.py --out results/real
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.policy_selection_headroom import POL, boot  # noqa: E402

SIZES = {0.01: "_f0.01", 0.05: "", 0.10: "_f0.1"}


def loso_rows(h):
    h = h.copy()
    h["source"] = h.trace.str.split(":").str[0]
    out = []
    for src in sorted(h.source.unique()):
        trn, tst = h[h.source != src], h[h.source == src]
        best = max(POL, key=lambda p: trn[f"h2_{p}"].mean())
        for _, r in tst.iterrows():
            arc = 1 - r["h2_ARC"]
            pick = max(POL, key=lambda p: r[f"h1_{p}"])
            orc = max(POL, key=lambda p: r[f"h2_{p}"])
            red = lambda p: (arc - (1 - r[f"h2_{p}"])) / arc * 100  # noqa: E731
            out.append({"trace": r.trace, "source": src, "one_hit": r.one_hit, "scan": r.one_hit > 0.90,
                        "always_S3FIFO": red("S3FIFO"), "best_fixed": red(best), "pick": red(pick),
                        "oracle": red(orc), "best_fixed_policy": best})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    L = ["1. SCAN-EXCLUSION SENSITIVITY (leave-one-source-out, relative miss-ratio reduction vs ARC, %, "
         "mean [95% CI over traces])"]
    sens = []
    for f, suf in SIZES.items():
        h = pd.read_csv(os.path.join(a.out, f"policy_halves_raw{suf}.csv"))
        kept = loso_rows(h[h.one_hit <= 0.90])
        allr = loso_rows(h)
        L.append(f"  cache {f:.0%}: {len(kept)} traces with the rule, {len(allr)} without")
        for c in ("always_S3FIFO", "best_fixed", "pick", "oracle"):
            m1, l1, h1 = boot(kept[c].values)
            m2, l2, h2 = boot(allr[c].values)
            L.append(f"    {c:<14} rule {m1:+.2f} [{l1:+.2f}, {h1:+.2f}]   all 42 {m2:+.2f} [{l2:+.2f}, {h2:+.2f}]")
        k = allr.set_index("trace")
        d1 = boot((kept.set_index("trace").pick - kept.set_index("trace").always_S3FIFO).values)
        d2 = boot((k.pick - k.always_S3FIFO).values)
        L.append(f"    pick minus always_S3FIFO: rule {d1[0]:+.2f} [{d1[1]:+.2f}, {d1[2]:+.2f}]   "
                 f"all 42 {d2[0]:+.2f} [{d2[1]:+.2f}, {d2[2]:+.2f}]")
        sc = allr[allr.scan]
        L.append(f"    the {len(sc)} scan traces alone: always S3-FIFO {sc.always_S3FIFO.mean():+.2f}, pick "
                 f"{sc.pick.mean():+.2f}, oracle {sc.oracle.mean():+.2f} (ARC hit rate is near zero there)")
        L.append(f"    best fixed policy per held-out source, all 42: "
                 f"{dict(allr.groupby('source').best_fixed_policy.first())}")
        allr["size"] = f
        sens.append(allr)
    pd.concat(sens).to_csv(os.path.join(a.out, "scan_sensitivity.csv"), index=False)

    feats = pd.read_csv(os.path.join(a.out, "selector_features.csv"))
    stats = ["one_hit", "unique_ratio", "top10_share", "seq_share", "acc_per_obj"]
    ws = feats.groupby("trace")[stats].mean()
    lr = pd.read_csv(os.path.join(a.out, "loso_results.csv"))
    l5 = lr[lr["size"] == 0.05].pivot_table(index="trace", columns="selector", values="rel_mr_reduction")
    tgt = pd.DataFrame({"room_5pct": l5.oracle - l5.always_S3FIFO,
                        "realized_5pct": l5.pick_first_half - l5.always_S3FIFO})
    tu = pd.read_csv(os.path.join(a.out, "s3fifo_tuning_summary_f0.1.csv")).set_index("trace")
    tgt["room_10pct_windows"] = tu.ideal_all - tu.S3FIFO_default
    d = ws.join(tgt, how="inner")
    d.to_csv(os.path.join(a.out, "workload_characteristics.csv"))
    L.append(f"\n2. WHEN DOES ADAPTATION PAY OFF? Spearman correlation over {len(d)} traces "
             "(window-mean statistic vs gain over always S3-FIFO)")
    corr = pd.DataFrame({t: [d[s].corr(d[t], method="spearman") for s in stats] for t in tgt.columns}, index=stats)
    L.append(corr.round(2).to_string())
    key = corr["room_5pct"].abs().idxmax()
    d["third"] = pd.qcut(d[key], 3, labels=["low", "mid", "high"])
    L.append(f"\n  5% results by thirds of {key} (mean %, n per third):")
    for th, g in d.groupby("third", observed=True):
        L.append(f"    {th:<4} {key} {g[key].min():.2f}-{g[key].max():.2f}: room {g.room_5pct.mean():+.2f}, "
                 f"realized {g.realized_5pct.mean():+.2f}, room at 10% per window {g.room_10pct_windows.mean():+.2f} "
                 f"(n={len(g)})")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "workload_analysis.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
