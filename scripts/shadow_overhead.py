"""Per-request overhead of the switcher's background shadows, with one-time cache setup subtracted
(libCacheSim allocates a large hash table per cache, which dominates short runs).
Native libCacheSim, cache 10%, first 3M requests; the 6 experts used by the cost-aware switcher.
For each trace: (time on the trace - time on a 1-request trace) per expert, summed over experts,
divided by the same quantity for ARC alone. Minimum of 3 runs each.

  python scripts/shadow_overhead.py --out results/real
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402
from scripts.cost_aware_switcher import paths  # noqa: E402
from scripts.hybrid_selector import sample_mask  # noqa: E402
from scripts.s3fifo_tuning import make  # noqa: E402

EXPERTS = ["S3FIFO(s=0.1,t=2)", "WTinyLFU", "ARC", "LHD", "S3FIFO(s=0.02,t=2)", "S3FIFO(s=0.1,t=1)"]
TRACES = ["msr:hm_0", "cp:w102", "ali:alibabaBlock_14"]


def run(lcs, fn, cand, cap, rep=3):
    best = 1e9
    for _ in range(rep):
        rp = lcs.ReaderInitParam(ignore_obj_size=True)
        rd = lcs.TraceReader(trace=fn, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE, reader_init_params=rp)
        t = time.perf_counter()
        make(lcs, cand, cap).process_trace(rd)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    import libcachesim as lcs
    tune = pd.read_csv(os.path.join(a.out, "s3fifo_tuning_raw_f0.1.csv")).set_index("trace")
    P, rows = paths(), []
    with tempfile.TemporaryDirectory() as tmp:
        for t in TRACES:
            rec = read_records(P[t], 3_000_000)
            cap = int(tune.loc[t, "capacity"])
            full, one, sub = (os.path.join(tmp, x) for x in ("f.bin", "one.bin", "s.bin"))
            rec.tofile(full)
            rec[:1].tofile(one)
            r = {"trace": t, "arc_s": run(lcs, full, "ARC", cap) - run(lcs, one, "ARC", cap)}
            r["full6_s"] = sum(run(lcs, full, c, cap) - run(lcs, one, c, cap) for c in EXPERTS)
            for rate in (0.01, 0.05):
                rec[sample_mask(rec["id"], rate)].tofile(sub)
                c = max(1, int(round(cap * rate)))
                r[f"mini{int(rate * 100)}_6_s"] = sum(run(lcs, sub, x, c) - run(lcs, one, x, c) for x in EXPERTS)
            rows.append(r)
            print(r, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.out, "shadow_overhead.csv"), index=False)
    txt = (f"per-request work relative to ARC alone (setup subtracted, {len(d)} traces): 6 full shadows "
           f"{(d.full6_s / d.arc_s).mean():.1f}x, 6 shadows at 1% {(d.mini1_6_s / d.arc_s).mean():.2f}x, "
           f"6 shadows at 5% {(d.mini5_6_s / d.arc_s).mean():.2f}x")
    print(txt)
    with open(os.path.join(a.out, "shadow_overhead.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
