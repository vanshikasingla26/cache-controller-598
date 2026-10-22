"""Baselines on real oracleGeneral traces (.bin or .zst) using libCacheSim's native reader (fast).

Cache size = fraction of the unique objects in the first N requests; objects have unit size
(ignore_obj_size). Policies: LRU, LFU, ARC, LeCaR, CACHEUS, S3-FIFO, SIEVE, W-TinyLFU, LHD, and
clairvoyant MIN (Belady). Also checks our own ARC against libCacheSim's ARC on the same slice.

  python -u scripts/real_baselines.py --dir /path/to/traces --n 3000000 --out results/real
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate  # noqa: E402
from shiftcache.policies import ARC  # noqa: E402
from shiftcache.realdata import load_oracle_general  # noqa: E402

POLICIES = ["LRU", "LFU", "ARC", "LeCaR", "Cacheus", "S3FIFO", "Sieve", "WTinyLFU", "LHD", "Belady"]


def native_hit(lcs, path, name, cap, n):
    rp = lcs.ReaderInitParam(ignore_obj_size=True)
    rp.cap_at_n_req = n
    rd = lcs.TraceReader(trace=path, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE,
                         reader_init_params=rp)
    cache = getattr(lcs, name)(cache_size=cap)
    miss, _ = cache.process_trace(rd)
    return 1 - miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.out, exist_ok=True)
    rows, parity = [], []
    for path in sorted(glob.glob(os.path.join(a.dir, "msr_*_oracleGeneral.zst"))):
        name = os.path.basename(path)[4:].replace("_oracleGeneral.zst", "")
        tr = load_oracle_general(path, name=name, max_requests=a.n)
        n, uniq = len(tr), tr.footprint
        one_hit = float(np.mean(np.unique(tr.keys, return_counts=True)[1] == 1))
        t0 = time.time()
        for f in a.fractions:
            cap = max(1, int(uniq * f))
            for pol in POLICIES:
                try:
                    hr = native_hit(lcs, path, pol, cap, n)
                except Exception as e:
                    hr = np.nan
                    print(f"  ! {name} {pol}: {str(e)[:100]}", flush=True)
                rows.append({"trace": name, "n": n, "unique": uniq, "one_hit_share": one_hit,
                             "fraction": f, "capacity": cap, "policy": "MIN" if pol == "Belady" else pol,
                             "hit_rate": hr})
        cap5 = max(1, int(uniq * 0.05))
        ours = simulate(tr, ARC(cap5)).hit_rate
        theirs = native_hit(lcs, path, "ARC", cap5, n)
        parity.append({"trace": name, "our_ARC": ours, "libcachesim_ARC": theirs, "abs_diff": abs(ours - theirs)})
        print(f"{name}: {n:,} req, {uniq:,} unique, one-hit {one_hit:.0%}  ({time.time() - t0:.0f}s)  "
              f"ARC parity diff {abs(ours - theirs):.2e}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "baselines_msr.csv"), index=False)
    pd.DataFrame(parity).to_csv(os.path.join(a.out, "arc_parity.csv"), index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for f in a.fractions:
        pv = df[df.fraction == f].pivot(index="trace", columns="policy", values="hit_rate").round(3)
        print(f"\n=== hit rate, cache = {f:.0%} of unique objects ===\n{pv.to_string()}")


if __name__ == "__main__":
    main()
