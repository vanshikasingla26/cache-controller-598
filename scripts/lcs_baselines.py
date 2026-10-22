"""Hit rates of strong baselines (libCacheSim) plus our ARC and Belady MIN on one trace.

All caches are object-count based (obj_size = 1), cache size = fraction of the trace's unique
objects. Policies: LRU, LFU, ARC, LeCaR, CACHEUS, S3-FIFO, SIEVE, W-TinyLFU, LHD, plus MIN.

  python scripts/lcs_baselines.py --trace data/cloudphysics_sample.oracleGeneral.bin
  python scripts/lcs_baselines.py --msr data/web_0.csv.gz --max-requests 5000000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate  # noqa: E402
from shiftcache.policies import ARC, Belady, next_use  # noqa: E402
from shiftcache.realdata import load_oracle_general  # noqa: E402
from shiftcache.trace import load_msr  # noqa: E402

LCS = ["LRU", "LFU", "ARC", "LeCaR", "Cacheus", "S3FIFO", "Sieve", "WTinyLFU", "LHD"]


def lcs_hit_rate(lcs, name, cap, keys):
    cache = getattr(lcs, name)(cache_size=cap)
    hits = 0
    for k in keys:
        hits += bool(cache.get(lcs.Request(obj_size=1, obj_id=int(k) % (1 << 62))))
    return hits / len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--msr")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    ap.add_argument("--out", default="results/baselines")
    a = ap.parse_args()
    import libcachesim as lcs
    if a.msr:
        tr = load_msr(a.msr, max_requests=a.max_requests)
    else:
        tr = load_oracle_general(a.trace, max_requests=a.max_requests)
    keys = tr.keys
    nu = next_use(keys)
    rows = []
    for f in a.fractions:
        cap = max(1, int(tr.footprint * f))
        for name in LCS:
            try:
                rows.append({"fraction": f, "capacity": cap, "policy": name,
                             "hit_rate": lcs_hit_rate(lcs, name, cap, keys.tolist())})
            except Exception as e:  # a policy may not support unit-size objects
                rows.append({"fraction": f, "capacity": cap, "policy": name, "hit_rate": np.nan,
                             "error": str(e)[:80]})
        rows.append({"fraction": f, "capacity": cap, "policy": "our-ARC",
                     "hit_rate": simulate(tr, ARC(cap)).hit_rate})
        rows.append({"fraction": f, "capacity": cap, "policy": "MIN",
                     "hit_rate": simulate(tr, Belady(cap, nu)).hit_rate})
        print(f"cache {f:.0%} done", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"{tr.name}_baselines.csv"), index=False)
    pv = df.pivot(index="policy", columns="fraction", values="hit_rate").round(4)
    print(f"\n{tr.name}: {len(tr):,} accesses, {tr.footprint:,} unique objects")
    print(pv.sort_values(pv.columns[1], ascending=False).to_string())


if __name__ == "__main__":
    main()
