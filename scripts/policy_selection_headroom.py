"""Is there room for choosing the cache POLICY per trace (instead of reacting to shifts)?

For every trace: first --n requests split into two halves. Each half is replayed by libCacheSim
(native, unit-size objects) with 9 online policies; cache = --fraction of the unique objects in the
whole slice (same cache size for both halves). Then, on the SECOND half only:
  ARC                     the baseline we have used so far
  best fixed policy       the single policy with the best mean hit rate across all TRAIN traces
                          (chosen without looking at test traces)
  pick-from-first-half    causal selector: the policy that was best on this trace's FIRST half
  oracle per trace        the best policy on the second half itself (ceiling, not achievable)
A learned selector is only interesting if it beats both ARC and the best fixed policy.

Exclusion rule (same as before, fixed in advance): traces whose slice is > 90% one-hit objects.
Train/test: datasets are split by trace (see TRAIN_RULE); results reported per dataset and overall.

  python -u scripts/policy_selection_headroom.py --out results/real
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402

POL = ["LRU", "LFU", "ARC", "LeCaR", "Cacheus", "S3FIFO", "Sieve", "WTinyLFU", "LHD"]
DATASETS = {"msr": "/mnt/user-data/uploads/msr_*_oracleGeneral.zst",
            "cp": "/home/claude/data/cp/*.zst", "ali": "/home/claude/data/ali/*.zst",
            "meta": "/home/claude/data/meta/*.zst"}


def name_of(path, tag):
    b = os.path.basename(path)
    for suf in (".oracleGeneral.bin.zst", ".oracleGeneral.zst", "_oracleGeneral.zst"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return f"{tag}:{b[4:] if b.startswith('msr_') else b}"


def native(lcs, path, pol, cap):
    rp = lcs.ReaderInitParam(ignore_obj_size=True)
    rd = lcs.TraceReader(trace=path, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE, reader_init_params=rp)
    miss, _ = getattr(lcs, pol)(cache_size=cap).process_trace(rd)
    return 1 - miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--tmp", default="/tmp/halves")
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.tmp, exist_ok=True)
    T0 = time.time()
    rows, done = [], set()
    raw = os.path.join(a.out, "policy_halves_raw.csv")
    if os.path.exists(raw):
        prev = pd.read_csv(raw)
        rows, done = prev.to_dict("records"), set(prev.trace)
        print(f"resuming: {len(done)} traces already done", flush=True)
    for tag, pat in DATASETS.items():
        for path in sorted(glob.glob(pat)):
            name = name_of(path, tag)
            if name in done:
                continue
            rec = read_records(path, a.n)
            ids = rec["id"]
            _, cnt = np.unique(ids, return_counts=True)
            one_hit = float(np.mean(cnt == 1))
            cap = max(1, int(len(cnt) * a.fraction))
            h = len(rec) // 2
            files = []
            for part, sl in (("h1", rec[:h]), ("h2", rec[h:])):
                f = os.path.join(a.tmp, f"{name.replace(':', '_')}_{part}.bin")
                sl.tofile(f)
                files.append(f)
            r = {"dataset": tag, "trace": name, "n": len(rec), "one_hit": one_hit, "capacity": cap}
            for pol in POL:
                r[f"h1_{pol}"] = native(lcs, files[0], pol, cap)
                r[f"h2_{pol}"] = native(lcs, files[1], pol, cap)
            for f in files:
                os.remove(f)
            rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "policy_halves_raw.csv"), index=False)
            print(f"  {name}: one-hit {one_hit:.0%} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "policy_halves_raw.csv"), index=False)
    print(analyse(df))


def split(df):
    """Train/test split fixed in advance: within each dataset, sort traces by name and assign groups
    alternately; related neighbours (cp w100-w106, ali 165/166) form one group."""
    def group(name):
        if name.startswith("cp:w10"):
            return "cp:w10x"
        if name in ("ali:alibabaBlock_165", "ali:alibabaBlock_166"):
            return "ali:16x"
        return name
    df = df.copy()
    df["group"] = df.trace.map(group)
    df["split"] = "test"
    for ds, sub in df.groupby("dataset"):
        groups = sorted(sub.group.unique())
        for i, g in enumerate(groups):
            if i % 2 == 0:
                df.loc[df.group == g, "split"] = "train"
    return df


def boot(x, B=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(df, max_onehit=0.90):
    df = split(df[df.one_hit <= max_onehit])
    tr, te = df[df.split == "train"], df[df.split == "test"]
    best_fixed = max(POL, key=lambda p: tr[f"h2_{p}"].mean())   # chosen on TRAIN only
    L = [f"traces used: {len(df)} (train {len(tr)}, test {len(te)}); excluded as >90% one-hit scans: see raw CSV",
         f"best fixed policy chosen on TRAIN: {best_fixed}"]

    def gains(sub):
        arc = sub["h2_ARC"]
        fixed = sub[f"h2_{best_fixed}"]
        pick = sub.apply(lambda r: r[f"h2_{max(POL, key=lambda p: r[f'h1_{p}'])}"], axis=1)
        orc = sub[[f"h2_{p}" for p in POL]].max(axis=1)
        return {"best fixed vs ARC": (fixed - arc) * 100, "pick-from-first-half vs ARC": (pick - arc) * 100,
                "pick-from-first-half vs best fixed": (pick - fixed) * 100,
                "oracle per trace vs ARC": (orc - arc) * 100, "oracle per trace vs best fixed": (orc - fixed) * 100}

    for label, sub in (("TEST traces", te), ("TRAIN traces", tr)):
        L.append(f"\n{label}: points of hit rate on the second half, mean [95% CI over traces]")
        for k, v in gains(sub).items():
            m, lo, hi = boot(v)
            L.append(f"  {k:<36} {m:+.2f} [{lo:+.2f},{hi:+.2f}]   positive on {int((v > 0).sum())}/{len(v)}")
    L.append("\nper dataset (TEST + TRAIN), oracle per trace vs ARC and vs best fixed:")
    for ds, sub in df.groupby("dataset"):
        g = gains(sub)
        L.append(f"  {ds:<5} n={len(sub):>2}  vs ARC {g['oracle per trace vs ARC'].mean():+.2f}   vs best fixed "
                 f"{g['oracle per trace vs best fixed'].mean():+.2f}   pick-from-first-half vs best fixed "
                 f"{g['pick-from-first-half vs best fixed'].mean():+.2f}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
