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
# byte mode: libCacheSim's W-TinyLFU sizes its count-min sketch by the cache capacity in BYTES and fails to
# allocate (about 1.25e9 counters for a 88 MB cache), so it is left out of byte-level runs
POL_BYTES = [p for p in POL if p != "WTinyLFU"]


def pols_in(df):
    return [p for p in POL if f"h2_{p}" in df.columns]
# expected layout (see README "Reproducing"): data/msr, data/cp, data/ali, data/meta
DATASETS = {"msr": "data/msr/msr_*_oracleGeneral.zst", "cp": "data/cp/*.zst",
            "ali": "data/ali/*.zst", "meta": "data/meta/*.zst"}


def name_of(path, tag):
    b = os.path.basename(path)
    for suf in (".oracleGeneral.bin.zst", ".oracleGeneral.zst", "_oracleGeneral.zst"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return f"{tag}:{b[4:] if b.startswith('msr_') else b}"


def native(lcs, path, pol, cap, sizes=False):
    """Hit rate from libCacheSim. sizes=False: unit-size objects, object hit rate (cap in objects).
    sizes=True: real request sizes, BYTE hit rate (cap in bytes)."""
    rp = lcs.ReaderInitParam(ignore_obj_size=not sizes)
    rd = lcs.TraceReader(trace=path, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE, reader_init_params=rp)
    miss, byte_miss = getattr(lcs, pol)(cache_size=cap).process_trace(rd)
    return 1 - (byte_miss if sizes else miss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--tmp", default="/tmp/halves")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--sizes", action="store_true",
                    help="real object sizes: cache = fraction of unique BYTES, metric = byte hit rate")
    a = ap.parse_args()
    import libcachesim as lcs
    rawname = "policy_halves_bytes_raw.csv" if a.sizes else "policy_halves_raw.csv"
    os.makedirs(a.tmp, exist_ok=True)
    T0 = time.time()
    rows, done = [], set()
    raw = os.path.join(a.out, rawname)
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
            if a.sizes:   # bytes of the unique objects (size at first occurrence)
                _, first_idx = np.unique(ids, return_index=True)
                cap = max(1, int(rec["size"][first_idx].astype(np.int64).sum() * a.fraction))
            else:
                cap = max(1, int(len(cnt) * a.fraction))
            h = len(rec) // 2
            files = []
            for part, sl in (("h1", rec[:h]), ("h2", rec[h:])):
                f = os.path.join(a.tmp, f"{name.replace(':', '_')}_{part}.bin")
                sl.tofile(f)
                files.append(f)
            r = {"dataset": tag, "trace": name, "n": len(rec), "one_hit": one_hit, "capacity": cap}
            for pol in (POL_BYTES if a.sizes else POL):
                r[f"h1_{pol}"] = native(lcs, files[0], pol, cap, a.sizes)
                r[f"h2_{pol}"] = native(lcs, files[1], pol, cap, a.sizes)
            for f in files:
                os.remove(f)
            rows.append(r)
            pd.DataFrame(rows).to_csv(raw, index=False)
            print(f"  {name}: one-hit {one_hit:.0%} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(raw, index=False)
    print(analyse(df))
    txt, r = analyse_2fold(df)
    r.to_csv(os.path.join(a.out, "policy_selection_bytes_2fold.csv" if a.sizes else "policy_selection_2fold.csv"),
             index=False)
    with open(os.path.join(a.out, "policy_selection_bytes_summary.txt" if a.sizes else "policy_selection_2fold_summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


def split(df, fold=0):
    """Train/test split fixed in advance: within each dataset, sort traces by name and assign groups
    alternately; related neighbours (cp w100-w106, ali 165/166) form one group. fold=1 swaps the
    roles, so over the two folds every trace is a test trace exactly once."""
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
            if i % 2 == fold:
                df.loc[df.group == g, "split"] = "train"
    return df


def boot(x, B=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fold_rows(df, fold, max_onehit=0.90):
    """Per-test-trace gains for one fold (best fixed policy chosen on that fold's TRAIN)."""
    d = split(df[df.one_hit <= max_onehit], fold)
    P = pols_in(df)
    tr, te = d[d.split == "train"], d[d.split == "test"]
    best = max(P, key=lambda p: tr[f"h2_{p}"].mean())
    out = te[["dataset", "trace"]].copy()
    out["fold"], out["best_fixed"] = fold, best
    arc = te["h2_ARC"]
    out["always_best_fixed"] = (te[f"h2_{best}"] - arc).values * 100
    out["pick_first_half"] = te.apply(
        lambda r: r[f"h2_{max(P, key=lambda p: r[f'h1_{p}'])}"], axis=1).values * 100 - arc.values * 100
    out["oracle"] = (te[[f"h2_{p}" for p in P]].max(axis=1) - arc).values * 100
    return out


def analyse_2fold(df):
    r = pd.concat([fold_rows(df, 0), fold_rows(df, 1)], ignore_index=True)
    L = [f"2-FOLD: every trace is a test trace once ({len(r)} traces). Best fixed policy per fold "
         f"(chosen on that fold's TRAIN): {dict(r.groupby('fold').best_fixed.first())}",
         "second-half hit rate vs ARC (points), mean [95% CI over traces], positive count"]
    for c in ("always_best_fixed", "pick_first_half", "oracle"):
        m, lo, hi = boot(r[c])
        L.append(f"  {c:<18} {m:+.2f} [{lo:+.2f},{hi:+.2f}]   {int((r[c] > 0).sum())}/{len(r)}")
    m, lo, hi = boot(r.pick_first_half - r.always_best_fixed)
    L.append(f"  pick_first_half minus always_best_fixed: {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    L.append("  per dataset, pick_first_half vs ARC: " + ", ".join(
        f"{ds} {g.pick_first_half.mean():+.2f} (n={len(g)})" for ds, g in r.groupby("dataset")))
    return "\n".join(L), r


def analyse(df, max_onehit=0.90):
    P = pols_in(df)
    df = split(df[df.one_hit <= max_onehit])
    tr, te = df[df.split == "train"], df[df.split == "test"]
    best_fixed = max(P, key=lambda p: tr[f"h2_{p}"].mean())   # chosen on TRAIN only
    L = [f"traces used: {len(df)} (train {len(tr)}, test {len(te)}); excluded as >90% one-hit scans: see raw CSV",
         f"best fixed policy chosen on TRAIN: {best_fixed}"]

    def gains(sub):
        arc = sub["h2_ARC"]
        fixed = sub[f"h2_{best_fixed}"]
        pick = sub.apply(lambda r: r[f"h2_{max(P, key=lambda p: r[f'h1_{p}'])}"], axis=1)
        orc = sub[[f"h2_{p}" for p in P]].max(axis=1)
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
