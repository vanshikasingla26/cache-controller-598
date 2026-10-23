"""Hybrid policy selector: miniature (spatially sampled) simulations + a learned correction.
Protocol fixed before results.

Selectors compared on the SAME second halves as policy_selection_headroom.py (two folds, every trace tested
once, best fixed policy chosen on each fold's TRAIN):
  mini-sim R%     simulate the 9 policies on a hash-sampled R% of objects of the FIRST half, with the cache
                  scaled by R (SHARDS-style fixed-rate sampling; Waldspurger et al. ATC'17 miniature
                  simulations), pick the best. No learning.
  hybrid (ML)     per policy, a regressor predicts the gain over ARC on the next 500k-request window from
                  [mini-sim gains over ARC on the current window (R = 5%)] + [cheap workload features]; trained
                  on TRAIN traces only (window j -> window j+1 rows), model (ridge / shallow GBR) chosen by
                  leave-one-trace-out CV; decision from window 3 (last 500k requests of the first half).
  features-only   the earlier learned selector (selector_test_results.csv), for reference
  full-sim pick   simulate the 9 policies on the whole first half (policy_selection_headroom.py)
Costs: wall time of sampling + 9 miniature simulations vs the full 9 simulations.

  python -u scripts/hybrid_selector.py --out results/real
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
from scripts.learned_selector import fit_predict, make_models  # noqa: E402
from scripts.policy_selection_headroom import DATASETS, POL, boot, name_of, native, split  # noqa: E402

W, NWIN = 500_000, 6
GOLD = np.uint64(0x9E3779B97F4A7C15)


def sample_mask(ids, rate):
    h = (ids.astype(np.uint64) * GOLD) >> np.uint64(40)
    return (h % np.uint64(10_000)) < np.uint64(int(rate * 10_000))


def mini(lcs, rec, cap, rate, tmp):
    """9 policies on a spatial sample of `rec` with a cache scaled by `rate`. Returns ({policy: hr}, seconds)."""
    t = time.perf_counter()
    sub = rec[sample_mask(rec["id"], rate)]
    fn = os.path.join(tmp, "mini.bin")
    sub.tofile(fn)
    c = max(1, int(round(cap * rate)))
    out = {p: native(lcs, fn, p, c) for p in POL}
    os.remove(fn)
    return out, time.perf_counter() - t


def collect(a, lcs, halves):
    raw = os.path.join(a.out, "hybrid_mini_raw.csv")
    rows = pd.read_csv(raw).to_dict("records") if os.path.exists(raw) else []
    done = {(r["trace"], r["what"]) for r in rows}
    T0 = time.time()
    for tag, pat in DATASETS.items():
        for path in sorted(glob.glob(pat)):
            name = name_of(path, tag)
            if name not in halves.index or (name, "w6") in done:
                continue
            cap = int(halves.loc[name, "capacity"])
            rec = read_records(path, W * NWIN)
            h = len(rec) // 2
            for rate in (0.01, 0.05):
                hr, sec = mini(lcs, rec[:h], cap, rate, a.tmp)
                rows.append({"trace": name, "what": f"h1@{rate}", "seconds": sec, **hr})
            for j in range(NWIN):
                hr, sec = mini(lcs, rec[j * W:(j + 1) * W], cap, 0.05, a.tmp)
                rows.append({"trace": name, "what": f"w{j + 1}", "seconds": sec, **hr})
            pd.DataFrame(rows).to_csv(raw, index=False)
            print(f"  {name} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/hybrid")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.tmp, exist_ok=True)
    halves_raw = pd.read_csv(os.path.join(a.out, "policy_halves_raw.csv"))
    used = halves_raw[halves_raw.one_hit <= 0.90].set_index("trace")
    mini_df = collect(a, lcs, used)
    feats = pd.read_csv(os.path.join(a.out, "selector_features.csv"))
    labs = pd.read_csv(os.path.join(a.out, "selector_window_labels.csv"))
    cost = pd.read_csv(os.path.join(a.out, "selector_cost.csv")).set_index("trace")
    fcols = [c for c in feats.columns if c not in ("trace", "split", "window", "feat_s")]

    win = mini_df[mini_df.what.str.startswith("w")].copy()
    win["window"] = win.what.str[1:].astype(int)
    mcols = [f"mini_{p}" for p in POL if p != "ARC"]
    for p in POL:
        if p != "ARC":
            win[f"mini_{p}"] = (win[p] - win["ARC"]) * 100
    X_all = feats[["trace", "window"] + fcols].merge(win[["trace", "window"] + mcols], on=["trace", "window"])
    cols = mcols + fcols

    rows, infos = [], []
    for fold in (0, 1):
        hv = split(halves_raw[halves_raw.one_hit <= 0.90], fold).set_index("trace")
        sp = hv["split"]
        d = X_all.assign(split=X_all.trace.map(sp)).dropna(subset=["split"])
        tr = d[d.split == "train"].merge(labs.assign(window=labs.window - 1), on=["trace", "window"])
        X, groups = tr[cols].values, tr.trace.values
        Y = np.stack([(tr[p] - tr["ARC"]).values * 100 for p in POL], axis=1)
        cv = {}
        for mname, make in make_models().items():
            g = []
            for t in np.unique(groups):
                m = groups == t
                P = fit_predict(make, X[~m], Y[~m], X[m])
                g += [Y[i, k] for i, k in zip(np.flatnonzero(m), P.argmax(axis=1))]
            cv[mname] = float(np.mean(g))
        best = max(cv, key=cv.get)
        te = d[(d.split == "test") & (d.window == 3)].set_index("trace")
        P = fit_predict(make_models()[best], X, Y, te[cols].values)
        tr_best = max(POL, key=lambda p: hv[hv.split == "train"][f"h2_{p}"].mean())
        for (trace, _), pred in zip(te.iterrows(), P):
            h = hv.loc[trace]
            arc = h["h2_ARC"]
            r = {"fold": fold, "trace": trace, "dataset": trace.split(":")[0],
                 "hybrid": (h[f"h2_{POL[int(np.argmax(pred))]}"] - arc) * 100,
                 "always_best_fixed": (h[f"h2_{tr_best}"] - arc) * 100,
                 "full_sim_pick": (h[f"h2_{max(POL, key=lambda p: h[f'h1_{p}'])}"] - arc) * 100,
                 "oracle": (max(h[f"h2_{p}"] for p in POL) - arc) * 100}
            for rate in (0.01, 0.05):
                mrow = mini_df[(mini_df.trace == trace) & (mini_df.what == f"h1@{rate}")].iloc[0]
                r[f"mini_sim_{int(rate * 100)}pct"] = (h[f"h2_{max(POL, key=lambda p: mrow[p])}"] - arc) * 100
                r[f"mini_sim_{int(rate * 100)}pct_s"] = mrow["seconds"]
            r["full_sim_s"] = cost.loc[trace, "sim9_first_half_s"] if trace in cost.index else np.nan
            rows.append(r)
        infos.append((fold, len(tr), tr.trace.nunique(), cv, best, tr_best))
    res = pd.DataFrame(rows)
    sel = pd.read_csv(os.path.join(a.out, "selector_test_results.csv"))[["trace", "learned"]]
    res = res.merge(sel.rename(columns={"learned": "features_only"}), on="trace", how="left")
    res.to_csv(os.path.join(a.out, "hybrid_test_results.csv"), index=False)

    L = [f"fold {f}: {n} training rows from {k} traces; LOTO-CV on TRAIN: " +
         ", ".join(f"{m} {v:+.2f}" for m, v in cv.items()) + f"; chosen {b}; best fixed on TRAIN {bf}"
         for f, n, k, cv, b, bf in infos]
    L.append(f"\nTWO FOLDS, {len(res)} test traces, second-half hit rate vs ARC (points), mean [95% CI], positive")
    for c in ("always_best_fixed", "features_only", "mini_sim_1pct", "mini_sim_5pct", "hybrid", "full_sim_pick",
              "oracle"):
        v = res[c].dropna().values
        m, lo, hi = boot(v)
        L.append(f"  {c:<18} {m:+.2f} [{lo:+.2f},{hi:+.2f}]   {int((v > 0).sum())}/{len(v)}")
    for x, y in (("hybrid", "always_best_fixed"), ("hybrid", "mini_sim_5pct"), ("mini_sim_5pct", "always_best_fixed")):
        m, lo, hi = boot((res[x] - res[y]).values)
        L.append(f"  {x} minus {y}: {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    L.append(f"\ncost per decision: mini-sim 1% {res.mini_sim_1pct_s.mean():.2f} s, 5% {res.mini_sim_5pct_s.mean():.2f} s "
             f"(sampling + 9 simulations) vs full simulation {res.full_sim_s.mean():.2f} s")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "hybrid_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
