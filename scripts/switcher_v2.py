"""Cost-aware switcher without full background shadows, plus longer history (Progress Report 2,
classmate feedback on the Nov 5 presentation). Protocol fixed before results; same 38 traces, 10% cache,
six 500k-request windows, carry mechanics, leave-one-source-out folds, expert sets and switching cost as
scripts/cost_aware_switcher.py.

Shadow estimates
  exact     every candidate runs in full in the background (what cost_aware_switcher.py assumed)
  mini R    every candidate runs on a hash-sampled R of the objects (same sampling as
            hybrid_selector.py) with the cache scaled by R, continuously over the trace; its hit rate on
            the sampled requests of a window is the estimate for that window (SHARDS-style spatial
            sampling, Waldspurger et al. FAST'15 / ATC'17). R in {1%, 5%}.
Switcher inputs at a window boundary: the experts' estimated gains over default S3-FIFO on the window
that just ended (+ optionally the window before it and the change between them), plus the cheap
workload features of that window. Training labels are EXACT next-window gains on the training sources
(training is offline). Decision rule as before: move only if the predicted gain over the current expert
exceeds the measured switching cost.
Also reported: fidelity of the estimates (does the sampled winner match the exact winner) and the
overhead of each estimate source in native libCacheSim time and in extra cache work.

  python -u scripts/switcher_v2.py --out results/real
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402
from scripts.cost_aware_switcher import CANDS, DEFAULT, K, fit_k, paths  # noqa: E402
from scripts.hybrid_selector import sample_mask  # noqa: E402
from scripts.learned_selector import make_models  # noqa: E402
from scripts.policy_selection_headroom import POL, boot  # noqa: E402
from scripts.s3fifo_tuning import NWIN, W, make, run_plan  # noqa: E402

RATES = (0.01, 0.05)
# S3-FIFO's 2% small queue needs a cache of at least 50 objects; below that the sampling rate is raised
# so the miniature cache stays at 50 (only binds at the 1% cache size; no 10% result changes)
MIN_MINI = 50


def shadow_estimates(lcs, ids, cap, rate):
    """per-window hit rate of every candidate on a hash-sampled stream with a scaled cache"""
    arr = np.asarray(ids, dtype=np.uint64)
    rate = max(rate, MIN_MINI / cap)   # SHARDS-style floor: the miniature cache keeps >= MIN_MINI objects
    keep = sample_mask(arr, rate)
    idx = np.flatnonzero(keep)
    sub = arr[keep].tolist()
    win = (idx // W).astype(int)
    c = max(MIN_MINI, int(round(cap * rate)))
    R = lcs.Request
    out, t0 = {}, time.perf_counter()
    for cand in CANDS:
        cache = make(lcs, cand, c)
        hits = np.fromiter((bool(cache.get(R(obj_size=1, obj_id=o))) for o in sub), dtype=bool, count=len(sub))
        out[cand] = np.array([hits[win == w].mean() if (win == w).any() else np.nan for w in range(NWIN)])
    return out, time.perf_counter() - t0, len(sub)


def native_seconds(lcs, fn, cand, cap):
    rp = lcs.ReaderInitParam(ignore_obj_size=True)
    rd = lcs.TraceReader(trace=fn, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE, reader_init_params=rp)
    t = time.perf_counter()
    make(lcs, cand, cap).process_trace(rd)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/switcher_v2")
    ap.add_argument("--fraction", type=float, default=0.10, help="cache size as a fraction of unique objects")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.tmp, exist_ok=True)
    T0 = time.time()
    fr = f"{a.fraction:g}"
    sfx = "" if fr == "0.1" else f"_f{fr}"   # 10% keeps the original file names
    tune = pd.read_csv(os.path.join(a.out, f"s3fifo_tuning_raw_f{fr}.csv")).set_index("trace")
    sw = pd.read_csv(os.path.join(a.out, f"switching_raw_f{fr}.csv")).set_index("trace")
    feats = pd.read_csv(os.path.join(a.out, "selector_features.csv"))
    fcols = [c for c in feats.columns if c not in ("trace", "split", "window", "feat_s")]
    F = {(r.trace, int(r.window)): r[fcols].values.astype(float) for _, r in feats.iterrows()}
    traces = list(tune.index)
    cont = {t: {c: np.array([tune.loc[t, f"cont_{c}_w{w + 1}"] for w in range(NWIN)]) for c in CANDS} for t in traces}
    src = {t: t.split(":")[0] for t in traces}
    P = paths()

    # ---- 1. sampled shadow estimates (resumable)
    ep = os.path.join(a.out, f"shadow_estimates{sfx}.csv")
    est_rows = pd.read_csv(ep).to_dict("records") if os.path.exists(ep) else []
    have = {(r["trace"], r["rate"]) for r in est_rows}
    for t in traces:
        if all((t, r) in have for r in RATES):
            continue
        ids = read_records(P[t], W * NWIN)["id"].tolist()
        cap = int(tune.loc[t, "capacity"])
        for rate in RATES:
            if (t, rate) in have:
                continue
            est, sec, n_s = shadow_estimates(lcs, ids, cap, rate)
            for cand, v in est.items():
                est_rows.append({"trace": t, "rate": rate, "cand": cand, "seconds_all17": sec, "n_sampled": n_s,
                                 **{f"w{w + 1}": v[w] for w in range(NWIN)}})
        pd.DataFrame(est_rows).to_csv(ep, index=False)
        print(f"  estimates {t} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    est = pd.DataFrame(est_rows)
    E = {"exact": cont}
    for rate in RATES:
        e = est[est.rate == rate]
        E[f"mini{int(rate * 100)}"] = {t: {c: g[[f"w{w + 1}" for w in range(NWIN)]].values[0].astype(float)
                                           for c, g in e[e.trace == t].groupby("cand")} for t in traces}

    def switch_costs(ts):
        cs = []
        for t in ts:
            plan = sw.loc[t, "plan"].split(">")
            cs += [sw.loc[t, f"cont_{plan[w]}_w{w + 1}"] - sw.loc[t, f"carry_w{w + 1}"]
                   for w in range(1, NWIN) if plan[w] != plan[w - 1]]
        return float(np.mean(cs))

    # ---- 2. switcher variants
    VARIANTS = [("exact", 1), ("exact", 2), ("mini1", 1), ("mini1", 2), ("mini5", 1)]
    plan_cache = {}
    rows, fid, info = [], [], []
    for held in sorted(set(src.values())):
        train = [t for t in traces if src[t] != held]
        test = [t for t in traces if src[t] == held]
        wins = pd.Series([max(CANDS, key=lambda c: cont[t][c][w]) for t in train for w in range(NWIN)])
        experts = [DEFAULT] + [c for c in wins.value_counts().index if c != DEFAULT][:K - 1]
        cost = switch_costs(train) * 100
        for t in test:   # estimate fidelity on the experts
            for name in ("mini1", "mini5"):
                for w in range(NWIN):
                    ex = [cont[t][e][w] for e in experts]
                    es = [E[name][t][e][w] for e in experts]
                    fid.append({"trace": t, "est": name, "window": w + 1,
                                "same_winner": int(np.argmax(ex) == np.argmax(es)),
                                "spearman": pd.Series(ex).corr(pd.Series(es), method="spearman"),
                                "abs_err_pts": float(np.mean(np.abs(np.array(ex) - np.array(es)))) * 100})
        for est_name, hist in VARIANTS:
            src_e = E[est_name]

            def xrow(t, w):
                g = np.array([src_e[t][e][w] - src_e[t][DEFAULT][w] for e in experts]) * 100
                x = [g, F[(t, w + 1)]]
                if hist == 2:
                    wp = max(w - 1, 0)
                    gp = np.array([src_e[t][e][wp] - src_e[t][DEFAULT][wp] for e in experts]) * 100
                    x += [gp, g - gp, F[(t, w + 1)] - F[(t, wp + 1)]]
                return np.nan_to_num(np.concatenate(x))

            X = np.array([xrow(t, w) for t in train for w in range(NWIN - 1)])
            Y = np.array([[(cont[t][e][w + 1] - cont[t][DEFAULT][w + 1]) * 100 for e in experts]
                          for t in train for w in range(NWIN - 1)])
            groups = np.array([t for t in train for w in range(NWIN - 1)])
            cv = {}
            for mname, mk in make_models().items():
                g = []
                for tt in np.unique(groups):
                    m = groups == tt
                    Pm = fit_k(mk, X[~m], Y[~m], X[m])
                    g += [Y[i, k] for i, k in zip(np.flatnonzero(m), Pm.argmax(axis=1))]
                cv[mname] = float(np.mean(g))
            best = max(cv, key=cv.get)
            info.append(f"held-out {held} {est_name}/h{hist}: model {best} (CV {cv[best]:+.2f}), cost {cost:.2f} pts")
            for t in test:
                pred = fit_k(make_models()[best], X, Y, np.array([xrow(t, w) for w in range(NWIN - 1)]))
                plan = [DEFAULT]
                for w in range(1, NWIN):
                    cur = experts.index(plan[-1])
                    score = pred[w - 1] - pred[w - 1][cur] - cost * (np.arange(len(experts)) != cur)
                    plan.append(experts[int(np.argmax(score))] if score.max() > 0 else plan[-1])
                key = (t, ">".join(plan))
                if key not in plan_cache:
                    if all(p == DEFAULT for p in plan):
                        plan_cache[key] = cont[t][DEFAULT]
                    else:
                        ids = read_records(P[t], W * NWIN)["id"].tolist()
                        plan_cache[key] = run_plan(lcs, ids, int(tune.loc[t, "capacity"]), plan)
                hr = plan_cache[key]
                rows.append({"trace": t, "source": held, "variant": f"{est_name}_h{hist}", "plan": key[1],
                             "switches": sum(plan[w] != plan[w - 1] for w in range(1, NWIN)),
                             **{f"w{w + 1}": hr[w] for w in range(NWIN)}})
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"switcher_v2_raw{sfx}.csv"), index=False)
            print(f"  fold {held} variant {est_name}/h{hist} done ({(time.time() - T0) / 60:.1f} min)", flush=True)
    pd.DataFrame(fid).to_csv(os.path.join(a.out, f"shadow_fidelity{sfx}.csv"), index=False)

    # ---- 3. overhead, native libCacheSim time on 6 traces (one per source plus two); 10% run only
    ov = []
    sample_traces = [] if sfx else [t for t in traces if t.startswith(("msr:", "cp:"))][:2] + \
                    [t for t in traces if t.startswith("ali:")][:2] + [t for t in traces if t.startswith("meta:")][:2]
    for t in sample_traces:
        rec = read_records(P[t], W * NWIN)
        cap = int(tune.loc[t, "capacity"])
        full = os.path.join(a.tmp, "full.bin")
        rec.tofile(full)
        base = native_seconds(lcs, full, "ARC", cap)
        exact6 = sum(native_seconds(lcs, full, c, cap) for c in CANDS[:0] + [DEFAULT, "WTinyLFU", "ARC", "LHD",
                                                                              "S3FIFO(s=0.02,t=2)", "S3FIFO(s=0.1,t=1)"])
        r = {"trace": t, "n": len(rec), "arc_s": base, "shadows_exact6_s": exact6}
        for rate in RATES:
            sub = rec[sample_mask(rec["id"], rate)]
            fn = os.path.join(a.tmp, "sub.bin")
            sub.tofile(fn)
            c = max(1, int(round(cap * rate)))
            r[f"shadows_mini{int(rate * 100)}_6_s"] = sum(
                native_seconds(lcs, fn, x, c) for x in [DEFAULT, "WTinyLFU", "ARC", "LHD",
                                                        "S3FIFO(s=0.02,t=2)", "S3FIFO(s=0.1,t=1)"])
        ov.append(r)
    if not sfx:
        pd.DataFrame(ov).to_csv(os.path.join(a.out, "switcher_overhead.csv"), index=False)

    # ---- 4. summary
    df = pd.DataFrame(rows)
    ws = range(2, NWIN + 1)
    out = []
    for t in traces:
        m_arc = np.mean([1 - cont[t]["ARC"][w - 1] for w in ws])
        red = lambda miss: (m_arc - miss) / m_arc * 100  # noqa: E731
        o = {"trace": t, "source": src[t], "S3FIFO_default": red(np.mean([1 - cont[t][DEFAULT][w - 1] for w in ws]))}
        for v, g in df[df.trace == t].groupby("variant"):
            r = g.iloc[0]
            o[v] = red(np.mean([1 - r[f"w{w}"] for w in ws]))
            o[f"sw_{v}"] = r.switches
        out.append(o)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, f"switcher_v2_summary{sfx}.csv"), index=False)
    vnames = [f"{e}_h{h}" for e, h in VARIANTS]
    L = [f"SWITCHER v2, leave-one-source-out, cache {a.fraction:.0%}, windows 2-6; relative miss-ratio reduction vs ARC (%), "
         f"mean [95% CI over {len(s)} traces]"]
    m, lo, hi = boot(s.S3FIFO_default.values)
    L.append(f"  {'S3FIFO_default':<12} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
    for v in vnames:
        m, lo, hi = boot(s[v].values)
        d, dlo, dhi = boot((s[v] - s.S3FIFO_default).values)
        L.append(f"  {v:<12} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   minus default {d:+.2f}% [{dlo:+.2f}, {dhi:+.2f}]   "
                 f"better on {int(((s[v] - s.S3FIFO_default) > 1e-9).sum())}/{len(s)}, "
                 f"worse on {int(((s[v] - s.S3FIFO_default) < -1e-9).sum())}/{len(s)}, switches {s[f'sw_{v}'].mean():.2f}")
    f = pd.DataFrame(fid)
    for name, g in f.groupby("est"):
        L.append(f"  estimate fidelity {name}: same winner among 6 experts {g.same_winner.mean():.0%} of windows, "
                 f"Spearman {g.spearman.mean():.2f}, mean abs error {g.abs_err_pts.mean():.2f} hit-rate points")
    o = pd.DataFrame(ov)
    if len(o):
      L.append(f"  overhead (native libCacheSim, {len(o)} traces, 3M requests, 6 experts): ARC alone {o.arc_s.mean():.2f} s; "
             f"exact shadows {o.shadows_exact6_s.mean():.2f} s ({o.shadows_exact6_s.mean() / o.arc_s.mean():.1f}x ARC); "
             + "; ".join(f"mini {int(r * 100)}% {o[f'shadows_mini{int(r * 100)}_6_s'].mean():.3f} s "
                         f"({o[f'shadows_mini{int(r * 100)}_6_s'].mean() / o.arc_s.mean():.2f}x ARC)" for r in RATES))
    L += info
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, f"switcher_v2_summary{sfx}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
