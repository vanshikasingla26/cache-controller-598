"""Window-length sensitivity (risk listed in Progress Report 2): every result so far uses 6 windows of 500k
requests. Re-run the non-learned strategies with 250k and 1M windows over the same first 3M requests.

Per trace (same 38 traces and capacities as s3fifo_tuning.py at this cache size):
  1. run each of the 17 candidates (8 S3-FIFO settings + 9 policies) continuously over 3M requests with
     per-request get() and keep its per-request hit vector, so any window length can be scored;
     sanity check: 500k-window hit rates must match the native ones in s3fifo_tuning_raw (max diff reported)
  2. for each window length, execute with carry mechanics (as s3fifo_tuning.run_plan) and score windows 2..N:
       S3FIFO_default, retune (best S3-FIFO setting on the last window), policy switching (best of 9 policies),
       hybrid (best of all 17), margin (hybrid that switches only past a margin chosen leave-one-source-out
       from margin_switcher.MARGINS, estimated score, switching cost from switching_raw clamped at zero),
       and the no-cost ceiling.
Metric as everywhere: relative miss-ratio reduction vs ARC over windows 2..N, mean [95% CI] over traces.

  python -u scripts/window_sensitivity.py --fraction 0.01 --out results/real --tmp /tmp/ws
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402
from scripts.policy_selection_headroom import POL, boot  # noqa: E402
from scripts.s3fifo_tuning import SETTINGS, make  # noqa: E402
from scripts.cost_aware_switcher import CANDS, DEFAULT, paths  # noqa: E402
from scripts.margin_switcher import MARGINS  # noqa: E402

N = 3_000_000
WINDOWS = (250_000, 500_000, 1_000_000)


def hits_vector(lcs, ids, cand, cap):
    R = lcs.Request
    c = make(lcs, cand, cap)
    return np.fromiter((bool(c.get(R(obj_size=1, obj_id=o))) for o in ids), dtype=bool, count=len(ids))


def run_plan_w(lcs, ids, cap, plan, W):
    """carry mechanics as s3fifo_tuning.run_plan, any window length"""
    R = lcs.Request
    nwin = len(plan)
    last, prev_counts = {}, {}
    cur = plan[0]
    cache = make(lcs, cur, cap)
    hr = np.zeros(nwin)
    for w in range(nwin):
        if w > 0 and plan[w] != cur:
            new = make(lcs, plan[w], cap)
            resident = [o for o in last if cache.find(R(obj_size=1, obj_id=o), False) is not None]
            resident.sort(key=last.get)
            for o in resident:
                new.insert(R(obj_size=1, obj_id=o))
                if prev_counts.get(o, 0) >= 2:
                    new.get(R(obj_size=1, obj_id=o))
            cache, cur = new, plan[w]
        h, counts, base = 0, {}, w * W
        for t, o in enumerate(ids[w * W:(w + 1) * W]):
            h += bool(cache.get(R(obj_size=1, obj_id=o)))
            last[o] = base + t
            counts[o] = counts.get(o, 0) + 1
        prev_counts = counts
        hr[w] = h / W
    return hr


def reactive_plan(cont, pool, start, margin=0.0):
    nwin = len(next(iter(cont.values())))
    plan = [start]
    for w in range(1, nwin):
        best = max(pool, key=lambda c: cont[c][w - 1])
        plan.append(best if (cont[best][w - 1] - cont[plan[-1]][w - 1]) * 100 > margin else plan[-1])
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/ws")
    a = ap.parse_args()
    import libcachesim as lcs
    fr = f"{a.fraction:g}"
    os.makedirs(a.tmp, exist_ok=True)
    tune = pd.read_csv(os.path.join(a.out, f"s3fifo_tuning_raw_f{fr}.csv")).set_index("trace")
    sw = pd.read_csv(os.path.join(a.out, f"switching_raw_f{fr}.csv")).set_index("trace")
    traces = list(tune.index)
    src = {t: t.split(":")[0] for t in traces}
    P = paths()
    T0 = time.time()

    # 1. per-request hit vectors (cached on disk, packed bits)
    check = []
    for t in traces:
        fn = os.path.join(a.tmp, f"{t.replace(':', '_')}_f{fr}.npz")
        if os.path.exists(fn):
            continue
        ids = read_records(P[t], N)["id"].tolist()
        cap = int(tune.loc[t, "capacity"])
        H = {c: hits_vector(lcs, ids, c, cap) for c in CANDS}
        d = max(abs(H[c][w * 500_000:(w + 1) * 500_000].mean() - tune.loc[t, f"cont_{c}_w{w + 1}"])
                for c in CANDS for w in range(6))
        check.append({"trace": t, "max_abs_diff_vs_native": d})
        np.savez_compressed(fn, **{c: np.packbits(H[c]) for c in CANDS}, n=len(ids))
        print(f"  hits {t}: max diff vs native {d:.5f} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    if check:
        pd.DataFrame(check).to_csv(os.path.join(a.out, f"window_sens_check_f{fr}.csv"), index=False)

    def load(t):
        z = np.load(os.path.join(a.tmp, f"{t.replace(':', '_')}_f{fr}.npz"))
        n = int(z["n"])
        return {c: np.unpackbits(z[c])[:n].astype(bool) for c in CANDS}

    def switch_cost(ts):
        cs = []
        for t in ts:
            plan = sw.loc[t, "plan"].split(">")
            cs += [sw.loc[t, f"cont_{plan[w]}_w{w + 1}"] - sw.loc[t, f"carry_w{w + 1}"]
                   for w in range(1, 6) if plan[w] != plan[w - 1]]
        return max(0.0, float(np.mean(cs)))

    # 2. strategies per window length
    rawp = os.path.join(a.out, f"window_sens_raw_f{fr}.csv")
    rows = pd.read_csv(rawp).to_dict("records") if os.path.exists(rawp) else []
    done = {(r["trace"], r["W"]) for r in rows}
    sets = list(SETTINGS)
    for W in WINDOWS:
        nwin = N // W
        conts = {}
        for t in traces:
            H = load(t)
            conts[t] = {c: H[c][:nwin * W].reshape(nwin, W).mean(axis=1) for c in CANDS}
        margin_of = {}
        for held in sorted(set(src.values())):
            train = [t for t in traces if src[t] != held]
            cost = switch_cost(train)

            def score(t, m):
                p = reactive_plan(conts[t], CANDS, DEFAULT, m)
                return np.mean([conts[t][p[w]][w] - cost * (p[w] != p[w - 1]) for w in range(1, nwin)])
            sc = {m: np.mean([score(t, m) for t in train]) for m in MARGINS}
            margin_of[held] = max(MARGINS, key=lambda m: (sc[m], m))
        for t in traces:
            if (t, W) in done:
                continue
            cont = conts[t]
            cap = int(tune.loc[t, "capacity"])
            ids = None
            plans = {"retune": reactive_plan(cont, sets, DEFAULT),
                     "polswitch": reactive_plan(cont, POL, "ARC"),
                     "hybrid": reactive_plan(cont, CANDS, DEFAULT),
                     "margin": reactive_plan(cont, CANDS, DEFAULT, margin_of[src[t]])}
            r = {"trace": t, "source": src[t], "W": W, "margin_used": margin_of[src[t]]}
            m_arc = np.mean(1 - cont["ARC"][1:])
            red = lambda hr: (m_arc - np.mean(1 - np.asarray(hr)[1:])) / m_arc * 100  # noqa: E731
            r["S3FIFO_default"] = red(cont[DEFAULT])
            r["ceiling"] = red([max(cont[c][w] for c in CANDS) for w in range(nwin)])
            memo = {}
            for k, p in plans.items():
                key = ">".join(p)
                if key not in memo:
                    if all(x == p[0] for x in p):
                        memo[key] = cont[p[0]]
                    else:
                        if ids is None:
                            ids = read_records(P[t], N)["id"].tolist()
                        memo[key] = run_plan_w(lcs, ids, cap, p, W)
                r[k] = red(memo[key])
                r[f"{k}_switches"] = sum(p[w] != p[w - 1] for w in range(1, nwin))
            rows.append(r)
            pd.DataFrame(rows).to_csv(rawp, index=False)
            print(f"  W={W} {t} ({(time.time() - T0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    L = [f"WINDOW-LENGTH SENSITIVITY, cache {a.fraction:.0%}, first 3M requests, windows 2..N; difference in relative "
         "miss-ratio reduction vs ARC from default S3-FIFO (%), mean [95% CI over traces]"]
    if check:
        L.append(f"  per-request hit vectors vs native 500k-window hit rates: max abs diff "
                 f"{pd.DataFrame(check).max_abs_diff_vs_native.max():.5f}")
    for W in WINDOWS:
        g = df[df.W == W]
        m, lo, hi = boot(g.S3FIFO_default.values)
        L.append(f"  W={W // 1000}k ({N // W} windows, {len(g)} traces): default S3-FIFO vs ARC {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
        for k in ("retune", "polswitch", "hybrid", "margin", "ceiling"):
            d = (g[k] - g.S3FIFO_default).values
            m, lo, hi = boot(d)
            sw_ = f", switches {g[f'{k}_switches'].mean():.2f}" if f"{k}_switches" in g else ""
            L.append(f"    {k:<10} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]  better on {int((d > 0).sum())}/{len(g)}{sw_}")
        L.append(f"    margins chosen: {dict(g.groupby('source').margin_used.first())}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, f"window_sens_summary_f{fr}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
