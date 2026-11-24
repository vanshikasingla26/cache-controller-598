"""Real object sizes (risk listed in Progress Report 2): every other result treats objects as unit size.
Re-run the non-learned strategies with each request's real size, cache capacity in bytes.

Per trace (same 38 traces, first 3M requests, six 500k windows):
  capacity = fraction x total bytes of the unique objects (first size seen per object)
  1. run each of the 17 candidates (8 S3-FIFO settings + 9 policies) continuously with per-request get(),
     keeping per-request hits, to get every candidate's object and byte hit rate in every window
  2. strategies, as in Progress Report 2 Table 4, decided on object hit rate and executed with carry mechanics
     (resident objects copied oldest-first with their last size, extra touch for objects seen >= 2 times):
       S3FIFO_default, retune (best S3-FIFO setting last window), polswitch (best of 9 policies, starts at ARC),
       hybrid (best of all 17), margin (hybrid that switches only past a margin chosen leave-one-source-out from
       margin_switcher.MARGINS; switching cost measured here from the executed hybrid plans of training traces,
       clamped at zero), ceiling (best candidate per window, no switching cost)
Metrics, windows 2-6: relative reduction vs ARC of the object miss ratio and of the byte miss ratio.

  python -u scripts/size_aware.py --fraction 0.01 --out results/real --tmp /tmp/size
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
from scripts.window_sensitivity import reactive_plan  # noqa: E402

N, W, NWIN = 3_000_000, 500_000, 6
STRATS = ("retune", "polswitch", "hybrid", "margin")


def hits_vector(lcs, ids, sizes, cand, cap):
    R = lcs.Request
    c = make(lcs, cand, cap)
    return np.fromiter((bool(c.get(R(obj_size=int(s), obj_id=int(o)))) for o, s in zip(ids, sizes)),
                       dtype=bool, count=len(ids))


def run_plan_sized(lcs, ids, sizes, cap, plan):
    R = lcs.Request
    last, size_of, prev_counts = {}, {}, {}
    cur = plan[0]
    cache = make(lcs, cur, cap)
    hits = np.zeros(len(ids), dtype=bool)
    for w in range(NWIN):
        if w > 0 and plan[w] != cur:
            new = make(lcs, plan[w], cap)
            resident = [o for o in last if cache.find(R(obj_size=size_of[o], obj_id=o), False) is not None]
            resident.sort(key=last.get)
            for o in resident:
                new.insert(R(obj_size=size_of[o], obj_id=o))
                if prev_counts.get(o, 0) >= 2:
                    new.get(R(obj_size=size_of[o], obj_id=o))
            cache, cur = new, plan[w]
        counts = {}
        for t in range(w * W, (w + 1) * W):
            o, s = int(ids[t]), int(sizes[t])
            hits[t] = bool(cache.get(R(obj_size=s, obj_id=o)))
            last[o], size_of[o] = t, s
            counts[o] = counts.get(o, 0) + 1
        prev_counts = counts
    return hits


def window_rates(h, sizes):
    hw = h[:NWIN * W].reshape(NWIN, W)
    sw = sizes[:NWIN * W].reshape(NWIN, W).astype(float)
    return hw.mean(axis=1), (hw * sw).sum(axis=1) / sw.sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/size")
    a = ap.parse_args()
    import libcachesim as lcs
    fr = f"{a.fraction:g}"
    os.makedirs(a.tmp, exist_ok=True)
    traces = list(pd.read_csv(os.path.join(a.out, "s3fifo_tuning_raw_f0.1.csv")).trace)
    src = {t: t.split(":")[0] for t in traces}
    P = paths()
    T0 = time.time()

    def load_trace(t):
        r = read_records(P[t], N)
        ids, sizes = r["id"].astype(np.uint64), r["size"].astype(np.int64)
        _, first = np.unique(ids, return_index=True)
        cap = max(1, int(round(a.fraction * sizes[first].sum())))
        return ids, sizes, cap

    # 1. per-candidate hits (object and byte hit rate per window), cached per trace
    obj, byt, caps = {}, {}, {}
    for t in traces:
        fn = os.path.join(a.tmp, f"{t.replace(':', '_')}_f{fr}.npz")
        if not os.path.exists(fn):
            ids, sizes, cap = load_trace(t)
            idl, szl = ids.tolist(), sizes.tolist()
            o_, b_ = {}, {}
            for c in CANDS:
                o_[c], b_[c] = window_rates(hits_vector(lcs, idl, szl, c, cap), sizes)
            np.savez(fn, cap=cap, **{f"o_{c}": o_[c] for c in CANDS}, **{f"b_{c}": b_[c] for c in CANDS})
            print(f"  hits {t} ({(time.time() - T0) / 60:.1f} min)", flush=True)
        z = np.load(fn)
        caps[t] = int(z["cap"])
        obj[t] = {c: z[f"o_{c}"] for c in CANDS}
        byt[t] = {c: z[f"b_{c}"] for c in CANDS}

    # 2. executed plans
    rawp = os.path.join(a.out, f"size_aware_raw_f{fr}.csv")
    rows = pd.read_csv(rawp).to_dict("records") if os.path.exists(rawp) else []
    done = {(r["trace"], r["strategy"]) for r in rows}
    cache_ids = {}

    def execute(t, plan):
        if all(p == plan[0] for p in plan):
            return obj[t][plan[0]], byt[t][plan[0]]
        if t not in cache_ids:
            cache_ids.clear()
            cache_ids[t] = load_trace(t)
        ids, sizes, cap = cache_ids[t]
        return window_rates(run_plan_sized(lcs, ids.tolist(), sizes.tolist(), cap, plan), sizes)

    def record(t, k, plan):
        if (t, k) in done:
            return
        o, b = execute(t, plan)
        r = {"trace": t, "source": src[t], "strategy": k, "plan": ">".join(plan),
             "switches": sum(plan[w] != plan[w - 1] for w in range(1, NWIN))}
        r.update({f"obj_w{w + 1}": o[w] for w in range(NWIN)})
        r.update({f"byte_w{w + 1}": b[w] for w in range(NWIN)})
        rows.append(r)
        done.add((t, k))
        pd.DataFrame(rows).to_csv(rawp, index=False)
        print(f"  {k} {t}: {r['switches']} switches ({(time.time() - T0) / 60:.1f} min)", flush=True)

    sets = list(SETTINGS)
    for t in traces:
        record(t, "retune", reactive_plan(obj[t], sets, DEFAULT))
        record(t, "polswitch", reactive_plan(obj[t], POL, "ARC"))
        record(t, "hybrid", reactive_plan(obj[t], CANDS, DEFAULT))
    df = pd.DataFrame(rows)
    hy = df[df.strategy == "hybrid"].set_index("trace")

    def cost_of(ts):
        cs = []
        for t in ts:
            p = hy.loc[t, "plan"].split(">")
            cs += [obj[t][p[w]][w] - hy.loc[t, f"obj_w{w + 1}"] for w in range(1, NWIN) if p[w] != p[w - 1]]
        return max(0.0, float(np.mean(cs))) if cs else 0.0

    info = []
    for held in sorted(set(src.values())):
        train = [t for t in traces if src[t] != held]
        cost = cost_of(train)

        def score(t, m):
            p = reactive_plan(obj[t], CANDS, DEFAULT, m)
            return np.mean([obj[t][p[w]][w] - cost * (p[w] != p[w - 1]) for w in range(1, NWIN)])
        sc = {m: np.mean([score(t, m) for t in train]) for m in MARGINS}
        m_star = max(MARGINS, key=lambda m: (sc[m], m))
        info.append(f"held-out {held}: switching cost {cost * 100:.2f} pts, margin {m_star:g}")
        for t in [t for t in traces if src[t] == held]:
            record(t, "margin", reactive_plan(obj[t], CANDS, DEFAULT, m_star))

    df = pd.DataFrame(rows)
    out = []
    for t in traces:
        def red(kind, rates):
            arc = np.mean(1 - (obj if kind == "obj" else byt)[t]["ARC"][1:])
            return (arc - np.mean(1 - np.asarray(rates)[1:])) / arc * 100 if arc > 0 else np.nan
        o = {"trace": t, "source": src[t]}
        for kind, D in (("obj", obj), ("byte", byt)):
            o[f"{kind}_S3FIFO_default"] = red(kind, D[t][DEFAULT])
            o[f"{kind}_ceiling"] = red(kind, [max(D[t][c][w] for c in CANDS) for w in range(NWIN)])
            for k in STRATS:
                r = df[(df.trace == t) & (df.strategy == k)].iloc[0]
                o[f"{kind}_{k}"] = red(kind, [r[f"{kind}_w{w + 1}"] for w in range(NWIN)])
                o[f"switches_{k}"] = r.switches
        out.append(o)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, f"size_aware_summary_f{fr}.csv"), index=False)
    L = [f"REAL OBJECT SIZES, cache {a.fraction:.0%} of unique bytes, LOSO for the margin, windows 2-6, "
         f"{len(s)} traces; mean [95% CI over traces]"]
    for kind, name in (("obj", "object miss ratio"), ("byte", "byte miss ratio")):
        g = s.dropna(subset=[f"{kind}_S3FIFO_default"])
        m, lo, hi = boot(g[f"{kind}_S3FIFO_default"].values)
        L.append(f"  {name}: default S3-FIFO vs ARC {m:+.2f}% [{lo:+.2f}, {hi:+.2f}] ({len(g)} traces); "
                 "strategies minus default S3-FIFO:")
        for k in STRATS + ("ceiling",):
            d = (g[f"{kind}_{k}"] - g[f"{kind}_S3FIFO_default"]).values
            m, lo, hi = boot(d)
            sw_ = f", switches {g[f'switches_{k}'].mean():.2f}" if k != "ceiling" else ""
            L.append(f"    {k:<10} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]  better on {int((d > 0).sum())}/{len(g)}{sw_}")
    L += info
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, f"size_aware_summary_f{fr}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
