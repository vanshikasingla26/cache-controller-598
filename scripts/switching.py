"""Policy switching with real mechanics and MEASURED switching cost (Progress Report 1 feedback).

Each trace (first 3M requests) is cut into 6 windows of 500k requests. Strategy (causal, no learning):
at every window boundary pick the policy with the best hit rate on the window that just ended, and switch
if it differs from the current one. Candidate policies: the 9 online libCacheSim policies.

Two switching mechanics, fixed in advance:
  carry   read the old policy's exact resident set (libCacheSim find without side effects), insert it into
          the new policy oldest-first by last access so recency order is kept; objects accessed >= 2 times
          in the window that just ended get one extra (uncounted) access right after insertion, so
          frequency-based policies see them as hot. (In S3-FIFO this keeps those objects plus whatever
          still fits the small queue.)
  replay  the new policy starts empty and replays the window that just ended (500k requests, uncounted),
          so it builds its own metadata; its resident set can differ from the old policy's.
Switching cost of a switch into B at window w = hit rate of B run continuously from the start, in window
w, minus hit rate of the switched cache in window w (positive = the switch lost hits). Continuous
per-window hit rates come from native libCacheSim runs on growing prefixes (exact, deterministic).

Reported vs ARC on windows 2..6 (relative miss-ratio reduction): always S3-FIFO, causal switching with
carry and with replay, and the cost-free upper bound (best continuous policy per window).

  python -u scripts/switching.py --fraction 0.05 --out results/real
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
from scripts.policy_selection_headroom import DATASETS, POL, boot, name_of  # noqa: E402

W, NWIN = 500_000, 6


def continuous_windows(lcs, fn, pol, cap):
    """Per-window hit rate of `pol` run continuously from request 0 (exact, via growing prefixes)."""
    hits = [0.0]
    for k in range(1, NWIN + 1):
        rp = lcs.ReaderInitParam(ignore_obj_size=True)
        rp.cap_at_n_req = k * W
        rd = lcs.TraceReader(trace=fn, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE, reader_init_params=rp)
        miss, _ = getattr(lcs, pol)(cache_size=cap).process_trace(rd)
        hits.append((1 - miss) * k * W)
    return np.diff(hits) / W


def run_switching(lcs, ids, cap, plan, mode):
    """Serve the trace window by window with policy plan[w]; switch with `mode` mechanics."""
    R = lcs.Request
    last, prev_counts = {}, {}
    cur = plan[0]
    cache = getattr(lcs, cur)(cache_size=cap)
    hr = np.zeros(NWIN)
    switch_s = []
    for w in range(NWIN):
        seg = ids[w * W:(w + 1) * W]
        if w > 0 and plan[w] != cur:
            t0 = time.perf_counter()
            new = getattr(lcs, plan[w])(cache_size=cap)
            if mode == "carry":
                resident = [o for o in last if cache.find(R(obj_size=1, obj_id=o), False) is not None]
                resident.sort(key=last.get)
                for o in resident:
                    new.insert(R(obj_size=1, obj_id=o))
                    if prev_counts.get(o, 0) >= 2:
                        new.get(R(obj_size=1, obj_id=o))
            else:   # replay the window that just ended
                for o in ids[(w - 1) * W:w * W]:
                    new.get(R(obj_size=1, obj_id=o))
            switch_s.append(time.perf_counter() - t0)
            cache, cur = new, plan[w]
        h, counts = 0, {}
        base = w * W
        for t, o in enumerate(seg):
            h += bool(cache.get(R(obj_size=1, obj_id=o)))
            last[o] = base + t
            counts[o] = counts.get(o, 0) + 1
        prev_counts = counts
        hr[w] = h / len(seg)
    return hr, switch_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/switch")
    ap.add_argument("--limit", type=int, default=0, help="only the first N traces (for timing)")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.tmp, exist_ok=True)
    halves = pd.read_csv(os.path.join(a.out, "policy_halves_raw.csv"))
    used = set(halves[halves.one_hit <= 0.90].trace)
    raw = os.path.join(a.out, f"switching_raw_f{a.fraction:g}.csv")
    rows = pd.read_csv(raw).to_dict("records") if os.path.exists(raw) else []
    done = {r["trace"] for r in rows}
    T0, n = time.time(), 0
    for tag, pat in DATASETS.items():
        for path in sorted(glob.glob(pat)):
            name = name_of(path, tag)
            if name not in used or name in done:
                continue
            if a.limit and n >= a.limit:
                break
            n += 1
            rec = read_records(path, W * NWIN)
            ids = rec["id"].tolist()
            cap = max(1, int(len(np.unique(rec["id"])) * a.fraction))
            fn = os.path.join(a.tmp, "t.bin")
            rec.tofile(fn)
            cont = {p: continuous_windows(lcs, fn, p, cap) for p in POL}
            os.remove(fn)
            plan = ["ARC"] + [max(POL, key=lambda p: cont[p][w - 1]) for w in range(1, NWIN)]
            r = {"trace": name, "source": tag, "capacity": cap, "plan": ">".join(plan),
                 "n_switches": sum(plan[w] != plan[w - 1] for w in range(1, NWIN))}
            for p in POL:
                r.update({f"cont_{p}_w{w + 1}": cont[p][w] for w in range(NWIN)})
            for mode in ("carry", "replay"):
                hr, sw = run_switching(lcs, ids, cap, plan, mode)
                r.update({f"{mode}_w{w + 1}": hr[w] for w in range(NWIN)})
                r[f"{mode}_switch_s"] = float(np.mean(sw)) if sw else 0.0
            rows.append(r)
            pd.DataFrame(rows).to_csv(raw, index=False)
            print(f"  {name}: plan {r['plan']} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    summarize(pd.DataFrame(rows), a)


def summarize(df, a):
    ws = range(2, NWIN + 1)

    def miss(col_fmt, r):
        return np.mean([1 - r[col_fmt.format(w=w)] for w in ws])

    out = []
    for _, r in df.iterrows():
        m_arc = miss("cont_ARC_w{w}", r)
        best = np.mean([1 - max(r[f"cont_{p}_w{w}"] for p in POL) for w in ws])
        o = {"trace": r.trace, "n_switches": r.n_switches,
             "always_S3FIFO": (m_arc - miss("cont_S3FIFO_w{w}", r)) / m_arc * 100,
             "switch_carry": (m_arc - miss("carry_w{w}", r)) / m_arc * 100,
             "switch_replay": (m_arc - miss("replay_w{w}", r)) / m_arc * 100,
             "ideal_no_cost": (m_arc - best) / m_arc * 100}
        plan = r.plan.split(">")
        for mode in ("carry", "replay"):
            costs = [(r[f"cont_{plan[w]}_w{w + 1}"] - r[f"{mode}_w{w + 1}"]) * 100
                     for w in range(1, NWIN) if plan[w] != plan[w - 1]]
            o[f"cost_{mode}"] = float(np.mean(costs)) if costs else np.nan
        out.append(o)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, f"switching_summary_f{a.fraction:g}.csv"), index=False)
    L = [f"cache {a.fraction:.0%}: {len(s)} traces, windows 2-6; relative miss-ratio reduction vs ARC (%), "
         "mean [95% CI over traces], positive/n"]
    for c in ("always_S3FIFO", "switch_carry", "switch_replay", "ideal_no_cost"):
        m, lo, hi = boot(s[c].values)
        L.append(f"  {c:<15} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   {int((s[c] > 0).sum())}/{len(s)}")
    for c in ("switch_carry", "switch_replay"):
        m, lo, hi = boot((s[c] - s.always_S3FIFO).values)
        L.append(f"  {c} minus always_S3FIFO: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
    for mode in ("carry", "replay"):
        v = s[f"cost_{mode}"].dropna().values
        if len(v):
            m, lo, hi = boot(v)
            L.append(f"  measured cost per switch ({mode}): {m:+.2f} hit-rate points in the window after the "
                     f"switch [{lo:+.2f}, {hi:+.2f}] ({len(v)} traces with >= 1 switch)")
    L.append(f"  switches per trace: mean {s.n_switches.mean():.1f}, traces with none {int((s.n_switches == 0).sum())}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, f"switching_summary_f{a.fraction:g}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
