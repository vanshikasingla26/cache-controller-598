"""Tune the near-best policy's parameters per window instead of (or as well as) switching policies
(Progress Report 1 feedback). Same traces, windows (6 x 500k requests), causal rule, switching mechanics
(carry) and metric as scripts/switching.py, so the numbers are directly comparable.

Candidates
  S3-FIFO settings: small_size_ratio in {0.02, 0.1, 0.25, 0.5} x move_to_main_threshold in {1, 2}
                    (0.1 / 2 is libCacheSim's default)
  policies:         the 9 online policies with default settings (from switching.py's raw file)
Strategies (windows 2..6, relative miss-ratio reduction vs ARC)
  S3-FIFO default            the default setting, run continuously
  S3-FIFO tuned once         the setting best on window 1, run continuously (no switching)
  S3-FIFO re-tuned           each window, the setting best on the previous window (switch with carry)
  hybrid                     each window, the best of policies + settings on the previous window (carry)
  policy switching           from switching.py (carry)
  ideal (no cost)            best continuous candidate per window, for settings only and for everything

  python -u scripts/s3fifo_tuning.py --fraction 0.10 --out results/real
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
SETTINGS = {f"S3FIFO(s={s},t={t})": dict(small_size_ratio=s, move_to_main_threshold=t)
            for s in (0.02, 0.1, 0.25, 0.5) for t in (1, 2)}


def make(lcs, cand, cap):
    if cand in SETTINGS:
        return lcs.S3FIFO(cache_size=cap, **SETTINGS[cand])
    return getattr(lcs, cand)(cache_size=cap)


def continuous_windows(lcs, fn, cand, cap):
    hits = [0.0]
    for k in range(1, NWIN + 1):
        rp = lcs.ReaderInitParam(ignore_obj_size=True)
        rp.cap_at_n_req = k * W
        rd = lcs.TraceReader(trace=fn, trace_type=lcs.TraceType.ORACLE_GENERAL_TRACE, reader_init_params=rp)
        miss, _ = make(lcs, cand, cap).process_trace(rd)
        hits.append((1 - miss) * k * W)
    return np.diff(hits) / W


def run_plan(lcs, ids, cap, plan):
    """carry mechanics, identical to switching.py"""
    R = lcs.Request
    last, prev_counts = {}, {}
    cur = plan[0]
    cache = make(lcs, cur, cap)
    hr = np.zeros(NWIN)
    for w in range(NWIN):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/s3tune")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.tmp, exist_ok=True)
    sw = pd.read_csv(os.path.join(a.out, f"switching_raw_f{a.fraction:g}.csv")).set_index("trace")
    raw = os.path.join(a.out, f"s3fifo_tuning_raw_f{a.fraction:g}.csv")
    rows = pd.read_csv(raw).to_dict("records") if os.path.exists(raw) else []
    done = {r["trace"] for r in rows}
    T0 = time.time()
    for tag, pat in DATASETS.items():
        for path in sorted(glob.glob(pat)):
            name = name_of(path, tag)
            if name not in sw.index or name in done:
                continue
            s = sw.loc[name]
            cap = int(s.capacity)
            rec = read_records(path, W * NWIN)
            ids = rec["id"].tolist()
            fn = os.path.join(a.tmp, "t.bin")
            rec.tofile(fn)
            cont = {c: continuous_windows(lcs, fn, c, cap) for c in SETTINGS}
            os.remove(fn)
            for p in POL:
                cont[p] = np.array([s[f"cont_{p}_w{w + 1}"] for w in range(NWIN)])
            sets, every = list(SETTINGS), list(SETTINGS) + POL
            once = max(sets, key=lambda c: cont[c][0])
            tune_plan = ["S3FIFO(s=0.1,t=2)"] + [max(sets, key=lambda c: cont[c][w - 1]) for w in range(1, NWIN)]
            hyb_plan = ["ARC"] + [max(every, key=lambda c: cont[c][w - 1]) for w in range(1, NWIN)]
            r = {"trace": name, "source": tag, "capacity": cap, "tuned_once": once,
                 "retune_plan": ">".join(tune_plan), "hybrid_plan": ">".join(hyb_plan)}
            for c in every:
                r.update({f"cont_{c}_w{w + 1}": cont[c][w] for w in range(NWIN)})
            r.update({f"retune_w{w + 1}": v for w, v in enumerate(run_plan(lcs, ids, cap, tune_plan))})
            r.update({f"hybrid_w{w + 1}": v for w, v in enumerate(run_plan(lcs, ids, cap, hyb_plan))})
            r.update({f"polswitch_w{w + 1}": s[f"carry_w{w + 1}"] for w in range(NWIN)})
            rows.append(r)
            pd.DataFrame(rows).to_csv(raw, index=False)
            print(f"  {name}: once {once} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    summarize(pd.DataFrame(rows), a)


def summarize(df, a):
    ws = range(2, NWIN + 1)
    sets, every = list(SETTINGS), list(SETTINGS) + POL
    out = []
    for _, r in df.iterrows():
        m_arc = np.mean([1 - r[f"cont_ARC_w{w}"] for w in ws])

        def red(miss):
            return (m_arc - miss) / m_arc * 100
        out.append({
            "trace": r.trace,
            "S3FIFO_default": red(np.mean([1 - r[f"cont_S3FIFO(s=0.1,t=2)_w{w}"] for w in ws])),
            "S3FIFO_tuned_once": red(np.mean([1 - r[f"cont_{r.tuned_once}_w{w}"] for w in ws])),
            "S3FIFO_retuned": red(np.mean([1 - r[f"retune_w{w}"] for w in ws])),
            "policy_switching": red(np.mean([1 - r[f"polswitch_w{w}"] for w in ws])),
            "hybrid": red(np.mean([1 - r[f"hybrid_w{w}"] for w in ws])),
            "ideal_settings": red(np.mean([1 - max(r[f"cont_{c}_w{w}"] for c in sets) for w in ws])),
            "ideal_all": red(np.mean([1 - max(r[f"cont_{c}_w{w}"] for c in every) for w in ws]))})
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, f"s3fifo_tuning_summary_f{a.fraction:g}.csv"), index=False)
    L = [f"cache {a.fraction:.0%}: {len(s)} traces, windows 2-6; relative miss-ratio reduction vs ARC (%), "
         "mean [95% CI over traces], positive/n"]
    for c in ("S3FIFO_default", "S3FIFO_tuned_once", "S3FIFO_retuned", "policy_switching", "hybrid",
              "ideal_settings", "ideal_all"):
        m, lo, hi = boot(s[c].values)
        L.append(f"  {c:<18} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   {int((s[c] > 0).sum())}/{len(s)}")
    for c in ("S3FIFO_tuned_once", "S3FIFO_retuned", "policy_switching", "hybrid"):
        m, lo, hi = boot((s[c] - s.S3FIFO_default).values)
        L.append(f"  {c} minus S3FIFO_default: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, f"s3fifo_tuning_summary_f{a.fraction:g}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
