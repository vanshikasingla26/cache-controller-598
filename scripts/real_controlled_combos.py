"""Controlled shifts on real MSR volumes, part 2 (proposal feedback):
  * COMBINATIONS of dimensions: popularity+scan, popularity+tenant, scan+tenant
  * CACHEUS and S3-FIFO (libCacheSim, per-request) measured inside every shift scenario, so the
    state-of-the-art adaptive policies are compared in the transition window, not only overall.
Same volumes, slice (first 1M requests), shift point (500k), gate and metrics as
real_controlled_shifts.py. Each dimension uses its own id namespace, so combined changes cannot collide.

  python -u scripts/real_controlled_combos.py --dir data/msr --out results/real
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.detector import score_detection  # noqa: E402
from shiftcache.policies import ARC, Belady, next_use  # noqa: E402
from shiftcache.trace import Trace  # noqa: E402
from scripts.lru_gate_control import LRUGatedARC  # noqa: E402
from scripts.real_controlled_shifts import GATE, TEST_V, TRAIN_V  # noqa: E402
from scripts.real_experiment import detect_fixed_from_overlaps, load_vols, mask_from, overlaps  # noqa: E402

BASE = {"popularity": 1 << 41, "scan": 1 << 42, "tenant": 1 << 43}
SINGLE = [("popularity", 1.0), ("scan", 0.3), ("tenant", 0.5)]
COMBOS = [[("popularity", 1.0), ("scan", 0.3)], [("popularity", 1.0), ("tenant", 0.5)],
          [("scan", 0.3), ("tenant", 0.5)]]


def apply(keys, s, dim, mag, other, rng):
    pre, post = keys[:s], keys[s:].copy()
    m = len(post)
    if dim == "popularity":
        u, cnt = np.unique(post, return_counts=True)
        order = np.argsort(-cnt)
        cum = np.cumsum(cnt[order]) / cnt.sum()
        hot = u[order[: int(np.searchsorted(cum, 0.5)) + 1]]
        chosen = hot[rng.random(len(hot)) < mag] if mag < 1 else hot
        remap = {int(k): BASE[dim] + i for i, k in enumerate(chosen.tolist())}
        post = np.array([remap.get(int(k), int(k)) for k in post.tolist()], dtype=np.int64)
    elif dim == "scan":
        is_scan = rng.random(m) < mag
        out = np.empty(m, dtype=np.int64)
        out[is_scan] = BASE[dim] + np.arange(int(is_scan.sum()))
        out[~is_scan] = post[: m - int(is_scan.sum())]
        post = out
    elif dim == "tenant":
        is_o = rng.random(m) < mag
        out = np.empty(m, dtype=np.int64)
        out[is_o] = other[: int(is_o.sum())] + BASE[dim]
        out[~is_o] = post[: m - int(is_o.sum())]
        post = out
    return np.concatenate([pre, post])


def lcs_hits(lcs, keys, pol, cap):
    c = getattr(lcs, pol)(cache_size=cap)
    Req = lcs.Request
    return np.fromiter((bool(c.get(Req(obj_size=1, obj_id=k % (1 << 62)))) for k in keys.tolist()),
                       dtype=bool, count=len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/msr")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--win", type=int, default=50_000)
    ap.add_argument("--tol", type=int, default=20_000)
    a = ap.parse_args()
    import libcachesim as lcs
    T0 = time.time()
    vols = load_vols(a.dir, TRAIN_V + TEST_V, a.n)
    s = a.n // 2
    raw = os.path.join(a.out, "controlled_combos_raw.csv")
    rows = pd.read_csv(raw).to_dict("records") if os.path.exists(raw) else []
    done = {(r["vol"], r["scenario"]) for r in rows}
    for group, names in (("test", TEST_V), ("train", TRAIN_V)):
        for i, v in enumerate(names):
            other = vols[names[(i + 1) % len(names)]]
            scen = [[x] for x in SINGLE] + COMBOS
            for j, steps in enumerate(scen):
                label = "+".join(f"{d}{m}" for d, m in steps)
                if (v, label) in done:
                    continue
                if len(steps) == 1 and group == "train":
                    continue        # single-dimension train runs already exist in controlled_shifts_raw.csv
                rng = np.random.default_rng(5000 + 100 * i + j)
                k = vols[v]
                for d, m in steps:
                    k = apply(k, s, d, m, other, rng)
                n = len(k)
                tr = Trace(k, np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool), name=f"{v}:{label}",
                           transitions=[s])
                cap = max(1, int(tr.footprint * a.fraction))
                gt = transition_mask(n, [s], a.win)
                st, en = detect_fixed_from_overlaps(overlaps(tr, GATE.window, GATE.ref_windows), GATE)
                det = mask_from(st, en, n)
                sc = score_detection(st, [s], a.tol)
                r_arc = simulate(tr, ARC(cap))
                base = split_hit_rates(r_arc, gt)
                hits = {"MIN": simulate(tr, Belady(cap, next_use(k))).hits,
                        "LRU-gated[gt]": simulate(tr, LRUGatedARC(cap, k=8, gate=gt)).hits,
                        "LRU-gated[det]": simulate(tr, LRUGatedARC(cap, k=8, gate=det)).hits}
                if group == "test":
                    for pol in ("Cacheus", "S3FIFO"):
                        hits[pol] = lcs_hits(lcs, k, pol, cap)
                row = {"group": group, "vol": v, "scenario": label, "n_dims": len(steps),
                       "detected": sc["tp"], "gate_frac": float(det.mean()),
                       "arc_transition_hr": base["transition_hr"], "arc_overall_hr": r_arc.hit_rate}
                for name, h in hits.items():
                    row[f"dT_{name}"] = (h[gt].mean() - base["transition_hr"]) * 100
                    row[f"dO_{name}"] = (h.mean() - r_arc.hit_rate) * 100
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw, index=False)
                print(f"  {group} {v} {label} done ({(time.time() - T0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = [c for c in ["detected", "gate_frac", "dT_LRU-gated[gt]", "dO_LRU-gated[gt]", "dT_LRU-gated[det]",
                        "dO_LRU-gated[det]", "dT_Cacheus", "dO_Cacheus", "dT_S3FIFO", "dO_S3FIFO", "dT_MIN"]
            if c in df.columns]
    txt = ("TEST volumes (4), mean per scenario; dT = first 50k accesses after the shift, dO = whole trace; "
           "points vs ARC\n" + df[df.group == "test"].groupby("scenario", sort=False)[cols].mean().round(2).to_string())
    if (df.group == "train").any():
        txt += ("\n\nTRAIN volumes (4), combinations only\n" +
                df[df.group == "train"].groupby("scenario", sort=False)[[c for c in cols if "Cacheus" not in c and "S3FIFO" not in c]].mean().round(2).to_string())
    print(txt)
    with open(os.path.join(a.out, "controlled_combos_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
