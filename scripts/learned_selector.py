"""Learned, cheap cache-policy selector (step 4). Protocol fixed before any result.

Question: can a model predict which policy will win from CHEAP statistics of recent requests, instead
of simulating all candidate policies? Evaluation reuses step 3 exactly (same 33 traces, same train/test
split, same second-half hit rates from results/real/policy_halves_raw.csv).

Features (one pass over a window of W=500k requests, no cache simulation): unique ratio, one-hit share,
share of accesses to the top 1% / 10% objects, sequential share, first-ever-access share, mean accesses
per object, and the normalised histogram of inter-reference gaps (log2 bins, as in the detector).

Training (TRAIN traces only): each trace's first 3M requests = 6 windows. Rows = (features of window j,
per-policy hit rate minus ARC's on window j+1), j = 1..5. Window hit rates come from libCacheSim on each
window alone (cold start), the same convention as step 3's halves. Candidate models: ridge regression
and shallow gradient boosting, one regressor per policy; chosen by leave-one-trace-out CV on TRAIN by
realised gain of the argmax choice.

Test (12 held-out traces): features of window 3 (the last 500k requests of the first half) -> predicted
gain per policy -> pick argmax (ARC = 0). Realised gain = second-half hit rate of the chosen policy minus
ARC's, from step 3. Compared with: always-S3FIFO, pick-from-first-half (simulates 9 policies on 1.5M),
oracle per trace. Cost: wall time of feature extraction on 500k vs the 9 simulations on 1.5M.

  python -u scripts/learned_selector.py --out results/real
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
from scripts.policy_selection_headroom import DATASETS, POL, boot, name_of, native, split  # noqa: E402

W, NWIN, NB = 500_000, 6, 18


def window_features(keys, seen, t0):
    """keys: int array of one window; seen: dict key->last index (trace history, updated in place)."""
    n = len(keys)
    u, cnt = np.unique(keys, return_counts=True)
    srt = np.sort(cnt)[::-1]
    k1, k10 = max(1, len(u) // 100), max(1, len(u) // 10)
    hist = np.zeros(NB + 1)
    first = 0
    for i, k in enumerate(keys.tolist()):
        prev = seen.get(k)
        if prev is None:
            first += 1
            hist[0] += 1
        else:
            hist[1 + min(NB - 1, int(t0 + i - prev).bit_length() - 1)] += 1
        seen[k] = t0 + i
    f = {"unique_ratio": len(u) / n, "one_hit": float(np.mean(cnt == 1)),
         "top1_share": srt[:k1].sum() / n, "top10_share": srt[:k10].sum() / n,
         "seq_share": float(np.mean(np.diff(keys) == 1)), "first_access": first / n,
         "acc_per_obj": n / len(u)}
    for b in range(NB + 1):
        f[f"gap{b}"] = hist[b] / n
    return f


def collect(a, lcs):
    path_feat = os.path.join(a.out, "selector_features.csv")
    path_lab = os.path.join(a.out, "selector_window_labels.csv")
    path_cost = os.path.join(a.out, "selector_cost.csv")
    feats = pd.read_csv(path_feat).to_dict("records") if os.path.exists(path_feat) else []
    labs = pd.read_csv(path_lab).to_dict("records") if os.path.exists(path_lab) else []
    cost = pd.read_csv(path_cost).to_dict("records") if os.path.exists(path_cost) else []
    done = {r["trace"] for r in feats}
    halves = split(pd.read_csv(os.path.join(a.out, "policy_halves_raw.csv")))
    halves = halves[halves.one_hit <= 0.90].set_index("trace")
    T0 = time.time()
    for tag, pat in DATASETS.items():
        for path in sorted(glob.glob(pat)):
            name = name_of(path, tag)
            if name not in halves.index or name in done:
                continue
            sp, cap = halves.loc[name, "split"], int(halves.loc[name, "capacity"])
            rec = read_records(path, W * NWIN)
            keys = rec["id"].astype(np.int64)
            seen = {}
            for j in range(NWIN):
                wk = keys[j * W:(j + 1) * W]
                t = time.perf_counter()
                f = window_features(wk, seen, j * W)
                ft = time.perf_counter() - t
                feats.append({"trace": name, "split": sp, "window": j + 1, "feat_s": ft, **f})
                if True:   # labels for every trace: each trace is TRAIN in one of the two folds
                    fn = os.path.join(a.tmp, "win.bin")
                    rec[j * W:(j + 1) * W].tofile(fn)
                    labs.append({"trace": name, "window": j + 1,
                                 **{p: native(lcs, fn, p, cap) for p in POL}})
                    os.remove(fn)
            if True:   # cost of the simulation-based selector: 9 policies on the first half
                fn = os.path.join(a.tmp, "h1.bin")
                rec[: len(rec) // 2].tofile(fn)
                t = time.perf_counter()
                for p in POL:
                    native(lcs, fn, p, cap)
                cost.append({"trace": name, "sim9_first_half_s": time.perf_counter() - t,
                             "features_window3_s": [r for r in feats if r["trace"] == name and r["window"] == 3][0]["feat_s"]})
                os.remove(fn)
            pd.DataFrame(feats).to_csv(path_feat, index=False)
            pd.DataFrame(labs).to_csv(path_lab, index=False)
            pd.DataFrame(cost).to_csv(path_cost, index=False)
            print(f"  {name} ({sp}) done ({(time.time() - T0) / 60:.1f} min)", flush=True)
    # backfill labels/cost for traces featurised in an earlier run as TEST only
    labelled, costed = {r["trace"] for r in labs}, {r["trace"] for r in cost}
    for tag, pat in DATASETS.items():
        for path in sorted(glob.glob(pat)):
            name = name_of(path, tag)
            if name not in halves.index or (name in labelled and name in costed):
                continue
            cap = int(halves.loc[name, "capacity"])
            rec = read_records(path, W * NWIN)
            if name not in labelled:
                for j in range(NWIN):
                    fn = os.path.join(a.tmp, "win.bin")
                    rec[j * W:(j + 1) * W].tofile(fn)
                    labs.append({"trace": name, "window": j + 1, **{p: native(lcs, fn, p, cap) for p in POL}})
                    os.remove(fn)
            if name not in costed:
                fn = os.path.join(a.tmp, "h1.bin")
                rec[: len(rec) // 2].tofile(fn)
                t = time.perf_counter()
                for p in POL:
                    native(lcs, fn, p, cap)
                cost.append({"trace": name, "sim9_first_half_s": time.perf_counter() - t,
                             "features_window3_s": [r for r in feats if r["trace"] == name and r["window"] == 3][0]["feat_s"]})
                os.remove(fn)
            pd.DataFrame(labs).to_csv(path_lab, index=False)
            pd.DataFrame(cost).to_csv(path_cost, index=False)
            print(f"  backfilled {name} ({(time.time() - T0) / 60:.1f} min)", flush=True)
    return pd.DataFrame(feats), pd.DataFrame(labs), pd.DataFrame(cost), halves


def make_models():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "gbr": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                     subsample=0.8, random_state=0)}


def fit_predict(make, Xtr, Ytr, Xte):
    """One regressor per non-ARC policy; returns predicted gain matrix (ARC column = 0)."""
    out = np.zeros((len(Xte), len(POL)))
    for i, p in enumerate(POL):
        if p == "ARC":
            continue
        m = make().fit(Xtr, Ytr[:, i])
        out[:, i] = m.predict(Xte)
    return out


def run_fold(fold, feats, labs, halves_raw, fcols):
    """Train on this fold's TRAIN traces, evaluate on its TEST traces. Returns (rows, info)."""
    hv = split(halves_raw[halves_raw.one_hit <= 0.90], fold).set_index("trace")
    sp = hv["split"]
    f = feats.assign(split=feats.trace.map(sp))
    tr = f[f.split == "train"].merge(labs.assign(window=labs.window - 1), on=["trace", "window"])
    X = tr[fcols].values
    Y = np.stack([(tr[p] - tr["ARC"]).values * 100 for p in POL], axis=1)
    groups = tr.trace.values
    cv = {}
    for name, make in make_models().items():
        g = []
        for t in np.unique(groups):
            m = groups == t
            P = fit_predict(make, X[~m], Y[~m], X[m])
            g += [Y[i, k] for i, k in zip(np.flatnonzero(m), P.argmax(axis=1))]
        cv[name] = float(np.mean(g))
    best = max(cv, key=cv.get)
    t = time.perf_counter()
    te = f[(f.split == "test") & (f.window == 3)].set_index("trace")
    fit_s = 0.0
    models = {}
    for i, p in enumerate(POL):
        if p == "ARC":
            continue
        t0 = time.perf_counter()
        models[p] = make_models()[best]().fit(X, Y[:, i])
        fit_s += time.perf_counter() - t0
    t0 = time.perf_counter()
    P = np.zeros((len(te), len(POL)))
    for i, p in enumerate(POL):
        if p != "ARC":
            P[:, i] = models[p].predict(te[fcols].values)
    pred_s = (time.perf_counter() - t0) / max(1, len(te))
    rows = []
    for (trace, _), pred in zip(te.iterrows(), P):
        h = hv.loc[trace]
        choice, pick = POL[int(np.argmax(pred))], max(POL, key=lambda p: h[f"h1_{p}"])
        rows.append({"fold": fold, "trace": trace, "dataset": trace.split(":")[0], "learned_choice": choice,
                     "first_half_choice": pick, "learned": (h[f"h2_{choice}"] - h["h2_ARC"]) * 100,
                     "always_S3FIFO": (h["h2_S3FIFO"] - h["h2_ARC"]) * 100,
                     "pick_first_half": (h[f"h2_{pick}"] - h["h2_ARC"]) * 100,
                     "oracle": (max(h[f"h2_{p}"] for p in POL) - h["h2_ARC"]) * 100})
    info = {"fold": fold, "train_rows": len(tr), "train_traces": tr.trace.nunique(), "cv": cv, "model": best,
            "always_S3FIFO_cv": float(Y[:, POL.index("S3FIFO")].mean()), "fit_s": fit_s, "predict_s": pred_s}
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--tmp", default="/tmp/selector")
    a = ap.parse_args()
    import libcachesim as lcs
    os.makedirs(a.tmp, exist_ok=True)
    feats, labs, cost, _ = collect(a, lcs)
    halves_raw = pd.read_csv(os.path.join(a.out, "policy_halves_raw.csv"))
    fcols = [c for c in feats.columns if c not in ("trace", "split", "window", "feat_s")]
    rows, infos = [], []
    for fold in (0, 1):
        r, info = run_fold(fold, feats, labs, halves_raw, fcols)
        rows += r
        infos.append(info)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "selector_test_results.csv"), index=False)
    L = []
    for info in infos:
        L.append(f"fold {info['fold']}: {info['train_rows']} training rows from {info['train_traces']} traces; "
                 f"leave-one-trace-out CV on TRAIN (next-window gain vs ARC): "
                 + ", ".join(f"{k} {v:+.2f}" for k, v in info["cv"].items())
                 + f", always-S3FIFO {info['always_S3FIFO_cv']:+.2f}; chosen {info['model']}")
    for name, sub in [("fold 0 (original split)", res[res.fold == 0]), ("2-FOLD, all traces", res)]:
        L.append(f"\n{name}: {len(sub)} test traces, second-half hit rate vs ARC (points), mean [95% CI], positive")
        for c in ("learned", "always_S3FIFO", "pick_first_half", "oracle"):
            m, lo, hi = boot(sub[c].values)
            L.append(f"  {c:<16} {m:+.2f} [{lo:+.2f},{hi:+.2f}]   {int((sub[c] > 0).sum())}/{len(sub)}")
        m, lo, hi = boot((sub.learned - sub.always_S3FIFO).values)
        L.append(f"  learned minus always_S3FIFO: {m:+.2f} [{lo:+.2f},{hi:+.2f}]; same choice as first-half "
                 f"simulation on {int((sub.learned_choice == sub.first_half_choice).sum())}/{len(sub)}")
    L.append("\nhidden costs (learned selector):")
    L.append(f"  training time (8 regressors, all TRAIN rows): " + ", ".join(f"fold {i['fold']} {i['fit_s']:.2f} s" for i in infos))
    L.append(f"  inference per decision: {np.mean([i['predict_s'] for i in infos]) * 1e3:.2f} ms")
    L.append(f"  feature extraction per decision (500k requests, Python): {cost.features_window3_s.mean():.2f} s; "
             f"simulation-based selector (9 native policies on 1.5M requests): {cost.sim9_first_half_s.mean():.2f} s")
    L.append("  feature state: one last-seen index per object seen so far (dictionary); see selector_state.csv")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "selector_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
