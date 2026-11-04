"""Cost-aware learned switcher (cache 10%, same traces/windows/mechanics as s3fifo_tuning.py).
Protocol fixed before results; evaluated leave-one-source-out (LOSO).

Per held-out source, using only the other three sources:
  experts   the K candidates (S3-FIFO settings and policies) that most often have the best hit rate in a
            window on the training traces; the default S3-FIFO setting is always included (Darwin-style
            small expert set).
  model     one regressor per expert predicts its next-window hit-rate gain over default S3-FIFO from
            [this window's gain of every expert over default] + [cheap workload features of this window];
            model type (ridge / shallow GBR) chosen by leave-one-trace-out CV on the training traces.
  cost      the measured switching cost (carry mechanics, scripts/switching.py), averaged over training
            traces only.
Decision at every window boundary, starting from default S3-FIFO: move from the current expert c to
e* = argmax_e [pred_e - pred_c - cost * (e != c)]; stay if no expert clears the cost.
Variant "no cost": the same model with cost = 0, to isolate what the cost rule does.
Plans are executed with real switching (carry). The experts' hit rates on the window that just ended are
assumed known (shadow simulation of K experts, as the naive rule also assumes).

  python -u scripts/cost_aware_switcher.py --out results/real
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402
from scripts.learned_selector import make_models  # noqa: E402
from scripts.policy_selection_headroom import DATASETS, POL, boot, name_of  # noqa: E402
from scripts.s3fifo_tuning import NWIN, SETTINGS, W, run_plan  # noqa: E402

DEFAULT = "S3FIFO(s=0.1,t=2)"
CANDS = list(SETTINGS) + POL
K = 6


def fit_k(make, Xtr, Ytr, Xte):
    """one regressor per expert column; column 0 is the default, whose gain over itself is 0"""
    out = np.zeros((len(Xte), Ytr.shape[1]))
    for i in range(1, Ytr.shape[1]):
        out[:, i] = make().fit(Xtr, Ytr[:, i]).predict(Xte)
    return out


def paths():
    import glob
    out = {}
    for tag, pat in DATASETS.items():
        for p in sorted(glob.glob(pat)):
            out[name_of(p, tag)] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    import libcachesim as lcs
    T0 = time.time()
    tune = pd.read_csv(os.path.join(a.out, "s3fifo_tuning_raw_f0.1.csv")).set_index("trace")
    sw = pd.read_csv(os.path.join(a.out, "switching_raw_f0.1.csv")).set_index("trace")
    feats = pd.read_csv(os.path.join(a.out, "selector_features.csv"))
    fcols = [c for c in feats.columns if c not in ("trace", "split", "window", "feat_s")]
    F = {(r.trace, int(r.window)): r[fcols].values.astype(float) for _, r in feats.iterrows()}
    cont = {t: {c: np.array([tune.loc[t, f"cont_{c}_w{w + 1}"] for w in range(NWIN)]) for c in CANDS}
            for t in tune.index}
    src = {t: t.split(":")[0] for t in tune.index}
    P = paths()

    def switch_costs(traces):
        cs = []
        for t in traces:
            plan = sw.loc[t, "plan"].split(">")
            for w in range(1, NWIN):
                if plan[w] != plan[w - 1]:
                    cs.append(sw.loc[t, f"cont_{plan[w]}_w{w + 1}"] - sw.loc[t, f"carry_w{w + 1}"])
        return float(np.mean(cs))

    rawp = os.path.join(a.out, "cost_aware_raw.csv")
    rows = pd.read_csv(rawp).to_dict("records") if os.path.exists(rawp) else []
    done = {r["trace"] for r in rows}
    info = []
    for held in sorted(set(src.values())):
        train = [t for t in tune.index if src[t] != held]
        test = [t for t in tune.index if src[t] == held]
        wins = pd.Series([max(CANDS, key=lambda c: cont[t][c][w]) for t in train for w in range(NWIN)])
        experts = [DEFAULT] + [c for c in wins.value_counts().index if c != DEFAULT][:K - 1]
        cost = switch_costs(train)

        def xrow(t, w):  # features at window index w (0-based)
            g = [cont[t][e][w] - cont[t][DEFAULT][w] for e in experts]
            return np.concatenate([np.array(g) * 100, F[(t, w + 1)]])

        X, Y, groups = [], [], []
        for t in train:
            for w in range(NWIN - 1):
                X.append(xrow(t, w))
                Y.append([(cont[t][e][w + 1] - cont[t][DEFAULT][w + 1]) * 100 for e in experts])
                groups.append(t)
        X, Y, groups = np.array(X), np.array(Y), np.array(groups)
        cv = {}
        for name, mk in make_models().items():
            g = []
            for t in np.unique(groups):
                m = groups == t
                Pm = fit_k(mk, X[~m], Y[~m], X[m])
                g += [Y[i, k] for i, k in zip(np.flatnonzero(m), Pm.argmax(axis=1))]
            cv[name] = float(np.mean(g))
        best = max(cv, key=cv.get)
        info.append(f"held-out {held}: experts {experts}; switch cost on train {cost * 100:.2f} pts; "
                    f"model {best} (CV {cv[best]:+.2f})")
        for t in test:
            if t in done:
                continue
            Xt = np.array([xrow(t, w) for w in range(NWIN - 1)])
            pred = fit_k(make_models()[best], X, Y, Xt)          # rows: decision after window w
            r = {"trace": t, "source": held}
            for variant, c_pts in (("cost_aware", cost * 100), ("no_cost", 0.0)):
                plan = [DEFAULT]
                for w in range(1, NWIN):
                    cur = experts.index(plan[-1])
                    score = pred[w - 1] - pred[w - 1][cur] - c_pts * (np.arange(len(experts)) != cur)
                    plan.append(experts[int(np.argmax(score))] if score.max() > 0 else plan[-1])
                if all(p == DEFAULT for p in plan):
                    hr = cont[t][DEFAULT]
                else:
                    rec = read_records(P[t], W * NWIN)
                    hr = run_plan(lcs, rec["id"].tolist(), int(tune.loc[t, "capacity"]), plan)
                r[f"{variant}_plan"] = ">".join(plan)
                r[f"{variant}_switches"] = sum(plan[w] != plan[w - 1] for w in range(1, NWIN))
                r.update({f"{variant}_w{w + 1}": hr[w] for w in range(NWIN)})
            rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "cost_aware_raw.csv"), index=False)
            print(f"  {t}: cost-aware {r['cost_aware_switches']} switches, no-cost {r['no_cost_switches']} "
                  f"({(time.time() - T0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    ws = range(2, NWIN + 1)
    out = []
    for _, r in df.iterrows():
        t = r.trace
        m_arc = np.mean([1 - cont[t]["ARC"][w - 1] for w in ws])
        red = lambda miss: (m_arc - miss) / m_arc * 100  # noqa: E731
        out.append({"trace": t, "source": r.source,
                    "S3FIFO_default": red(np.mean([1 - cont[t][DEFAULT][w - 1] for w in ws])),
                    "naive_hybrid": red(np.mean([1 - tune.loc[t, f"hybrid_w{w}"] for w in ws])),
                    "learned_no_cost": red(np.mean([1 - r[f"no_cost_w{w}"] for w in ws])),
                    "learned_cost_aware": red(np.mean([1 - r[f"cost_aware_w{w}"] for w in ws])),
                    "ceiling_all": red(np.mean([1 - max(cont[t][c][w - 1] for c in CANDS) for w in ws])),
                    "switches_cost_aware": r.cost_aware_switches, "switches_no_cost": r.no_cost_switches})
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, "cost_aware_summary.csv"), index=False)
    L = ["COST-AWARE LEARNED SWITCHER, leave-one-source-out, cache 10%, windows 2-6; relative miss-ratio "
         f"reduction vs ARC (%), mean [95% CI over {len(s)} traces], positive/n"]
    for c in ("S3FIFO_default", "naive_hybrid", "learned_no_cost", "learned_cost_aware", "ceiling_all"):
        m, lo, hi = boot(s[c].values)
        L.append(f"  {c:<20} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   {int((s[c] > 0).sum())}/{len(s)}")
    for c in ("naive_hybrid", "learned_no_cost", "learned_cost_aware"):
        m, lo, hi = boot((s[c] - s.S3FIFO_default).values)
        L.append(f"  {c} minus S3FIFO_default: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   "
                 f"better on {int(((s[c] - s.S3FIFO_default) > 0).sum())}/{len(s)}")
    L.append(f"  switches per trace: cost-aware {s.switches_cost_aware.mean():.1f}, "
             f"no-cost {s.switches_no_cost.mean():.1f}")
    L.append("  per held-out source, learned_cost_aware minus S3FIFO_default: " + ", ".join(
        f"{k} {(g.learned_cost_aware - g.S3FIFO_default).mean():+.2f}%" for k, g in s.groupby("source")))
    L += info
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "cost_aware_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
