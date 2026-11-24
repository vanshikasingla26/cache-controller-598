"""Switch-or-stay gate on top of the reactive rule, trained on all three cache sizes pooled.
Motivation (Progress Report 2): the cost-aware switcher predicts six experts' next-window gains from ~150 rows
per fold and does not transfer to a held-out source; the margin rule works at 1% and 10% but its margin is
unstable at 5%. Two changes, protocol fixed before results:
  (a) predict an easier target: at each window boundary the candidate is last window's winner among all 17
      (8 S3-FIFO settings + 9 policies); a classifier only decides whether moving to it pays off next window
      (continuous hit-rate difference minus the measured switching cost);
  (b) pool training rows from the 1%, 5% and 10% caches (log10 cache fraction and the cost are features),
      about three times the data.
Variants, all leave-one-source-out (train on the other three sources, every cache size):
  gate_pooled    logistic regression (standardized inputs, C=1), switch if P(pays) > 0.5, pooled sizes
  gate_same      same model trained on the test cache size only (isolates the pooling)
  margin_pooled  the margin rule with one margin chosen on the training sources over all three sizes
Features at a boundary: winner's lead over the current choice, current's and winner's gain over default S3-FIFO
on the last window (hit-rate points), log10 fraction, switching cost, cheap workload features of the last window.
Training rows come from the boundaries where the naive rule (margin 0, starting from default S3-FIFO) would
switch, along its own trajectory. Plans are executed exactly with carry mechanics.

  python -u scripts/gate_switcher.py --out results/real
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache.realdata import read_records  # noqa: E402
from scripts.policy_selection_headroom import boot  # noqa: E402
from scripts.s3fifo_tuning import NWIN, W, run_plan  # noqa: E402
from scripts.cost_aware_switcher import CANDS, DEFAULT, paths  # noqa: E402
from scripts.margin_switcher import MARGINS, plan_for  # noqa: E402

SIZES = (0.01, 0.05, 0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    import libcachesim as lcs
    T0 = time.time()
    feats = pd.read_csv(os.path.join(a.out, "selector_features.csv"))
    fcols = [c for c in feats.columns if c not in ("trace", "split", "window", "feat_s")]
    F = {(r.trace, int(r.window)): r[fcols].values.astype(float) for _, r in feats.iterrows()}
    D = {}
    for f in SIZES:
        fr = f"{f:g}"
        tune = pd.read_csv(os.path.join(a.out, f"s3fifo_tuning_raw_f{fr}.csv")).set_index("trace")
        sw = pd.read_csv(os.path.join(a.out, f"switching_raw_f{fr}.csv")).set_index("trace")
        cont = {t: {c: np.array([tune.loc[t, f"cont_{c}_w{w + 1}"] for w in range(NWIN)]) for c in CANDS}
                for t in tune.index}
        D[f] = {"tune": tune, "sw": sw, "cont": cont}
    traces = list(D[0.10]["tune"].index)
    src = {t: t.split(":")[0] for t in traces}
    P = paths()

    def cost_of(f, ts):
        sw = D[f]["sw"]
        cs = []
        for t in ts:
            plan = sw.loc[t, "plan"].split(">")
            cs += [sw.loc[t, f"cont_{plan[w]}_w{w + 1}"] - sw.loc[t, f"carry_w{w + 1}"]
                   for w in range(1, NWIN) if plan[w] != plan[w - 1]]
        return max(0.0, float(np.mean(cs))) * 100

    def xrow(f, t, w, cur, best, cost):
        c = D[f]["cont"][t]
        return np.concatenate([[(c[best][w - 1] - c[cur][w - 1]) * 100, (c[cur][w - 1] - c[DEFAULT][w - 1]) * 100,
                                (c[best][w - 1] - c[DEFAULT][w - 1]) * 100, np.log10(f), cost], F[(t, w)]])

    def train_rows(fs, ts, costs):
        X, y = [], []
        for f in fs:
            for t in ts:
                c = D[f]["cont"][t]
                cur = DEFAULT
                for w in range(1, NWIN):
                    best = max(CANDS, key=lambda k: c[k][w - 1])
                    if best == cur:
                        continue
                    X.append(xrow(f, t, w, cur, best, costs[f]))
                    y.append(int((c[best][w] - c[cur][w]) * 100 - costs[f] > 0))
                    cur = best
        return np.array(X), np.array(y)

    def gate_plan(f, t, model, cost):
        c = D[f]["cont"][t]
        plan = [DEFAULT]
        for w in range(1, NWIN):
            cur = plan[-1]
            best = max(CANDS, key=lambda k: c[k][w - 1])
            if best != cur and model.predict_proba(xrow(f, t, w, cur, best, cost)[None, :])[0, 1] > 0.5:
                plan.append(best)
            else:
                plan.append(cur)
        return plan

    rawp = os.path.join(a.out, "gate_switcher_raw.csv")
    rows = pd.read_csv(rawp).to_dict("records") if os.path.exists(rawp) else []
    done = {(r["trace"], r["size"]) for r in rows}
    info = []
    memo = {}
    for held in sorted(set(src.values())):
        train = [t for t in traces if src[t] != held]
        test = [t for t in traces if src[t] == held]
        costs = {f: cost_of(f, train) for f in SIZES}
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))  # noqa: E731
        Xp, yp = train_rows(SIZES, train, costs)
        pooled = mk().fit(Xp, yp)
        same = {}
        for f in SIZES:
            Xs, ys = train_rows((f,), train, costs)
            same[f] = mk().fit(Xs, ys) if len(set(ys)) > 1 else None
        sc = {m: np.mean([np.mean([D[f]["cont"][t][p[w]][w] - costs[f] / 100 * (p[w] != p[w - 1])
                                   for w in range(1, NWIN)])
                          for f in SIZES for t in train for p in [plan_for(D[f]["cont"][t], m)]]) for m in MARGINS}
        m_pool = max(MARGINS, key=lambda m: (sc[m], m))
        info.append(f"held-out {held}: pooled rows {len(yp)} ({yp.mean():.0%} pay), margin_pooled {m_pool:g} pts, "
                    f"costs " + ", ".join(f"{f:.0%} {costs[f]:.2f}" for f in SIZES))
        for f in SIZES:
            for t in test:
                if (t, f) in done:
                    continue
                cont = D[f]["cont"][t]
                cap = int(D[f]["tune"].loc[t, "capacity"])
                plans = {"gate_pooled": gate_plan(f, t, pooled, costs[f]),
                         "gate_same": gate_plan(f, t, same[f], costs[f]) if same[f] is not None else [DEFAULT] * NWIN,
                         "margin_pooled": plan_for(cont, m_pool)}
                r = {"trace": t, "source": held, "size": f, "margin_pooled": m_pool}
                for k, p in plans.items():
                    key = (t, f, ">".join(p))
                    if key not in memo:
                        if all(x == DEFAULT for x in p):
                            memo[key] = cont[DEFAULT]
                        else:
                            ids = read_records(P[t], W * NWIN)["id"].tolist()
                            memo[key] = run_plan(lcs, ids, cap, p)
                    r.update({f"{k}_w{w + 1}": memo[key][w] for w in range(NWIN)})
                    r[f"{k}_switches"] = sum(p[w] != p[w - 1] for w in range(1, NWIN))
                rows.append(r)
                pd.DataFrame(rows).to_csv(rawp, index=False)
                print(f"  {f:.0%} {t}: switches gate {r['gate_pooled_switches']}, same {r['gate_same_switches']}, "
                      f"margin {r['margin_pooled_switches']} ({(time.time() - T0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    L = ["SWITCH-OR-STAY GATE, leave-one-source-out, windows 2-6; difference in relative miss-ratio reduction vs ARC "
         "from default S3-FIFO (%), mean [95% CI over traces]"]
    out = []
    for f in SIZES:
        g = df[df["size"] == f]
        for _, r in g.iterrows():
            cont = D[f]["cont"][r.trace]
            m_arc = np.mean([1 - cont["ARC"][w] for w in range(1, NWIN)])
            red = lambda hr: (m_arc - np.mean([1 - h for h in hr])) / m_arc * 100  # noqa: E731
            base = red([cont[DEFAULT][w] for w in range(1, NWIN)])
            o = {"trace": r.trace, "size": f}
            for k in ("gate_pooled", "gate_same", "margin_pooled"):
                o[k] = red([r[f"{k}_w{w + 1}"] for w in range(1, NWIN)]) - base
                o[f"{k}_switches"] = r[f"{k}_switches"]
            out.append(o)
        s = pd.DataFrame(out)
        s = s[s["size"] == f]
        L.append(f"  cache {f:.0%} ({len(s)} traces)")
        for k in ("gate_pooled", "gate_same", "margin_pooled"):
            m, lo, hi = boot(s[k].values)
            L.append(f"    {k:<14} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]   better on {int((s[k] > 0).sum())}, worse on "
                     f"{int((s[k] < 0).sum())}, switches {s[f'{k}_switches'].mean():.2f}")
    pd.DataFrame(out).to_csv(os.path.join(a.out, "gate_switcher_summary.csv"), index=False)
    L += info
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "gate_switcher_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
