"""Leave-one-source-out (LOSO) evaluation of policy selection, reported as RELATIVE MISS-RATIO
REDUCTION versus ARC and broken down by cache size (Progress Report 1 feedback).

For each trace source (MSR, CloudPhysics, Alibaba, Meta) in turn: everything that is learned or chosen
(the best fixed policy, the learned selectors and their model choice) uses only the other three sources,
and the held-out source is the test set. Every trace is tested exactly once.

Metric per test trace, on the second half of the trace:
    relative miss-ratio reduction = (miss_ratio_ARC - miss_ratio_selector) / miss_ratio_ARC * 100   [%]
(also reported as absolute hit-rate points). Cache sizes: 1%, 5%, 10% of unique objects.

Selectors at every size: always S3-FIFO, best fixed policy on the training sources, pick-from-first-half
(simulates 9 policies on the first half; nothing learned), oracle per trace.
At 5% also: learned features-only selector and the hybrid (5% miniature simulation + features), retrained
per held-out source with the same model-choice procedure as learned_selector.py / hybrid_selector.py.

  python -u scripts/loso_eval.py --out results/real
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.learned_selector import fit_predict, make_models  # noqa: E402
from scripts.policy_selection_headroom import POL, boot  # noqa: E402

SIZES = {0.01: "_f0.01", 0.05: "", 0.10: "_f0.1"}


def rel(h, pol):
    """relative miss-ratio reduction (%) of `pol` vs ARC on the second half"""
    return (h[f"h2_{pol}"] - h["h2_ARC"]) / (1 - h["h2_ARC"]) * 100


def pts(h, pol):
    return (h[f"h2_{pol}"] - h["h2_ARC"]) * 100


def choose_and_fit(X, Y, groups):
    cv = {}
    for name, make in make_models().items():
        g = []
        for t in np.unique(groups):
            m = groups == t
            P = fit_predict(make, X[~m], Y[~m], X[m])
            g += [Y[i, k] for i, k in zip(np.flatnonzero(m), P.argmax(axis=1))]
        cv[name] = float(np.mean(g))
    return max(cv, key=cv.get), cv


def learned_rows(out, halves5):
    """LOSO decisions of the features-only and hybrid selectors at 5% (returns {trace: {sel: policy}})."""
    feats = pd.read_csv(os.path.join(out, "selector_features.csv"))
    labs = pd.read_csv(os.path.join(out, "selector_window_labels.csv"))
    mini = pd.read_csv(os.path.join(out, "hybrid_mini_raw.csv"))
    fcols = [c for c in feats.columns if c not in ("trace", "split", "window", "feat_s")]
    win = mini[mini.what.str.startswith("w")].copy()
    win["window"] = win.what.str[1:].astype(int)
    mcols = []
    for p in POL:
        if p != "ARC":
            win[f"mini_{p}"] = (win[p] - win["ARC"]) * 100
            mcols.append(f"mini_{p}")
    X_all = feats[["trace", "window"] + fcols].merge(win[["trace", "window"] + mcols], on=["trace", "window"])
    X_all = X_all[X_all.trace.isin(halves5.trace)]
    X_all["source"] = X_all.trace.str.split(":").str[0]
    decisions, info = {}, []
    for name, cols in (("features_only", fcols), ("hybrid", mcols + fcols)):
        for src in sorted(X_all.source.unique()):
            tr = X_all[X_all.source != src].merge(labs.assign(window=labs.window - 1), on=["trace", "window"])
            X, groups = tr[cols].values, tr.trace.values
            Y = np.stack([(tr[p] - tr["ARC"]).values * 100 for p in POL], axis=1)
            best, cv = choose_and_fit(X, Y, groups)
            te = X_all[(X_all.source == src) & (X_all.window == 3)]
            P = fit_predict(make_models()[best], X, Y, te[cols].values)
            for t, pred in zip(te.trace, P):
                decisions.setdefault(t, {})[name] = POL[int(np.argmax(pred))]
            info.append(f"  {name:<14} held-out {src:<5} trained on {tr.trace.nunique()} traces / {len(tr)} rows, "
                        f"model {best} (CV {cv[best]:+.2f})")
    return decisions, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    rows = []
    halves5 = None
    for f, suf in SIZES.items():
        path = os.path.join(a.out, f"policy_halves_raw{suf}.csv")
        if not os.path.exists(path):
            print(f"missing {path}; skipping size {f}")
            continue
        h = pd.read_csv(path)
        h = h[h.one_hit <= 0.90].copy()
        h["source"] = h.trace.str.split(":").str[0]
        if f == 0.05:
            halves5 = h
        for src in sorted(h.source.unique()):
            trn, tst = h[h.source != src], h[h.source == src]
            best = max(POL, key=lambda p: trn[f"h2_{p}"].mean())
            for _, r in tst.iterrows():
                pick = max(POL, key=lambda p: r[f"h1_{p}"])
                orc = max(POL, key=lambda p: r[f"h2_{p}"])
                for sel, pol in (("always_S3FIFO", "S3FIFO"), ("best_fixed_on_train", best),
                                 ("pick_first_half", pick), ("oracle", orc)):
                    rows.append({"size": f, "source": src, "trace": r.trace, "selector": sel, "policy": pol,
                                 "rel_mr_reduction": rel(r, pol), "points": pts(r, pol),
                                 "arc_miss_ratio": 1 - r["h2_ARC"]})
    info = []
    if halves5 is not None:
        dec, info = learned_rows(a.out, halves5)
        h5 = halves5.set_index("trace")
        for t, d in dec.items():
            r = h5.loc[t]
            for sel, pol in d.items():
                rows.append({"size": 0.05, "source": r.source, "trace": t, "selector": sel, "policy": pol,
                             "rel_mr_reduction": rel(r, pol), "points": pts(r, pol),
                             "arc_miss_ratio": 1 - r["h2_ARC"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "loso_results.csv"), index=False)

    order = ["always_S3FIFO", "best_fixed_on_train", "features_only", "hybrid", "pick_first_half", "oracle"]
    L = ["LEAVE-ONE-SOURCE-OUT: train on three sources, test on the fourth; every trace tested once.",
         "Relative miss-ratio reduction vs ARC on the second half (%), mean [95% CI over traces], positive/n"]
    for f in sorted(df["size"].unique()):
        d = df[df["size"] == f]
        L.append(f"\ncache = {f:.0%} of unique objects ({d.trace.nunique()} traces; ARC miss ratio "
                 f"median {d.drop_duplicates('trace').arc_miss_ratio.median():.2f})")
        for sel in order:
            v = d[d.selector == sel]
            if v.empty:
                continue
            m, lo, hi = boot(v.rel_mr_reduction.values)
            pm, plo, phi = boot(v.points.values)
            L.append(f"  {sel:<20} {m:+6.2f}% [{lo:+.2f}, {hi:+.2f}]   {int((v.rel_mr_reduction > 0).sum())}/{len(v)}"
                     f"   (points {pm:+.2f} [{plo:+.2f}, {phi:+.2f}])")
        s3 = d[d.selector == "always_S3FIFO"].set_index("trace").rel_mr_reduction
        for sel in ("features_only", "hybrid", "pick_first_half"):
            v = d[d.selector == sel].set_index("trace").rel_mr_reduction
            if v.empty:
                continue
            diff = (v - s3.loc[v.index]).values
            m, lo, hi = boot(diff)
            L.append(f"  {sel} minus always_S3FIFO: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
    d5 = df[df["size"] == 0.05]
    L.append("\nper held-out source at 5% (relative miss-ratio reduction %, mean over its traces):")
    piv = d5.pivot_table(index="selector", columns="source", values="rel_mr_reduction", aggfunc="mean").reindex(order)
    L.append(piv.round(2).to_string())
    L.append("\nlearned models per held-out source:")
    L += info
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "loso_summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
