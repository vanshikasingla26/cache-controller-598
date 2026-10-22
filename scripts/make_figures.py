"""Figures from the real MSR results (no new simulation; reads results/real/*.csv).

  python scripts/make_figures.py --out results/figures
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.summarize_real import cluster_ci  # noqa: E402

R = "results/real"


def fig_baselines(out):
    df = pd.read_csv(f"{R}/baselines_msr.csv")
    d = df[df.fraction == 0.05]
    pv = d.pivot(index="trace", columns="policy", values="hit_rate")
    pv = pv.sort_values("ARC", ascending=False)
    pols = ["LRU", "ARC", "Cacheus", "S3FIFO", "WTinyLFU"]
    x = np.arange(len(pv))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, p in enumerate(pols):
        ax.bar(x + (i - 2) * w, pv[p], w, label=p)
    ax.scatter(x, pv["MIN"], marker="_", s=300, color="k", label="MIN (clairvoyant)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(pv.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("hit rate")
    ax.set_title("Real MSR-Cambridge volumes, first 3M requests, cache = 5% of unique objects", fontsize=9)
    ax.legend(fontsize=7, ncol=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_baselines_msr.png", dpi=160)
    plt.close(fig)


def fig_splices(out):
    s = pd.read_csv(f"{R}/real_splice_raw.csv")
    arc = s[s.policy == "ARC"].set_index("pair")
    for c in ("transition_hr", "overall_hr"):
        s["d_" + c] = s.apply(lambda r: (r[c] - arc.loc[r.pair, c]) * 100, axis=1)
    show = [("LRU-gated[gt]", "known shift,\nno model"), ("gbt[gt]", "known shift,\nlearned"),
            ("LRU-gated[det]", "detector,\nno model"), ("gbt[det]", "detector,\nlearned"),
            ("gbt[random]", "random gate,\nlearned")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, col, title in [(axes[0], "d_transition_hr", "first 50k accesses after the shift"),
                           (axes[1], "d_overall_hr", "whole trace")]:
        ms, los, his = [], [], []
        for p, _ in show:
            m, lo, hi, _ = cluster_ci(s[s.policy == p], col, "dst")
            ms.append(m)
            los.append(m - lo)
            his.append(hi - m)
        colors = ["tab:green" if v > 0 else "tab:red" for v in ms]
        ax.bar(range(len(show)), ms, yerr=[los, his], color=colors, capsize=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels([lab for _, lab in show], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("hit rate vs ARC (points)")
    fig.suptitle(f"Real spliced pairs ({s.pair.nunique()} pairs, {s.dst.nunique()} held-out destination volumes), "
                 "95% CI over volumes", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_real_splices.png", dpi=160)
    plt.close(fig)


def fig_gate(out):
    g = pd.read_csv(f"{R}/gate_tuning_test.csv")
    g["dst"] = np.where(g.kind == "splice", g.trace.str.split(">").str[1], g.trace)
    labels, ms, los, his = [], [], [], []
    for gate in ("F1-tuned fixed (previous gate)", "hit-rate-tuned"):
        for kd in ("splice", "natural"):
            m, lo, hi, _ = cluster_ci(g[(g.gate == gate) & (g.kind == kd)], "real", "dst")
            labels.append(f"{'F1-tuned' if gate.startswith('F1') else 'hit-rate-tuned'}\n{kd}")
            ms.append(m)
            los.append(m - lo)
            his.append(hi - m)
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(range(4), ms, yerr=[los, his], color="tab:red", capsize=4, label="real, LRU-gated ARC")
    ax.axhline(0.74, color="tab:green", ls="--", lw=1, label="ceiling: perfect per-window gate (+0.7)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("overall hit rate vs ARC (points)")
    ax.set_title("Gate tuned for detection vs for hit rate (held-out MSR volumes)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_gate_tuning.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_baselines(a.out)
    fig_splices(a.out)
    fig_gate(a.out)
    print("figures written to", a.out)


def fig_controlled(out):
    d = pd.read_csv(f"{R}/controlled_shifts_raw.csv")
    t = d[d.group == "test"].copy()
    t["scenario"] = t.dim + " " + t.mag.astype(str)
    t["avoid_pts"] = t.avoidable_misses / 50_000 * 100
    order = list(dict.fromkeys(t.scenario))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, col, title in [(axes[0], "dT_LRU-gated[gt]", "LRU-like action with the TRUE change point\n(first 50k accesses after the shift, vs ARC)"),
                           (axes[1], "avoid_pts", "extra MIN - ARC gap right after the shift\n(transition-specific room; <=0 means none)")]:
        ms, los, his = [], [], []
        for sc in order:
            m, lo, hi, _ = cluster_ci(t[t.scenario == sc], col, "vol")
            ms.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.bar(range(len(order)), ms, yerr=[los, his], capsize=3,
               color=["tab:green" if v > 0 else "tab:red" for v in ms])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("points")
        ax.set_title(title, fontsize=8)
    fig.suptitle("Controlled one-dimension shifts injected into 4 held-out real MSR volumes (95% CI over volumes)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/fig4_controlled_shifts.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    fig_controlled("results/figures")


def fig_policy_selection(out):
    sel = pd.read_csv(f"{R}/selector_test_results.csv")          # both folds, every trace tested once
    vals = {"always S3-FIFO\n(best on train)": sel.always_S3FIFO, "learned cheap\nselector": sel.learned,
            "pick best from first\nhalf (simulation)": sel.pick_first_half, "oracle per trace\n(ceiling)": sel.oracle}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for i, (lab, v) in enumerate(vals.items()):
        v = v.values
        m = v[rng.integers(0, len(v), size=(10_000, len(v)))].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        ax.bar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], capsize=4,
               color="tab:green" if lo > 0 else "tab:gray")
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(list(vals), fontsize=8)
    ax.set_ylabel("hit rate vs ARC, second half (points)")
    ax.set_title(f"Policy selection, {len(sel)} real traces (MSR, CloudPhysics, Alibaba, Meta), two folds\n"
                 "bars: mean with 95% CI over traces (green = CI above zero); dots: individual traces", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/fig5_policy_selection.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_policy_selection("results/figures")
