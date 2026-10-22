"""Run LRU / LFU / ARC / Belady across cache sizes.

  # real trace (download from SNIA IOTTA: MSR Cambridge)
  python scripts/run_baselines.py --msr data/web_0.csv.gz --fractions 0.01 0.05 0.1

  # synthetic spliced trace (no download needed)
  python scripts/run_baselines.py --synthetic

Writes results/baselines.csv and a per-window hit-rate plot per cache size.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import load_msr, simulate  # noqa: E402
from shiftcache.policies import ARC, LFU, LRU, Belady, next_use  # noqa: E402
from scripts.demo_traces import demo_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msr", help="MSR-Cambridge CSV(.gz)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--reads-only", action="store_true")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    tr = load_msr(a.msr, max_requests=a.max_requests) if a.msr else demo_trace()
    if a.reads_only:
        tr = tr.reads_only()
    fp = tr.footprint
    print(f"{tr.name}: {len(tr):,} block accesses, {tr.n_requests:,} requests, footprint {fp:,} blocks")
    nu = next_use(tr.keys)
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for f in a.fractions:
        cap = max(1, int(fp * f))
        results = [simulate(tr, P, window=a.window) for P in
                   (LRU(cap), LFU(cap), ARC(cap), Belady(cap, nu))]
        for r in results:
            s = r.summary()
            s.update(fraction=f, request_hit_rate=round(r.request_hit_rate(tr.req_ids), 5))
            rows.append(s)
            print(f"  f={f:<5} cap={cap:<8} {r.policy:<7} hr={r.hit_rate:.4f}  ({r.wall_s:.1f}s)")
        plot_windows(results, tr, f, a.out)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "baselines.csv"), index=False)
    print(f"wrote {a.out}/baselines.csv")


def plot_windows(results, tr, frac, out, smooth=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        w = pd.Series(r.window_hit_rates()).rolling(smooth, min_periods=1).mean()
        ax.plot(w.index * r.window, w.values, label=r.policy, lw=1.2)
        if r.p_trace is not None:
            ax2.plot(pd.Series(range(len(r.p_trace))) * r.window, r.p_trace / r.capacity, color="C2")
    for t in tr.transitions:
        ax.axvline(t, color="k", ls="--", lw=0.8)
        ax2.axvline(t, color="k", ls="--", lw=0.8)
    ax.set_ylabel("hit rate (rolling)")
    ax.set_title(f"{tr.name}, cache = {frac:.0%} of footprint")
    ax.legend(ncol=4, fontsize=8)
    ax2.set_ylabel("ARC p / c")
    ax2.set_xlabel("block access")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"windows_{tr.name}_f{frac}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
