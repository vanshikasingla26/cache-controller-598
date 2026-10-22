"""RQ1 quick check: detector vs ARC's own adaptation, on the demo trace and a few
synthetic servers. Prints detection precision/recall/delay and ARC's p half-swing
time per shift, and saves an overlap + p plot.

  python scripts/detect_eval.py
  python scripts/detect_eval.py --ref-windows 1      # the proposal's literal rule
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate  # noqa: E402
from shiftcache.detector import (DetectorConfig, arc_adaptation_delay,  # noqa: E402
                                 run_detector, score_detection)
from shiftcache.policies import ARC  # noqa: E402
from scripts.demo_traces import demo_trace  # noqa: E402
from scripts.workloads import TRAIN_SEEDS, random_server  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-windows", type=int, default=10)
    ap.add_argument("--start-thresh", type=float, default=0.6)
    ap.add_argument("--persist", type=int, default=3)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--tol", type=int, default=20_000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = DetectorConfig(ref_windows=a.ref_windows, start_thresh=a.start_thresh, persist=a.persist)
    os.makedirs(a.out, exist_ok=True)
    print(cfg)

    for tr in [demo_trace()] + [random_server(s) for s in TRAIN_SEEDS[:3]]:
        cap = max(1, int(tr.footprint * a.fraction))
        mask, d = run_detector(tr.keys, cfg)
        s = score_detection(d.starts, tr.transitions, a.tol)
        r = simulate(tr, ARC(cap))
        print(f"\n{tr.name}: {len(tr):,} accesses, shifts at {tr.transitions}")
        print(f"  detector starts {d.starts}")
        print(f"  precision {s['precision']:.2f}  recall {s['recall']:.2f}  delays {s['delays']}  "
              f"flagged {mask.mean():.1%} of accesses")
        for x in arc_adaptation_delay(r.p_trace, 1000, tr.transitions, len(tr), cap):
            print(f"  shift {x['shift']:>8,}: ARC p peak swing {x['peak_frac']:.2f}c, "
                  f"half-swing after {x['t_half']}, peak after {x['t_peak']}")
        plot(tr, d, r, cap, cfg, a.out)


def plot(tr, d, r, cap, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ov = np.array(d.overlaps)
    ax.plot(np.arange(1, len(ov) + 1) * cfg.window, ov, lw=0.8)
    ax.axhline(cfg.start_thresh, color="r", ls=":", lw=0.8)
    ax.set_ylabel("overlap vs ref")
    for s, e in zip(d.starts, d.ends + [len(tr)] * (len(d.starts) - len(d.ends))):
        ax.axvspan(s, e, color="orange", alpha=0.2)
        ax2.axvspan(s, e, color="orange", alpha=0.2)
    ax2.plot(np.arange(len(r.p_trace)) * r.window, r.p_trace / cap, color="C2")
    ax2.set_ylabel("ARC p / c")
    for t in tr.transitions:
        ax.axvline(t, color="k", ls="--", lw=0.8)
        ax2.axvline(t, color="k", ls="--", lw=0.8)
    ax.set_title(f"{tr.name}: detector (orange = flagged) vs ARC p; dashed = true shift")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"detector_{tr.name}_ref{cfg.ref_windows}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
