"""Randomized multi-phase synthetic 'servers' for train/test splits by whole trace.

Each server has a few recurring working sets and switches among phases of
different kinds. This is a stand-in until MSR-Cambridge / spliced MSR traces are
in; the same pipeline runs on those unchanged.
"""
import numpy as np

from shiftcache import synthetic


def random_server(seed: int, n_phases: int = 6, scale: int = 1):
    rng = np.random.default_rng(10_000 + seed)
    n_ws = int(rng.integers(2, 4))
    ws_univ = [int(rng.integers(15_000, 60_000)) * scale for _ in range(n_ws)]
    segs, prev_ws = [], None
    for i in range(n_phases):
        kind = rng.choice(["zipf", "zipf", "mix", "scan"], p=[0.35, 0.25, 0.25, 0.15])
        if i == 0:
            kind = "zipf"
        ws = int(rng.choice([w for w in range(n_ws) if w != prev_ws] or [0]))
        n = int(rng.integers(80_000, 220_000)) * scale
        if kind == "scan":
            segs.append({"kind": "scan", "n": n // 3, "universe": n // 3, "ws": 100 + i})
            prev_ws = None
            continue
        seg = {"kind": kind, "n": n, "universe": ws_univ[ws],
               "alpha": float(rng.uniform(0.6, 1.1)), "ws": ws}
        if kind == "mix":
            seg["scan_frac"] = float(rng.uniform(0.15, 0.4))
            seg["scan_universe"] = 5 * ws_univ[ws]
        segs.append(seg)
        prev_ws = ws
    return synthetic(segs, seed=seed, name=f"srv{seed}")


TRAIN_SEEDS = list(range(0, 8))
TEST_SEEDS = list(range(100, 108))
