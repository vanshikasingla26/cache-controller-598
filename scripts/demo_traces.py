"""Synthetic spliced traces with known transition points (detector ground truth)."""
from shiftcache import synthetic


def demo_trace(seed: int = 0):
    return synthetic([
        {"kind": "zipf", "n": 300_000, "universe": 50_000, "alpha": 1.0, "ws": 0},  # frequency-friendly
        {"kind": "zipf", "n": 200_000, "universe": 30_000, "alpha": 0.7, "ws": 1},  # new working set
        {"kind": "scan", "n": 60_000, "universe": 60_000, "ws": 2},                  # one-pass scan
        {"kind": "zipf", "n": 300_000, "universe": 50_000, "alpha": 1.0, "ws": 0},  # return to ws0
    ], seed=seed, name="demo")
