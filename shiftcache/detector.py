"""Shift detector: reads ONLY the raw request stream (never any policy's state).

Each non-overlapping window of `window` accesses is summarized as a histogram of
log2 reuse gaps (gap = t - previous access time of the same key), plus one extra
bin for first-ever accesses ("cold"). Each window is compared to a REFERENCE =
mean histogram of the previous `ref_windows` windows, by histogram overlap
sum_i min(p_i, q_i) in [0, 1].

ref_windows=1 is the proposal's literal "each window vs the one immediately
before". Measured on the demo trace, that gives a single-window dip at an abrupt
shift (post-shift windows resemble each other), which `persist` then filters out,
so it detects nothing. ref_windows ~ 10 keeps the dip visible for ~10 windows.

  start trigger : overlap < start_thresh for `persist` windows in a row
  end trigger   : after at least `min_hold` windows, overlap >= end_thresh for
                  `persist` windows in a row (the stream has settled)  --  or
                  `max_hold` windows elapse.

The flag only changes at window boundaries, so decisions inside window w use
information from windows <= w-1 (no peeking).

Memory note: `last_seen` grows with the number of distinct keys. For MSR scale
this is fine in Python; a production version would use a bounded sketch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_GAP_BINS = 32          # log2 gap bins 0..31 (gaps up to ~4e9)
COLD_BIN = N_GAP_BINS    # first-ever access


@dataclass
class DetectorConfig:
    window: int = 1000
    start_thresh: float = 0.6
    end_thresh: float = 0.8
    persist: int = 3
    ref_windows: int = 10
    min_hold: int = 50
    max_hold: int = 150


class ShiftDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.last_seen: dict[int, int] = {}
        self.hist = np.zeros(N_GAP_BINS + 1)
        self.ref: list[np.ndarray] = []
        self.n_in_win = 0
        self.in_transition = False
        self.low_run = self.high_run = self.held = 0
        # logs
        self.overlaps: list[float] = []
        self.starts: list[int] = []      # access index at which the flag turned on
        self.ends: list[int] = []

    def update(self, key: int, t: int) -> bool:
        """Feed one access; returns the flag that applies to THIS access."""
        prev = self.last_seen.get(key)
        self.hist[COLD_BIN if prev is None else min(int(t - prev).bit_length(), N_GAP_BINS - 1)] += 1
        self.last_seen[key] = t
        self.n_in_win += 1
        flag = self.in_transition
        if self.n_in_win == self.cfg.window:
            self._close_window(t + 1)
        return flag

    def _close_window(self, t_next: int):
        c = self.cfg
        h = self.hist / self.hist.sum()
        if self.ref:
            ov = float(np.minimum(h, np.mean(self.ref, axis=0)).sum())
            self.overlaps.append(ov)
            if not self.in_transition:
                self.low_run = self.low_run + 1 if ov < c.start_thresh else 0
                if self.low_run >= c.persist:
                    self.in_transition, self.held, self.high_run = True, 0, 0
                    self.starts.append(t_next)
            else:
                self.held += 1
                self.high_run = self.high_run + 1 if ov >= c.end_thresh else 0
                if (self.held >= c.min_hold and self.high_run >= c.persist) or self.held >= c.max_hold:
                    self.in_transition, self.low_run = False, 0
                    self.ends.append(t_next)
        self.ref.append(h)
        if len(self.ref) > c.ref_windows:
            self.ref.pop(0)
        self.hist = np.zeros(N_GAP_BINS + 1)
        self.n_in_win = 0


def run_detector(keys, cfg: DetectorConfig | None = None) -> tuple[np.ndarray, ShiftDetector]:
    """Offline pass: per-access in_transition mask + the detector (for its logs)."""
    d = ShiftDetector(cfg)
    ks = keys.tolist() if hasattr(keys, "tolist") else list(keys)
    mask = np.fromiter((d.update(k, t) for t, k in enumerate(ks)), dtype=bool, count=len(ks))
    return mask, d


def score_detection(starts: list[int], truth: list[int], tol: int) -> dict:
    """Match detected starts to ground-truth shifts: a detection within [s, s+tol]
    of a true shift s is a true positive (first one only)."""
    used, delays = set(), []
    for s in truth:
        hit = next((i for i, d in enumerate(starts) if i not in used and s <= d <= s + tol), None)
        if hit is not None:
            used.add(hit)
            delays.append(starts[hit] - s)
    tp = len(delays)
    return {"tp": tp, "fp": len(starts) - tp, "fn": len(truth) - tp,
            "precision": tp / len(starts) if starts else float("nan"),
            "recall": tp / len(truth) if truth else float("nan"),
            "delays": delays}


def arc_adaptation_delay(p_trace: np.ndarray, window: int, truth: list[int], n: int,
                         capacity: int, frac: float = 0.5, min_move: float = 0.05) -> list[dict]:
    """How long ARC's p takes to react to each shift.

    p does not settle at a new level after a shift: it swings out and decays back.
    So we measure the swing: peak excursion |p - p_before| within the segment, and
    the accesses until p first covers `frac` of that peak (t_half) and reaches it
    (t_peak). If the peak is below `min_move` * capacity, ARC effectively did not
    react (t_half = None)."""
    out = []
    bounds = list(truth) + [n]
    for i, s in enumerate(truth):
        w0, w1 = s // window, bounds[i + 1] // window
        before = p_trace[w0 - 1] if w0 > 0 else 0.0
        exc = np.abs(p_trace[w0:w1] - before)
        peak = float(exc.max()) if exc.size else 0.0
        if peak < min_move * capacity:
            out.append({"shift": s, "peak_frac": peak / capacity, "t_half": None, "t_peak": None})
            continue
        t_half = int((np.argmax(exc >= frac * peak) + 1) * window)
        t_peak = int((np.argmax(exc) + 1) * window)
        out.append({"shift": s, "peak_frac": peak / capacity, "t_half": t_half, "t_peak": t_peak})
    return out
