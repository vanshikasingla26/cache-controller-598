"""Self-calibrating shift detector.

Same statistic as detector.ShiftDetector (histogram overlap of a window against
the mean of the previous `ref_windows` windows), but the thresholds are not fixed
numbers. They are set from the trace's own recent overlap values:

    center, scale = median, 1.4826 * MAD of recent *non-transition* overlaps
    start : overlap < center - z_start * scale   for `persist` windows in a row
    end   : after `min_hold` windows, overlap >= center - z_end * scale for
            `persist` windows in a row (center/scale frozen at transition start),
            or `max_hold` windows elapse

Why: a threshold tuned on smooth synthetic traces (typical overlap 0.94) fires
almost constantly on a real I/O trace (typical overlap 0.70, much more variable),
see README. A trace-relative threshold asks "is this window unusual FOR THIS
TRACE", which is what a shift detector should ask.

Baseline hygiene: an overlap value only enters the baseline after `persist`
windows have passed without a transition starting, so the first low windows of a
real shift do not contaminate the baseline. The first `warmup` windows are
ignored (cold start looks like a shift: the share of first-ever accesses decays
for the first ~40 windows on a Zipf trace).

The decision logic lives in AdaptiveLogic.step(), a pure function of the overlap
sequence, so the online detector and offline tuning run the same code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import COLD_BIN, N_GAP_BINS, ShiftDetector


@dataclass
class AdaptiveConfig:
    window: int = 1000
    ref_windows: int = 10
    z_start: float = 4.0
    z_end: float = 2.0
    persist: int = 3
    warmup: int = 20          # windows ignored at the start of the trace
    base_windows: int = 40    # rolling baseline length (non-transition windows)
    min_base: int = 15        # windows needed before any flag can fire
    min_scale: float = 0.01   # floor on the robust scale
    min_hold: int = 20
    max_hold: int = 150


class AdaptiveLogic:
    """State machine over overlap values. step(ov) -> (in_transition, event)."""

    def __init__(self, cfg: AdaptiveConfig):
        self.cfg = cfg
        self.n = 0                      # overlap values seen
        self.base: list[float] = []
        self.pending: list[float] = []
        self.in_transition = False
        self.low_run = self.high_run = self.held = 0
        self.frozen = (0.0, cfg.min_scale)

    def _stats(self):
        b = np.asarray(self.base)
        med = float(np.median(b))
        mad = float(np.median(np.abs(b - med)))
        return med, max(1.4826 * mad, self.cfg.min_scale)

    def step(self, ov: float):
        c = self.cfg
        self.n += 1
        event = None
        if self.n <= c.warmup:
            return self.in_transition, event
        if not self.in_transition:
            if len(self.base) < c.min_base:
                self.base.append(ov)
                return self.in_transition, event
            center, scale = self._stats()
            self.low_run = self.low_run + 1 if ov < center - c.z_start * scale else 0
            self.pending.append(ov)
            if self.low_run >= c.persist:
                self.in_transition, self.held, self.high_run = True, 0, 0
                self.frozen = (center, scale)
                self.pending.clear()
                event = "start"
            else:
                while len(self.pending) > c.persist:
                    self.base.append(self.pending.pop(0))
                if len(self.base) > c.base_windows:
                    self.base = self.base[-c.base_windows:]
        else:
            self.held += 1
            center, scale = self.frozen
            self.high_run = self.high_run + 1 if ov >= center - c.z_end * scale else 0
            if (self.held >= c.min_hold and self.high_run >= c.persist) or self.held >= c.max_hold:
                self.in_transition, self.low_run = False, 0
                event = "end"
        return self.in_transition, event


class AdaptiveShiftDetector(ShiftDetector):
    """Drop-in for ShiftDetector (ControlledARC accepts it via isinstance)."""

    def __init__(self, cfg: AdaptiveConfig | None = None):
        cfg = cfg or AdaptiveConfig()
        super().__init__(cfg)           # window / ref_windows are read from cfg
        self.logic = AdaptiveLogic(cfg)

    def _close_window(self, t_next: int):
        c = self.cfg
        h = self.hist / self.hist.sum()
        if self.ref:
            ov = float(np.minimum(h, np.mean(self.ref, axis=0)).sum())
            self.overlaps.append(ov)
            self.in_transition, ev = self.logic.step(ov)
            if ev == "start":
                self.starts.append(t_next)
            elif ev == "end":
                self.ends.append(t_next)
        self.ref.append(h)
        if len(self.ref) > c.ref_windows:
            self.ref.pop(0)
        self.hist = np.zeros(N_GAP_BINS + 1)
        self.n_in_win = 0


def run_adaptive(keys, cfg: AdaptiveConfig | None = None):
    """Offline pass: (per-access in_transition mask, detector)."""
    d = AdaptiveShiftDetector(cfg)
    ks = keys.tolist() if hasattr(keys, "tolist") else list(keys)
    mask = np.fromiter((d.update(k, t) for t, k in enumerate(ks)), dtype=bool, count=len(ks))
    return mask, d


def detect_from_overlaps(overlaps, cfg: AdaptiveConfig):
    """Offline: starts/ends (access index) from a precomputed overlap series.
    overlaps[i] belongs to window i+1 (the first window has no reference), so
    its window closes at access (i + 2) * window."""
    logic = AdaptiveLogic(cfg)
    starts, ends = [], []
    for i, ov in enumerate(overlaps):
        _, ev = logic.step(float(ov))
        t_next = (i + 2) * cfg.window
        if ev == "start":
            starts.append(t_next)
        elif ev == "end":
            ends.append(t_next)
    return starts, ends
