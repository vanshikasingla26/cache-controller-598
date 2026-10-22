"""Shift-aware controller: ARC + a learned re-ranker that runs only when gated on.

When the gate is on for access t and REPLACE must evict, candidates are the k
items nearest the LRU end of BOTH T1 and T2 (the headroom study showed the win is
in *which list* to evict from). Each candidate gets

    arc_keep = its position in ARC's own eviction order, scaled to [0, 1]
               (0 = the block ARC would evict)
    p        = model's P(reused within the next `horizon` accesses)
    w        = w_max * |2p - 1|            (confidence: 0 at p=0.5)
    score    = (1 - w) * arc_keep + w * p

and the lowest score is evicted. A shaky model (p near 0.5) leaves ARC's choice.
The victim goes to the ghost list matching the list it lived in, so ARC's
bookkeeping (and p adaptation) stays consistent.

Gate options (RQ4):
  None                    -> never (plain ARC; features still tracked)
  "always"                -> HALP-style always-on
  bool ndarray over t     -> ground-truth windows, random windows, or a
                             precomputed detector mask (the detector reads only
                             the stream, so offline == online)
  ShiftDetector instance  -> run the detector online inside access()

Collection mode (model=None, collect=True, nu given): at every gated eviction,
record features + labels for all candidates, then evict exactly as ARC would.
"""
from __future__ import annotations

import math
import time
from array import array
from itertools import islice

import numpy as np

from ..detector import ShiftDetector
from .arc import ARC

FEATURES = [
    "log_age", "log_freq", "log_last_gap", "has_gap", "in_T2", "list_pos",
    "in_arc_src", "arc_keep", "log_ghost", "p_frac", "t1_frac",
    "b1_rate", "b2_rate", "miss_rate", "log_since_gate",
]


class ControlledARC(ARC):
    def __init__(self, capacity, k: int = 8, gate=None, model=None, w_max: float = 1.0,
                 collect: bool = False, nu: np.ndarray | None = None,
                 horizon: int | None = None, ctx_window: int = 1000, name: str | None = None):
        super().__init__(capacity)
        self.k, self.gate, self.model, self.w_max = k, gate, model, w_max
        self.collect, self.nu = collect, nu
        self.horizon = horizon or capacity
        self.ctx_window = ctx_window
        self.name = name or "ControlledARC"
        # per-key history (includes evicted keys, like a ghost-aware feature store)
        self.last_t: dict[int, int] = {}
        self.freq: dict[int, int] = {}
        self.last_gap: dict[int, int] = {}
        self.ghost: dict[int, int] = {}
        self.key_nu: dict[int, int] = {}
        # context counters: rates from the last complete ctx window
        self._cb1 = self._cb2 = self._cmiss = self._cn = 0
        self.b1_rate = self.b2_rate = self.miss_rate = 0.0
        self.gate_since = -1
        self._gate_now = False
        self._in_replace = False
        # logs
        self._rows = array("d")          # flat, len(FEATURES) per row
        self._labels = array("b")
        self._groups = array("l")
        self.collect_every = 1           # record every n-th gated decision
        self.n_decisions = self.n_model_calls = self.n_changed = self.n_list_flips = 0
        self.model_s = 0.0

    # --------------------------------------------------------------- gating
    def _gate(self, key, t) -> bool:
        g = self.gate
        if g is None:
            return False
        if isinstance(g, str):
            return g == "always"
        if isinstance(g, ShiftDetector):
            return g.update(key, t)
        return bool(g[t])

    # --------------------------------------------------------------- access
    def access(self, key, t):
        on = self._gate(key, t)
        if on and not self._gate_now:
            self.gate_since = t
        self._gate_now = on
        self._t = t
        hit = super().access(key, t)
        # history updates (after the decision, so features never see the present access)
        prev = self.last_t.get(key)
        if prev is not None:
            self.last_gap[key] = t - prev
        self.last_t[key] = t
        self.freq[key] = self.freq.get(key, 0) + 1
        if self.nu is not None:
            self.key_nu[key] = int(self.nu[t])
        ev = self.last_event
        if ev == "B1" or ev == "B2":
            self.ghost[key] = self.ghost.get(key, 0) + 1
            if ev == "B1":
                self._cb1 += 1
            else:
                self._cb2 += 1
        if not hit:
            self._cmiss += 1
        self._cn += 1
        if self._cn == self.ctx_window:
            n = self._cn
            self.b1_rate, self.b2_rate, self.miss_rate = self._cb1 / n, self._cb2 / n, self._cmiss / n
            self._cb1 = self._cb2 = self._cmiss = self._cn = 0
        return hit

    # ------------------------------------------------------------- features
    def _features(self, cands, src, t):
        c = self.capacity
        n = len(cands)
        ctx = [self.p / c, len(self.T1) / c, self.b1_rate, self.b2_rate, self.miss_rate,
               math.log1p(t - self.gate_since) if self.gate_since >= 0 else 0.0]
        rows = []
        for rank, (lname, key, pos) in enumerate(cands):
            g = self.last_gap.get(key)
            rows.append([
                math.log1p(t - self.last_t.get(key, t)),
                math.log1p(self.freq.get(key, 0)),
                math.log1p(g) if g is not None else 0.0,
                1.0 if g is not None else 0.0,
                1.0 if lname == "T2" else 0.0,
                pos / max(self.k - 1, 1),
                1.0 if lname == src else 0.0,
                rank / max(n - 1, 1),
                math.log1p(self.ghost.get(key, 0)),
                *ctx[:2], ctx[2], ctx[3], ctx[4], ctx[5],
            ])
        return rows

    def _arc_order(self, src, lists):
        """Candidates in ARC's own eviction order: chosen list's tail first."""
        other = "T2" if src == "T1" else "T1"
        out = []
        for lname in (src, other):
            if lname in lists:
                lst = self.T1 if lname == "T1" else self.T2
                out += [(lname, key, i) for i, key in enumerate(islice(iter(lst), self.k))]
        return out

    def _decide(self, src, lists, t):
        """Return (list_name, victim) or None to fall back to ARC."""
        cands = self._arc_order(src, lists)
        if len(cands) <= 1:
            return None
        X = self._features(cands, src, t)
        self.n_decisions += 1
        if self.collect and self.nu is not None and self.n_decisions % self.collect_every == 0:
            gid = self.n_decisions
            for (lname, key, _), row in zip(cands, X):
                self._rows.extend(row)
                self._labels.append(int(self.key_nu.get(key, 0) - t <= self.horizon))
                self._groups.append(gid)
        if self.model is None:
            return None
        t0 = time.perf_counter()
        p = self.model.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]
        self.model_s += time.perf_counter() - t0
        self.n_model_calls += 1
        n = len(cands)
        keep = np.arange(n) / (n - 1)
        w = self.w_max * np.abs(2 * p - 1)
        score = (1 - w) * keep + w * p
        i = int(np.argmin(score))
        if i != 0:
            self.n_changed += 1
        if cands[i][0] != src:
            self.n_list_flips += 1
        return cands[i][0], cands[i][1]

    # ------------------------------------------------------------ eviction
    def _replace(self, key, t):
        if not self._gate_now:
            return super()._replace(key, t)
        src = self._replace_source(key in self.B2)
        lists = [l for l in ("T1", "T2") if (self.T1 if l == "T1" else self.T2)]
        d = self._decide(src, lists, t)
        if d is None:
            self._in_replace = True
            try:
                return super()._replace(key, t)
            finally:
                self._in_replace = False
        lname, v = d
        if lname == "T1":
            del self.T1[v]
            self.B1[v] = None
        else:
            del self.T2[v]
            self.B2[v] = None

    def choose_victim(self, list_name, lst, key, t):
        # Only reached directly in Case IV with T1 == c (T2 empty): same-list re-rank.
        if self._in_replace or not self._gate_now or lst is not self.T1 or self.T2:
            return next(iter(lst))
        d = self._decide("T1", ["T1"], t)
        return next(iter(lst)) if d is None else d[1]

    @property
    def rows(self) -> np.ndarray:
        return np.frombuffer(self._rows, dtype=np.float64).reshape(-1, len(FEATURES)).copy()

    @property
    def labels(self) -> np.ndarray:
        return np.frombuffer(self._labels, dtype=np.int8).astype(np.int64)

    @property
    def groups(self) -> np.ndarray:
        return np.frombuffer(self._groups, dtype=np.int64).copy()

    def stats(self) -> dict:
        return {"decisions": self.n_decisions, "model_calls": self.n_model_calls,
                "changed": self.n_changed, "list_flips": self.n_list_flips,
                "us_per_call": 1e6 * self.model_s / self.n_model_calls if self.n_model_calls else 0.0}
