from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .policies.arc import ARC
from .trace import Trace

# Fixed per-block costs (microseconds), roughly "DRAM/SSD cache in front of HDD".
# Proposal says fixed per-hit/per-miss costs; tune and cite in the report.
DEFAULT_HIT_US = 100.0
DEFAULT_MISS_US = 5000.0


@dataclass
class Result:
    policy: str
    trace: str
    capacity: int
    hits: np.ndarray                 # bool per block access
    window: int
    p_trace: np.ndarray | None = None  # ARC p sampled at end of each window
    wall_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Block hit rate == byte hit rate (all blocks are 4KB)."""
        return float(self.hits.mean())

    def window_hit_rates(self) -> np.ndarray:
        n = len(self.hits) // self.window * self.window
        return self.hits[:n].reshape(-1, self.window).mean(axis=1)

    def request_hit_rate(self, req_ids: np.ndarray) -> float:
        """Fraction of requests whose blocks ALL hit."""
        misses_per_req = np.bincount(req_ids, weights=~self.hits, minlength=req_ids[-1] + 1)
        touched = np.bincount(req_ids, minlength=req_ids[-1] + 1) > 0
        return float((misses_per_req[touched] == 0).mean())

    def mean_latency_us(self, hit_us=DEFAULT_HIT_US, miss_us=DEFAULT_MISS_US) -> float:
        h = self.hits.mean()
        return float(h * hit_us + (1 - h) * miss_us)

    def summary(self) -> dict:
        return {"policy": self.policy, "trace": self.trace, "capacity": self.capacity,
                "hit_rate": round(self.hit_rate, 5),
                "latency_us": round(self.mean_latency_us(), 1),
                "wall_s": round(self.wall_s, 2), **self.extra}


def simulate(trace: Trace, policy, window: int = 1000, write_allocate: bool = True,
             progress: bool = False) -> Result:
    keys = trace.keys.tolist()          # python ints: much faster dict ops than np.int64
    writes = trace.is_write.tolist()
    n = len(keys)
    hits = np.zeros(n, dtype=bool)
    is_arc = isinstance(policy, ARC)
    p_trace = np.zeros(n // window + 1) if is_arc else None
    access = policy.access

    t0 = time.perf_counter()
    for t in range(n):
        k = keys[t]
        if not write_allocate and writes[t] and k not in policy:
            continue                     # write bypasses cache: counted as miss
        if access(k, t):
            hits[t] = True
        if is_arc and (t + 1) % window == 0:
            p_trace[(t + 1) // window - 1] = policy.p
        if progress and t % 1_000_000 == 0 and t:
            print(f"  {policy.name}: {t:,}/{n:,}", flush=True)
    wall = time.perf_counter() - t0

    res = Result(policy.name, trace.name, policy.capacity, hits, window,
                 p_trace[: n // window] if is_arc else None, wall)
    if is_arc:
        res.extra.update(b1_ghost_hits=policy.n_b1_hits, b2_ghost_hits=policy.n_b2_hits)
    return res


# ------------------------------------------------------------ transition metrics
def transition_mask(n: int, transitions: list[int], length: int) -> np.ndarray:
    """True for accesses within `length` accesses after any ground-truth transition."""
    m = np.zeros(n, dtype=bool)
    for tr in transitions:
        m[tr: min(n, tr + length)] = True
    return m


def split_hit_rates(res: Result, mask: np.ndarray) -> dict:
    return {"transition_hr": float(res.hits[mask].mean()) if mask.any() else float("nan"),
            "steady_hr": float(res.hits[~mask].mean()) if (~mask).any() else float("nan")}
