from __future__ import annotations

import heapq

import numpy as np

from .basic import Policy

NEVER = np.iinfo(np.int64).max


def next_use(keys: np.ndarray) -> np.ndarray:
    """nu[t] = index of the next access to keys[t] after t, or NEVER.

    This is the single source of truth for Belady *and* for re-ranker labels
    ("reused within N accesses" == nu[t] - t <= N).
    """
    n = len(keys)
    nu = np.full(n, NEVER, dtype=np.int64)
    last = {}
    for t in range(n - 1, -1, -1):
        k = int(keys[t])
        nu[t] = last.get(k, NEVER)
        last[k] = t
    return nu


class Belady(Policy):
    """Offline optimal (MIN): evict the block whose next use is furthest away.

    Max-heap on next-use with lazy deletion. Needs the next_use array for the
    exact same key sequence you simulate.
    """
    name = "Belady"

    def __init__(self, capacity, nu: np.ndarray):
        super().__init__(capacity)
        self.nu = nu
        self.cur = {}     # key -> its current next-use time
        self.heap = []    # (-next_use, key)

    def access(self, key, t):
        n = int(self.nu[t])
        hit = key in self.cur
        if not hit and len(self.cur) >= self.capacity:
            while True:
                negn, v = heapq.heappop(self.heap)
                if self.cur.get(v) == -negn:
                    del self.cur[v]
                    break
        self.cur[key] = n
        heapq.heappush(self.heap, (-n, key))
        if len(self.heap) > 4 * self.capacity + 1024:  # compact stale entries
            self.heap = [(-nv, k) for k, nv in self.cur.items()]
            heapq.heapify(self.heap)
        return hit

    def __len__(self):
        return len(self.cur)

    def __contains__(self, key):
        return key in self.cur
