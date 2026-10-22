"""Oracle re-ranker over ARC's tail (headroom experiment #2).

Identical to ARC except that when REPLACE evicts from T1/T2, it looks at the k
items nearest the LRU end and evicts the one whose next use is furthest away.
No learned re-ranker restricted to the same k candidates can beat this, so the
gap (Oracle-k vs ARC) is the ceiling on what the project can win.

`active` lets you restrict the oracle to certain time ranges (e.g. only inside
ground-truth transition windows) to bound a controller that only runs then.
"""
from __future__ import annotations

from itertools import islice

import numpy as np

from .arc import ARC


class OracleTailARC(ARC):
    def __init__(self, capacity, nu: np.ndarray, k: int = 8, active: np.ndarray | None = None):
        super().__init__(capacity)
        self.nu, self.k = nu, k
        self.active = active            # bool array over t, or None = always on
        self.key_nu = {}                # key -> next use as of its last access
        self.name = f"Oracle-ARC(k={k})" + ("" if active is None else "[windows]")
        self.n_overrides = 0

    def access(self, key, t):
        hit = super().access(key, t)
        self.key_nu[key] = int(self.nu[t])
        return hit

    def choose_victim(self, list_name, lst, key, t):
        if self.active is not None and not self.active[t]:
            return next(iter(lst))
        cands = list(islice(iter(lst), self.k))
        best = max(cands, key=lambda c: self.key_nu.get(c, 0))
        if best != cands[0]:
            self.n_overrides += 1
        return best


class OracleCrossListARC(OracleTailARC):
    """Oracle whose candidates come from the tails of BOTH T1 and T2.

    This can override ARC's list choice (i.e. correct a lagging p), not just the
    order within one list. Victim goes to the ghost list matching where it lived,
    so ARC's own bookkeeping stays consistent.
    """

    def __init__(self, capacity, nu, k=8, active=None):
        super().__init__(capacity, nu, k, active)
        self.name = f"Oracle-X(k={k})" + ("" if active is None else "[windows]")
        self.n_list_flips = 0

    def _replace(self, key, t):
        if self.active is not None and not self.active[t]:
            return super()._replace(key, t)
        src = self._replace_source(key in self.B2)
        cands = [("T1", c) for c in islice(iter(self.T1), self.k)] + \
                [("T2", c) for c in islice(iter(self.T2), self.k)]
        lst_name, v = max(cands, key=lambda lc: self.key_nu.get(lc[1], 0))
        if lst_name != src:
            self.n_list_flips += 1
        if lst_name == "T1":
            del self.T1[v]
            self.B1[v] = None
        else:
            del self.T2[v]
            self.B2[v] = None
