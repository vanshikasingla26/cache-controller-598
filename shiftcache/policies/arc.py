"""ARC (Megiddo & Modha, FAST '03), transcribed from the paper's Figure 4.

Lists (OrderedDict, first item = LRU end, last item = MRU end):
    T1: in cache, seen once recently      B1: ghosts evicted from T1
    T2: in cache, seen >= twice recently  B2: ghosts evicted from T2
    p : target size of T1 (the adaptation parameter we log)

Hooks for the shift-aware controller (used later, harmless now):
    candidates(key, k) -> (list_name, [k keys nearest the LRU end of the list
                            REPLACE would evict from if `key` missed now])
    choose_victim(list_name, lst, key, t) -> key to evict. Default = LRU end,
                            i.e. exact ARC. The re-ranker overrides this.
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import islice

from .basic import Policy


class ARC(Policy):
    name = "ARC"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.T1, self.T2 = OrderedDict(), OrderedDict()
        self.B1, self.B2 = OrderedDict(), OrderedDict()
        self.p = 0.0
        # instrumentation
        self.last_event = None          # 'T1'|'T2' hit, 'B1'|'B2' ghost hit, 'miss'
        self.n_b1_hits = self.n_b2_hits = 0

    # ------------------------------------------------------------------ hooks
    def choose_victim(self, list_name, lst, key, t):
        return next(iter(lst))

    def _replace_source(self, in_b2: bool, p: float | None = None) -> str:
        p = self.p if p is None else p
        t1 = len(self.T1)
        if t1 >= 1 and (t1 > p or (in_b2 and t1 == p)):
            return "T1"
        return "T2"

    def _p_after(self, key) -> float:
        """p as it will be when REPLACE runs for this access (ghost hits move p first)."""
        c = self.capacity
        if key in self.B1:
            return min(float(c), self.p + max(len(self.B2) / len(self.B1), 1.0))
        if key in self.B2:
            return max(0.0, self.p - max(len(self.B1) / len(self.B2), 1.0))
        return self.p

    def candidates(self, key, k):
        """Eviction candidates for a miss on `key`, in ARC's order (LRU first).

        Accounts for the p update that ghost hits apply before REPLACE.
        Call this *before* access(key, t).
        """
        src = self._replace_source(key in self.B2, self._p_after(key))
        lst = self.T1 if src == "T1" else self.T2
        return src, list(islice(iter(lst), k))

    # ---------------------------------------------------------------- core
    def _replace(self, key, t):
        src = self._replace_source(key in self.B2)
        if src == "T1":
            v = self.choose_victim("T1", self.T1, key, t)
            del self.T1[v]
            self.B1[v] = None
        else:
            v = self.choose_victim("T2", self.T2, key, t)
            del self.T2[v]
            self.B2[v] = None

    def access(self, key, t):
        c = self.capacity
        T1, T2, B1, B2 = self.T1, self.T2, self.B1, self.B2

        # Case I: hit
        if key in T1:
            del T1[key]
            T2[key] = None
            self.last_event = "T1"
            return True
        if key in T2:
            T2.move_to_end(key)
            self.last_event = "T2"
            return True

        # Case II: ghost hit in B1 -> grow recency side
        if key in B1:
            self.p = min(float(c), self.p + max(len(B2) / len(B1), 1.0))
            self._replace(key, t)
            del B1[key]
            T2[key] = None
            self.last_event = "B1"
            self.n_b1_hits += 1
            return False

        # Case III: ghost hit in B2 -> grow frequency side
        if key in B2:
            self.p = max(0.0, self.p - max(len(B1) / len(B2), 1.0))
            self._replace(key, t)
            del B2[key]
            T2[key] = None
            self.last_event = "B2"
            self.n_b2_hits += 1
            return False

        # Case IV: complete miss
        l1 = len(T1) + len(B1)
        if l1 == c:
            if len(T1) < c:
                B1.popitem(last=False)
                self._replace(key, t)
            else:
                v = self.choose_victim("T1", T1, key, t)
                del T1[v]  # evicted with no ghost, per the paper
        elif l1 < c:
            total = l1 + len(T2) + len(B2)
            if total >= c:
                if total == 2 * c:
                    B2.popitem(last=False)
                self._replace(key, t)
        T1[key] = None
        self.last_event = "miss"
        return False

    def __len__(self):
        return len(self.T1) + len(self.T2)

    def __contains__(self, key):
        return key in self.T1 or key in self.T2

    def check_invariants(self):
        c = self.capacity
        assert len(self.T1) + len(self.T2) <= c
        assert len(self.T1) + len(self.B1) <= c
        assert len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) <= 2 * c
        assert 0 <= self.p <= c
        if self.B1 or self.B2:
            assert len(self.T1) + len(self.T2) == c
        s = [set(self.T1), set(self.T2), set(self.B1), set(self.B2)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert not (s[i] & s[j])
