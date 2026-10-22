from __future__ import annotations

from collections import OrderedDict, defaultdict


class Policy:
    """Interface every policy implements.

    access(key, t) -> True on hit, False on miss (the key is admitted on a miss).
    t is the index of this access in the block-access sequence; only offline
    policies (Belady) and learned controllers use it.
    """
    name = "base"

    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity

    def access(self, key: int, t: int) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key) -> bool:
        raise NotImplementedError


class LRU(Policy):
    name = "LRU"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.d = OrderedDict()

    def access(self, key, t):
        d = self.d
        if key in d:
            d.move_to_end(key)
            return True
        if len(d) >= self.capacity:
            d.popitem(last=False)
        d[key] = None
        return False

    def __len__(self):
        return len(self.d)

    def __contains__(self, key):
        return key in self.d


class LFU(Policy):
    """O(1) LFU: frequency buckets, LRU tie-break inside a bucket.

    Note: in-cache counts only (frequency resets on eviction), the textbook
    variant. Document this choice in the report.
    """
    name = "LFU"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.freq = {}
        self.buckets = defaultdict(OrderedDict)
        self.min_f = 0

    def _bump(self, key):
        f = self.freq[key]
        b = self.buckets[f]
        del b[key]
        if not b:
            del self.buckets[f]
            if self.min_f == f:
                self.min_f = f + 1
        self.freq[key] = f + 1
        self.buckets[f + 1][key] = None

    def access(self, key, t):
        if key in self.freq:
            self._bump(key)
            return True
        if len(self.freq) >= self.capacity:
            b = self.buckets[self.min_f]
            victim, _ = b.popitem(last=False)
            if not b:
                del self.buckets[self.min_f]
            del self.freq[victim]
        self.freq[key] = 1
        self.buckets[1][key] = None
        self.min_f = 1
        return False

    def __len__(self):
        return len(self.freq)

    def __contains__(self, key):
        return key in self.freq
