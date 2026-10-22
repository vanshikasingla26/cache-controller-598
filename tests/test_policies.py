import itertools

import numpy as np
import pytest

from shiftcache import simulate, synthetic, splice
from shiftcache.policies import LRU, LFU, ARC, Belady, OracleTailARC, next_use, NEVER


def small_trace(seed=0, n=30_000):
    return synthetic([
        {"kind": "zipf", "n": n // 2, "universe": 2_000, "alpha": 0.8, "ws": 0},
        {"kind": "scan", "n": n // 6, "universe": 3_000, "ws": 1},
        {"kind": "zipf", "n": n // 3, "universe": 1_500, "alpha": 1.0, "ws": 2},
    ], seed=seed)


# ---------------------------------------------------------------- basics
def test_lru_textbook():
    p = LRU(2)
    seq = [1, 2, 1, 3, 2, 1]
    assert [p.access(k, t) for t, k in enumerate(seq)] == [False, False, True, False, False, False]


def test_lfu_evicts_least_frequent():
    p = LFU(2)
    for t, k in enumerate([1, 1, 2, 3]):   # 2 has freq 1 -> evicted by 3
        p.access(k, t)
    assert 1 in p and 3 in p and 2 not in p


@pytest.mark.parametrize("cls", [LRU, LFU, ARC])
def test_capacity_never_exceeded(cls):
    tr = small_trace()
    pol = cls(100)
    for t, k in enumerate(tr.keys.tolist()):
        pol.access(k, t)
        assert len(pol) <= 100


# ---------------------------------------------------------------- ARC
def test_arc_invariants_hold_throughout():
    tr = small_trace(seed=3)
    for c in (1, 7, 64, 500):
        a = ARC(c)
        for t, k in enumerate(tr.keys.tolist()):
            a.access(k, t)
            if t % 97 == 0:
                a.check_invariants()
        a.check_invariants()


def test_arc_p_moves_on_ghost_hits():
    a = ARC(4)
    # 1 promoted to T2; 2,3,4 fill T1; 5 pushes 2 into ghost B1; 2 returns -> B1 hit
    for t, k in enumerate([1, 1, 2, 3, 4, 5, 2]):
        a.access(k, t)
    assert a.last_event == "B1" and a.p > 0


def test_arc_candidates_match_replace_choice():
    """candidates()[0] must be exactly what ARC would evict."""
    tr = small_trace(seed=5)
    a = ARC(50)
    keys = tr.keys.tolist()
    checked = 0
    for t, k in enumerate(keys):
        if k not in a and len(a) == a.capacity and not (len(a.T1) + len(a.B1) == a.capacity and len(a.T1) == a.capacity):
            src, cands = a.candidates(k, 4)
            before = set(a.T1) | set(a.T2)
            a.access(k, t)
            evicted = before - (set(a.T1) | set(a.T2))
            assert evicted == {cands[0]}, (t, src)
            checked += 1
        else:
            a.access(k, t)
    assert checked > 1000


@pytest.mark.parametrize("cap", [16, 128, 700])
def test_arc_matches_libcachesim(cap):
    lcs = pytest.importorskip("libcachesim")
    tr = small_trace(seed=11)
    ours = simulate(tr, ARC(cap)).hit_rate
    ref = lcs.ARC(cache_size=cap)
    hits = 0
    for k in tr.keys.tolist():
        hits += bool(ref.get(lcs.Request(obj_size=1, obj_id=int(k) % (1 << 62))))
    theirs = hits / len(tr)
    # Verified bit-exact against libCacheSim 0.3.5; any drift means we broke ARC.
    assert ours == theirs, (ours, theirs)


# ---------------------------------------------------------------- Belady
def test_next_use():
    nu = next_use(np.array([5, 6, 5, 7, 6]))
    assert nu.tolist() == [2, 4, NEVER, NEVER, NEVER]


def brute_force_opt(keys, cap):
    """Exhaustive optimum over eviction choices (tiny inputs only)."""
    best = [len(keys)]

    def rec(t, cache, hits):
        if t == len(keys):
            best[0] = min(best[0], len(keys) - hits)
            return
        k = keys[t]
        if k in cache:
            return rec(t + 1, cache, hits + 1)
        if len(cache) < cap:
            return rec(t + 1, cache | {k}, hits)
        for v in cache:
            rec(t + 1, (cache - {v}) | {k}, hits)

    rec(0, frozenset(), 0)
    return best[0]


def test_belady_is_optimal_on_small_inputs():
    rng = np.random.default_rng(0)
    for _ in range(40):
        keys = rng.integers(0, 5, size=10)
        nu = next_use(keys)
        b = Belady(2, nu)
        misses = sum(not b.access(int(k), t) for t, k in enumerate(keys))
        assert misses == brute_force_opt(keys.tolist(), 2)


def test_ordering_belady_beats_everyone():
    tr = small_trace(seed=2)
    nu = next_use(tr.keys)
    cap = 200
    hr = {p.name: simulate(tr, p).hit_rate for p in (LRU(cap), LFU(cap), ARC(cap), Belady(cap, nu))}
    assert hr["Belady"] >= max(v for k, v in hr.items() if k != "Belady")


def test_oracle_k1_equals_arc_and_k_helps():
    tr = small_trace(seed=4)
    nu = next_use(tr.keys)
    arc = simulate(tr, ARC(150)).hit_rate
    assert simulate(tr, OracleTailARC(150, nu, k=1)).hit_rate == arc
    assert simulate(tr, OracleTailARC(150, nu, k=16)).hit_rate >= arc


def test_splice_ground_truth():
    a, b = small_trace(0, 6000), small_trace(1, 6000)
    s = splice(a, b, 4000, 3000)
    assert len(s) == 7000 and s.transitions == [4000]
    assert not set(s.keys[:4000].tolist()) & set(s.keys[4000:].tolist())


def test_cross_list_oracle_keeps_arc_invariants():
    from shiftcache.policies import OracleCrossListARC
    tr = small_trace(seed=7)
    nu = next_use(tr.keys)
    a = OracleCrossListARC(80, nu, k=4)
    for t, k in enumerate(tr.keys.tolist()):
        a.access(k, t)
        if t % 53 == 0:
            a.check_invariants()
    assert a.n_list_flips > 0
