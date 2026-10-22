import numpy as np

from shiftcache import simulate, synthetic
from shiftcache.detector import DetectorConfig, ShiftDetector, run_detector, score_detection
from shiftcache.policies import ARC, ControlledARC, next_use, FEATURES


def tr_():
    return synthetic([
        {"kind": "zipf", "n": 40_000, "universe": 3_000, "alpha": 1.0, "ws": 0},
        {"kind": "zipf", "n": 30_000, "universe": 2_000, "alpha": 0.7, "ws": 1},
        {"kind": "zipf", "n": 30_000, "universe": 3_000, "alpha": 1.0, "ws": 0},
    ], seed=1)


class Const:
    def __init__(self, v): self.v = v
    def predict_proba(self, X):
        p = np.full(len(X), self.v); return np.c_[1 - p, p]


class Rand:
    def __init__(self): self.r = np.random.default_rng(0)
    def predict_proba(self, X):
        p = self.r.random(len(X)); return np.c_[1 - p, p]


def test_no_gate_is_exact_arc():
    tr = tr_()
    assert simulate(tr, ControlledARC(200)).hit_rate == simulate(tr, ARC(200)).hit_rate


def test_unconfident_model_is_exact_arc():
    tr = tr_()
    r = simulate(tr, ControlledARC(200, gate="always", model=Const(0.5))).hit_rate
    assert r == simulate(tr, ARC(200)).hit_rate


def test_random_model_keeps_invariants_and_flips_lists():
    tr = tr_()
    a = ControlledARC(150, gate="always", model=Rand())
    for t, k in enumerate(tr.keys.tolist()):
        a.access(k, t)
        if t % 61 == 0:
            a.check_invariants()
    assert a.n_list_flips > 0 and len(a) <= 150


def test_collection_does_not_change_arc_and_labels_are_future_only():
    tr = tr_()
    nu = next_use(tr.keys)
    c = ControlledARC(200, gate="always", collect=True, nu=nu)
    assert simulate(tr, c).hit_rate == simulate(tr, ARC(200)).hit_rate
    X = c.rows
    assert X.shape[1] == len(FEATURES) and len(c.labels) == len(X)
    assert 0 < np.mean(c.labels) < 1


def test_detector_finds_shifts_and_online_equals_offline():
    tr = tr_()
    cfg = DetectorConfig(min_hold=10, start_thresh=0.7, persist=2)
    mask, d = run_detector(tr.keys, cfg)
    sc = score_detection(d.starts, tr.transitions, 10_000)
    assert sc["recall"] == 1.0 and sc["fp"] == 0
    online = ShiftDetector(cfg)
    assert [online.update(k, t) for t, k in enumerate(tr.keys.tolist())] == mask.tolist()


def test_detector_quiet_on_stationary_stream():
    st = synthetic([{"kind": "zipf", "n": 200_000, "universe": 20_000, "alpha": 0.9}], seed=5)
    _, d = run_detector(st.keys)
    assert d.starts == []
