import numpy as np

from shiftcache import synthetic
from shiftcache.detector import score_detection
from shiftcache.detector_adaptive import (AdaptiveConfig, AdaptiveLogic, detect_from_overlaps,
                                          run_adaptive)


def _server(seed=0):
    segs = [{"kind": "zipf", "n": 150_000, "universe": 30_000, "alpha": 0.9, "ws": 0},
            {"kind": "zipf", "n": 150_000, "universe": 30_000, "alpha": 0.9, "ws": 1}]
    return synthetic(segs, seed=seed, name="two_ws")


def test_online_matches_offline():
    tr = _server()
    cfg = AdaptiveConfig(window=1000, ref_windows=10, z_start=4, persist=3)
    _, d = run_adaptive(tr.keys, cfg)
    starts, ends = detect_from_overlaps(d.overlaps, cfg)
    assert starts == d.starts and ends == d.ends


def test_detects_working_set_change_and_not_before():
    tr = _server()
    cfg = AdaptiveConfig(window=1000, ref_windows=10, z_start=4, persist=3)
    _, d = run_adaptive(tr.keys, cfg)
    s = tr.transitions[0]
    s_eval = score_detection(d.starts, tr.transitions, 20_000)
    assert s_eval["tp"] == 1 and s_eval["fp"] == 0
    assert all(x >= s for x in d.starts)  # no pre-shift false alarm


def test_quiet_on_stationary_trace():
    segs = [{"kind": "zipf", "n": 300_000, "universe": 30_000, "alpha": 0.9, "ws": 0}]
    tr = synthetic(segs, seed=3, name="stationary")
    _, d = run_adaptive(tr.keys, AdaptiveConfig())
    assert d.starts == []


def test_noisy_stationary_overlaps_do_not_fire():
    # a "real-trace-like" series: low centre, wide spread; fixed threshold 0.8
    # would fire constantly, a trace-relative one must not
    rng = np.random.default_rng(0)
    ov = np.clip(rng.normal(0.70, 0.10, 400), 0, 1)
    starts, _ = detect_from_overlaps(ov, AdaptiveConfig(z_start=4, persist=3))
    assert starts == []
    assert (ov < 0.8).mean() > 0.5  # confirms the fixed threshold would have fired a lot


def test_clear_dip_fires_then_ends_and_baseline_stays_clean():
    rng = np.random.default_rng(1)
    ov = list(np.clip(rng.normal(0.70, 0.03, 120), 0, 1))
    ov[60:68] = [0.30, 0.25, 0.28, 0.33, 0.40, 0.50, 0.60, 0.68]  # shift, recovery
    cfg = AdaptiveConfig(z_start=4, persist=3, min_hold=5)
    logic = AdaptiveLogic(cfg)
    events = [logic.step(float(v))[1] for v in ov]
    assert "start" in events and "end" in events
    assert events.index("start") >= 62          # needs `persist` low windows
    med = float(np.median(logic.base))
    assert 0.65 < med < 0.75                     # dip did not pollute the baseline


def test_no_flag_during_warmup_or_before_min_base():
    cfg = AdaptiveConfig(warmup=20, min_base=15)
    logic = AdaptiveLogic(cfg)
    flags = [logic.step(0.1)[1] for _ in range(35)]  # absurdly low from the start
    assert all(f is None for f in flags)
