from scripts.real_experiment import detect_fixed_from_overlaps, mask_from, overlaps
from scripts.workloads import random_server
from shiftcache.detector import DetectorConfig, run_detector


def test_offline_fixed_detector_equals_class():
    tr = random_server(100)
    for cfg in (DetectorConfig(window=1000, start_thresh=0.8, end_thresh=0.8, persist=2,
                               ref_windows=5, min_hold=50, max_hold=150),
                DetectorConfig(window=1000, start_thresh=0.6, end_thresh=0.6, persist=3, ref_windows=10)):
        _, d = run_detector(tr.keys, cfg)
        st, en = detect_fixed_from_overlaps(overlaps(tr, cfg.window, cfg.ref_windows), cfg)
        assert st == d.starts and en == d.ends


def test_mask_from():
    m = mask_from([10, 50], [20], 60)
    assert m[10:20].all() and not m[20:50].any() and m[50:].all()
