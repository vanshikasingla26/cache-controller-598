import numpy as np
import pandas as pd

from scripts.learned_selector import window_features
from scripts.policy_selection_headroom import POL, fold_rows, split
from scripts.real_baselines import trace_name
from shiftcache.realdata import ORACLE_GENERAL, read_records


def test_window_features_toy():
    f = window_features(np.array([1, 2, 3, 1, 2, 1, 9, 10, 11], dtype=np.int64), {}, 0)
    assert abs(f["unique_ratio"] - 6 / 9) < 1e-9
    assert abs(f["one_hit"] - 4 / 6) < 1e-9          # 3, 9, 10, 11 seen once
    assert abs(f["seq_share"] - 5 / 8) < 1e-9        # 1->2, 2->3, 1->2, 9->10, 10->11
    assert abs(f["first_access"] - 6 / 9) < 1e-9
    assert abs(f["gap2"] - 3 / 9) < 1e-9             # gaps 3, 3, 2 all fall in log2 bin 2


def _fake(n_per_ds=6):
    rows = []
    for ds in ("msr", "cp"):
        for i in range(n_per_ds):
            r = {"dataset": ds, "trace": f"{ds}:t{i}", "n": 10, "one_hit": 0.5, "capacity": 5}
            for p in POL:
                r[f"h1_{p}"] = 0.5 + (0.1 if p == "LHD" else 0)       # LHD best on first half
                r[f"h2_{p}"] = 0.5 + (0.05 if p == "LHD" else 0) + (0.02 if p == "S3FIFO" else 0)
            rows.append(r)
    return pd.DataFrame(rows)


def test_two_folds_test_each_trace_exactly_once():
    df = _fake()
    t0 = set(split(df, 0).query("split == 'test'").trace)
    t1 = set(split(df, 1).query("split == 'test'").trace)
    assert t0.isdisjoint(t1) and t0 | t1 == set(df.trace)


def test_fold_rows_pick_uses_first_half_choice():
    r = fold_rows(_fake(), 0)
    assert np.allclose(r.pick_first_half, 5.0)       # LHD: +0.05 over ARC on the second half
    assert np.allclose(r.oracle, 5.0)


def test_trace_name():
    assert trace_name("x/msr_hm_0_oracleGeneral.zst", "msr") == "hm_0"
    assert trace_name("x/w100.oracleGeneral.bin.zst", "cp") == "cp:w100"
    assert trace_name("x/alibabaBlock_0.oracleGeneral.zst", "ali") == "ali:alibabaBlock_0"


def test_read_records_bin_and_zst(tmp_path):
    import zstandard
    a = np.zeros(5, dtype=ORACLE_GENERAL)
    a["id"] = [5, 6, 5, 7, 5]
    a["nxt"] = [3, -1, 5, -1, -1]
    b = tmp_path / "t.bin"
    a.tofile(b)
    z = tmp_path / "t.zst"
    z.write_bytes(zstandard.ZstdCompressor().compress(b.read_bytes()))
    assert np.array_equal(read_records(str(b)), a)
    assert np.array_equal(read_records(str(z), 3), a[:3])
