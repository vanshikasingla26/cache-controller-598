"""Loaders for real traces that are reachable without registration.

load_oracle_general: libCacheSim 'oracleGeneral' binary format, 24 bytes/record,
little-endian, packed: uint32 clock_time, uint64 obj_id, uint32 obj_size,
int64 next_access_vtime (1-indexed; -1 if never again). The format was checked
against the CloudPhysics sample: next_access_vtime == own recomputed next index + 1
on every record.

Object sizes are ignored: all policies here are object-count based.
"""
from __future__ import annotations

import numpy as np

from .trace import Trace

ORACLE_GENERAL = np.dtype([("t", "<u4"), ("id", "<u8"), ("size", "<u4"), ("nxt", "<i8")])
SAMPLE_URL = ("https://raw.githubusercontent.com/1a1a11a/libCacheSim/develop/data/"
              "cloudPhysicsIO.oracleGeneral.bin")


def load_oracle_general(path: str, name: str | None = None, max_requests: int | None = None) -> Trace:
    a = np.fromfile(path, dtype=ORACLE_GENERAL)
    if max_requests:
        a = a[:max_requests]
    n = len(a)
    return Trace(a["id"].astype(np.int64), np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool),
                 name=name or path.rsplit("/", 1)[-1])
