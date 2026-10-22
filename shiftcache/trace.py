"""Trace loading and generation.

Everything downstream works on a flat *block-access* sequence:
    keys    : np.int64 array, one entry per 4KB block touched
    req_ids : np.int64 array, which original request each block access came from
    is_write: np.bool_ array, per block access

Splitting variable-size I/O into fixed 4KB blocks keeps ARC's uniform-page
assumption valid; block hit rate is then also the byte hit rate.
"""
from __future__ import annotations

import gzip

from dataclasses import dataclass, field

import numpy as np

BLOCK_SIZE = 4096


@dataclass
class Trace:
    keys: np.ndarray
    req_ids: np.ndarray
    is_write: np.ndarray
    name: str = "trace"
    # Ground-truth transition points (block-access indices), only known for synthetic traces.
    transitions: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def n_requests(self) -> int:
        return int(self.req_ids[-1]) + 1 if len(self.req_ids) else 0

    @property
    def footprint(self) -> int:
        """Number of unique blocks: cache sizes are set as fractions of this."""
        return int(np.unique(self.keys).size)

    def reads_only(self) -> "Trace":
        m = ~self.is_write
        _, new_req = np.unique(self.req_ids[m], return_inverse=True)
        return Trace(self.keys[m], new_req.astype(np.int64), self.is_write[m], self.name + "-R")

    def head(self, n: int) -> "Trace":
        return Trace(self.keys[:n], self.req_ids[:n], self.is_write[:n], self.name,
                     [t for t in self.transitions if t < n])


# --------------------------------------------------------------------------- MSR
def load_msr(path: str, disk: int | None = None, max_requests: int | None = None) -> Trace:
    """Load an MSR-Cambridge CSV (optionally .gz).

    Columns: Timestamp, Hostname, DiskNumber, Type, Offset, Size, ResponseTime
    Each request covering bytes [Offset, Offset+Size) is split into 4KB block keys.
    Keys are namespaced by disk number so multi-disk files do not collide.
    """
    import pandas as pd

    df = pd.read_csv(
        path, header=None, nrows=max_requests,
        names=["ts", "host", "disk", "type", "offset", "size", "resp"],
        dtype={"ts": np.int64, "host": str, "disk": np.int64, "type": str,
               "offset": np.int64, "size": np.int64, "resp": np.int64},
        compression="gzip" if path.endswith(".gz") else None,
    )
    if disk is not None:
        df = df[df["disk"] == disk]
    df = df[df["size"] > 0].sort_values("ts", kind="stable").reset_index(drop=True)

    first = df["offset"].to_numpy() // BLOCK_SIZE
    last = (df["offset"].to_numpy() + df["size"].to_numpy() - 1) // BLOCK_SIZE
    nblocks = (last - first + 1).astype(np.int64)

    req_ids = np.repeat(np.arange(len(df), dtype=np.int64), nblocks)
    # offset within request: 0,1,2,... for each request
    starts = np.repeat(np.cumsum(nblocks) - nblocks, nblocks)
    within = np.arange(nblocks.sum(), dtype=np.int64) - starts
    blocks = np.repeat(first, nblocks) + within
    disks = np.repeat(df["disk"].to_numpy(), nblocks)
    keys = (disks << 48) | blocks  # namespace by disk
    is_write = np.repeat(df["type"].str.lower().eq("write").to_numpy(), nblocks)

    name = path.split("/")[-1].split(".")[0]
    return Trace(keys.astype(np.int64), req_ids, is_write, name)


# --------------------------------------------------------------------- synthetic
def _zipf_segment(rng, n, universe, alpha, offset):
    ranks = np.arange(1, universe + 1, dtype=np.float64)
    p = ranks ** (-alpha)
    p /= p.sum()
    perm = rng.permutation(universe)  # popularity order unrelated to key id
    return offset + perm[rng.choice(universe, size=n, p=p)]


def _scan_segment(n, universe, offset):
    return offset + (np.arange(n) % universe)


def synthetic(segments: list[dict], seed: int = 0, name: str = "synthetic") -> Trace:
    """Build a spliced trace with *known* transition points.

    segments: list of dicts, e.g.
        {"kind": "zipf", "n": 200_000, "universe": 20_000, "alpha": 0.9, "ws": 0}
        {"kind": "scan", "n": 50_000,  "universe": 60_000, "ws": 2}
        {"kind": "mix",  "n": 100_000, "universe": 20_000, "alpha": 0.9, "ws": 0,
         "scan_frac": 0.3, "scan_universe": 200_000}      # hot set + background scan
    'ws' picks a working-set id; segments with the same ws share keys, so you can
    shift away and come back.
    """
    rng = np.random.default_rng(seed)
    parts, transitions, pos = [], [], 0
    for i, s in enumerate(segments):
        off = int(s.get("ws", i)) * 10**9
        if s["kind"] == "zipf":
            seg = _zipf_segment(rng, s["n"], s["universe"], s.get("alpha", 0.9), off)
        elif s["kind"] == "scan":
            seg = _scan_segment(s["n"], s["universe"], off)
        elif s["kind"] == "mix":
            # hot zipf set with a background sequential scan interleaved (e.g. backup over OLTP)
            z = _zipf_segment(rng, s["n"], s["universe"], s.get("alpha", 0.9), off)
            sc = _scan_segment(s["n"], s.get("scan_universe", 10 * s["universe"]), off + s["universe"])
            seg = np.where(rng.random(s["n"]) < s.get("scan_frac", 0.3), sc, z)
        elif s["kind"] == "uniform":
            seg = off + rng.integers(0, s["universe"], size=s["n"])
        else:
            raise ValueError(s["kind"])
        if i > 0:
            transitions.append(pos)
        parts.append(seg.astype(np.int64))
        pos += s["n"]
    keys = np.concatenate(parts)
    n = len(keys)
    return Trace(keys, np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool), name, transitions)


def splice(a: Trace, b: Trace, a_len: int, b_len: int, name: str | None = None) -> Trace:
    """Concatenate the first a_len accesses of a with the first b_len of b.

    Used on real traces (two MSR volumes) to get a real-workload shift with a known
    ground-truth location. Keys from b are offset so the volumes don't collide.
    """
    ka, kb = a.keys[:a_len], b.keys[:b_len] + (1 << 62)
    ra = a.req_ids[:a_len]
    rb = b.req_ids[:b_len] - b.req_ids[0] + (ra[-1] + 1 if len(ra) else 0)
    return Trace(np.concatenate([ka, kb]), np.concatenate([ra, rb]),
                 np.concatenate([a.is_write[:a_len], b.is_write[:b_len]]),
                 name or f"{a.name}+{b.name}", [a_len])
