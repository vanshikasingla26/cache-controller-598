"""Quick start: miniature versions of the main results in about two minutes, on public data only.

  pip install -r requirements.txt
  python scripts/quickstart.py

Uses the small real CloudPhysics sample shipped with libCacheSim (downloaded to data/ if missing,
113,872 requests) and one synthetic server. It is a smoke test of the whole pipeline, not a replacement for
the full experiments: one short trace cannot support the confidence intervals reported in the README.
"""
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shiftcache import simulate, split_hit_rates, transition_mask  # noqa: E402
from shiftcache.policies import ARC, Belady, next_use  # noqa: E402
from shiftcache.realdata import SAMPLE_URL, load_oracle_general, read_records  # noqa: E402
from scripts.lru_gate_control import LRUGatedARC  # noqa: E402
from scripts.policy_selection_headroom import POL, native  # noqa: E402
from scripts.real_experiment import SYNTH_FIXED, detect_fixed_from_overlaps, mask_from, overlaps  # noqa: E402
from scripts.workloads import random_server  # noqa: E402

OUT = "results/quickstart"


def main():
    import libcachesim as lcs
    T0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    path = "data/cloudphysics_sample.oracleGeneral.bin"
    if not os.path.exists(path):
        urllib.request.urlretrieve(SAMPLE_URL, path)
    L = []
    tr = load_oracle_general(path, name="cloudphysics_sample")
    cap = max(1, int(tr.footprint * 0.05))
    L.append(f"[1] real trace: {len(tr):,} requests, {tr.footprint:,} unique objects, cache = {cap:,} objects (5%)")

    ours = simulate(tr, ARC(cap)).hit_rate
    theirs = native(lcs, path, "ARC", cap)
    L.append(f"[2] ARC check: this repo {ours:.4f} vs libCacheSim {theirs:.4f} (difference {abs(ours - theirs):.1e})")

    hr = {p: native(lcs, path, p, cap) for p in POL}
    hr["MIN"] = simulate(tr, Belady(cap, next_use(tr.keys))).hit_rate
    L.append("[3] hit rates: " + ", ".join(f"{p} {v:.3f}" for p, v in sorted(hr.items(), key=lambda x: -x[1])))

    rec = read_records(path)
    h = len(rec) // 2
    files = [os.path.join(OUT, "h1.bin"), os.path.join(OUT, "h2.bin")]
    rec[:h].tofile(files[0])
    rec[h:].tofile(files[1])
    h1 = {p: native(lcs, files[0], p, cap) for p in POL}
    h2 = {p: native(lcs, files[1], p, cap) for p in POL}
    pick = max(POL, key=h1.get)
    L.append(f"[4] policy selection: best on first half = {pick}; on the second half it scores "
             f"{(h2[pick] - h2['ARC']) * 100:+.2f} points vs ARC (oracle {(max(h2.values()) - h2['ARC']) * 100:+.2f})")
    for f in files:
        os.remove(f)

    syn = random_server(100)
    c2 = max(1, int(syn.footprint * 0.05))
    gt = transition_mask(len(syn), syn.transitions, 50_000)
    st, en = detect_fixed_from_overlaps(overlaps(syn, SYNTH_FIXED.window, SYNTH_FIXED.ref_windows), SYNTH_FIXED)
    det = mask_from(st, en, len(syn))
    base = simulate(syn, ARC(c2))
    b = split_hit_rates(base, gt)
    parts = []
    for name, gate in (("true shift", gt), ("detector", det)):
        r = simulate(syn, LRUGatedARC(c2, k=8, gate=gate))
        s = split_hit_rates(r, gt)
        parts.append(f"{name}: {(s['transition_hr'] - b['transition_hr']) * 100:+.2f} after shifts, "
                     f"{(r.hit_rate - base.hit_rate) * 100:+.2f} overall")
    L.append(f"[5] synthetic server ({len(syn):,} accesses, {len(syn.transitions)} shifts), LRU-like action vs ARC: "
             + "; ".join(parts))
    L.append(f"\ndone in {time.time() - T0:.0f} s. Full experiments: see README 'Reproducing'.")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
