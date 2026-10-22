# cache-controller-598: shift-aware cache controller (CSE 598, Fall 2026)

Team project: Vanshika Singla and Shubham Harkare.

## Provenance
The simulator, the ARC / LRU / LFU / Belady implementations, the shift detector, the learned
re-ranker and the experiment pipeline below were written by Shubham Harkare and imported into this
repository from his repo (snapshot `feature_002`; its status section is dated Sep 29, 2026).
Everything added afterwards is in `git log`. Additions so far:

* `scripts/lru_gate_control.py`: control experiment, ARC that behaves like LRU while the shift gate
  is on (no model), same held-out servers and detector as `scripts/experiment.py`.
* `results/lru_gate/`, `results/full/experiment_ci.txt`: control results and the 95% bootstrap CIs
  computed from `results/full/experiment_raw.csv` (the original script only printed them).

## Known issues in the imported snapshot
* The `OracleTailARC` docstring and the headroom text call the oracle a ceiling no learned
  re-ranker can beat. `gbt[det]` beats `Oracle-X[det]` on all 8 held-out servers, so it is a greedy
  per-decision oracle, not a ceiling.
* All results are on synthetic servers from one generator family. On the real CloudPhysics sample
  from libCacheSim (113,872 requests) the detector tuned on synthetic data flags ~90% of accesses.
  It needs per-trace calibration before any real-trace claim.

---

# shiftcache: Shift-Aware Cache Controller (CSE 598, Fall 2026)

Trace-driven block-cache simulator for the shift-aware controller project:
a shift detector that reads only the raw request stream, plus a learned
re-ranker that overrides ARC's eviction choice only while a shift is flagged.

## Layout
```
shiftcache/
  trace.py        MSR-Cambridge loader (4KB block splitting), synthetic + spliced traces w/ ground truth
                  synthetic segment kinds: zipf, scan, uniform, mix (hot zipf set + background scan)
  simulator.py    simulate(), per-window hit rates, ARC p trace, latency model, transition/steady split
  detector.py     ShiftDetector (reuse-gap histogram overlap vs reference windows, start/end triggers),
                  run_detector(), score_detection(), arc_adaptation_delay()
  models.py       FastLR, FastGBT: sklearn-trained, numpy inference (checked equal to sklearn)
  policies/
    basic.py      LRU, LFU (O(1), in-cache counts)
    arc.py        ARC (Megiddo & Modha); hooks: candidates(), choose_victim(); logs p + ghost hits
    belady.py     Belady MIN + next_use() (also the re-ranker label source)
    oracle.py     OracleTailARC (same-list tail) / OracleCrossListARC (T1+T2 tails)
    controlled.py ControlledARC: cross-list re-ranker, gating (off / always / mask / online detector),
                  training-data collection mode
scripts/
  run_baselines.py   LRU/LFU/ARC/Belady across cache sizes, CSV + per-window plots
  headroom.py        Belady gap + oracle re-ranker ceilings, transition vs steady
  demo_traces.py     synthetic spliced trace used before real data is in
  workloads.py       randomized multi-phase synthetic "servers" (TRAIN_SEEDS / TEST_SEEDS)
  detect_eval.py     RQ1 check: detector vs ARC p reaction time, per trace, with plots
  experiment.py      full pipeline: tune detector -> collect rows -> train -> evaluate gates -> CIs
tests/               pytest suite (25 tests); ARC and LRU verified bit-exact vs libCacheSim
```

## Setup
```
pip install numpy pandas matplotlib pytest scikit-learn libcachesim
python -m pytest -q tests          # expect 25 passed, 0 skipped
```
Run everything from the repo root (scripts import `shiftcache.*` and `scripts.*`).

## Data
MSR-Cambridge: SNIA IOTTA repository ("MSR Cambridge Traces"). Put files in `data/`
(e.g. `data/web_0.csv.gz`). Columns: Timestamp, Hostname, DiskNumber, Type, Offset, Size, ResponseTime.
CloudPhysics: via the cacheMon collection (libCacheSim format), not yet integrated.

## Run
Baselines and headroom:
```
python scripts/run_baselines.py --synthetic
python scripts/run_baselines.py --msr data/web_0.csv.gz --fractions 0.01 0.05 0.1
python scripts/headroom.py --synthetic --win 50000
python scripts/headroom.py --splice data/web_0.csv.gz data/src1_2.csv.gz --fraction 0.05
```
Detector (RQ1), ~1 min; writes `results/detector_<trace>_ref<R>.png`:
```
python scripts/detect_eval.py                    # reference = mean of previous 10 windows
python scripts/detect_eval.py --ref-windows 1    # proposal's literal consecutive-window rule
```
Full pipeline (RQ1-RQ5, held-out synthetic servers):
```
python -u scripts/experiment.py --quick --out results/quick | tee results/quick.log   # 3 train / 3 test
python -u scripts/experiment.py --out results/full | tee results/full.log             # 8 train / 8 test
```
Options: `--fraction` (cache size), `--k` (candidates per list), `--win` (ground-truth transition
window), `--w-max` (blend cap), `--models lr gbt`, `--skip-always gbt` (skip the slow always-on run),
`--collect-every` (subsample training decisions). Outputs: `experiment_raw.csv`,
`experiment_summary.csv`, `detection_vs_arc.csv`; `experiment_raw.partial.csv` is written after
each test server. Needs ~2-3 GB RAM.

## Pipeline
1. **Detector** (`detector.py`). Each 1,000-access window becomes a histogram of log2 reuse gaps
   plus a first-access bin. Overlap `sum(min(p, q))` against the mean of the previous `ref_windows`
   windows. Start: overlap < `start_thresh` for `persist` windows. End: after `min_hold` windows,
   overlap >= `end_thresh` for `persist` windows, or `max_hold` windows. The flag only changes at
   window boundaries. Tuned on TRAIN servers only.
2. **Candidates**: the k blocks nearest the LRU end of BOTH T1 and T2, in ARC's own order
   (chosen list's tail first).
3. **Features** (15, `FEATURES`): age, history frequency, last reuse gap, list, position, ARC rank,
   ghost-hit count, p/c, |T1|/c, recent B1/B2 ghost-hit and miss rates, time since gate on.
4. **Label**: candidate reused within `horizon` accesses (default = cache size), from `next_use()`.
   Offline only; never used at decision time.
5. **Decision**: `score = (1 - w) * arc_keep + w * p`, `w = w_max * |2p - 1|`; evict the lowest
   score. The victim goes to the ghost list it came from, so ARC's p adaptation stays consistent.
6. **Gates compared** (RQ4): always-on (HALP-style), random windows with the same lengths as the
   detector's, ground-truth windows, detector.

## Design decisions (record in the report)
- Requests split into 4KB blocks; block hit rate == byte hit rate.
- Write-allocate by default (`simulate(..., write_allocate=False)` for read-cache mode).
- Cache size = fraction of the trace's unique-block footprint.
- Latency: fixed per-block cost (default 100us hit / 5000us miss) -- tune + cite.
- Transition windows for evaluation come from ground truth (synthetic / spliced), never the detector.
- Belady admits every miss (no bypass), matching the policies it bounds.
- Detector compares against a reference of several past windows, not just the previous one:
  with `ref_windows=1` an abrupt shift gives a single-window dip that `persist` filters out, so
  the proposal's literal rule detects nothing (see `detector_*_ref1.png`).
- ARC's p swings and decays after a shift rather than settling. ARC's reaction time is measured
  as accesses until p covers half of its peak excursion; shifts where p moves < 5% of c count as
  "ARC did not react" (e.g. scans, where p does not move).
- Train/test split is by whole server (trace), never by pooled segments.
- Training rows come from detector-flagged windows with ARC making the decisions.
- Model inference uses numpy paths so overhead numbers reflect the model, not sklearn call cost.

## Status (Sep 29, 2026)
Done: simulator + baselines, oracle headroom, detector with tuning, re-ranker pipeline, gating
ablations, bootstrap CIs; all on SYNTHETIC traces.

Preliminary synthetic result (`--quick`, 3 held-out servers; too few for a reliable CI):
detector recall 0.93 / precision 1.00, median delay ~1.8K accesses vs ARC p half-swing ~16K;
GBT gated by the detector +1.72 pts transition hit rate vs ARC, steady state not hurt; always-on
and random gating do not reproduce the gain; LR (AUC 0.60) has no effect.

Open issues:
- GBT[det] roughly equals Oracle-X[det]: run leakage / feature-importance checks before claiming.
- Confidence weight `|2p - 1|` treats p near 0 as confident; with a ~6% positive rate the model
  mostly overrides ARC. Needs a base-rate-relative confidence variant.
- Detector re-trigger for shifts arriving while the flag is already on (not implemented).
- Scan-to-scan shifts are invisible to the reuse-gap signal (all accesses are first-time).

Next: MSR-Cambridge traces (baselines, spliced-pair headroom, real-data mode in `experiment.py`),
full 8/8 synthetic run, CloudPhysics, libCacheSim baselines (LeCaR, CACHEUS, S3-FIFO, SIEVE),
cache-size and k sweeps.