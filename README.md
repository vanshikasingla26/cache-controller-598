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

* `shiftcache/detector_adaptive.py`: self-calibrating detector (thresholds are median - z * robust
  scale of the trace's own recent overlap values, not fixed numbers). Drop-in for `ShiftDetector`.
  Decision logic is a pure function of the overlap series, tested online == offline.
* `shiftcache/realdata.py`, `scripts/fetch_sample_trace.py`: loader for libCacheSim `oracleGeneral`
  traces and a download of the small real CloudPhysics sample (113,872 requests).
* `scripts/detector_eval.py`, `results/detector_eval/`: fixed vs adaptive detector.
* `scripts/lcs_baselines.py`, `results/baselines/`: LRU, LFU, ARC, LeCaR, CACHEUS, S3-FIFO, SIEVE,
  W-TinyLFU, LHD (libCacheSim) plus MIN on any trace.
* `tests/test_detector_adaptive.py`.

## Learned cheap policy selector (step 4, `scripts/learned_selector.py`)
Protocol fixed before results: same 33 traces, same train/test split and second-half hit rates as the
policy-selection section. Features = one pass over the last 500k requests (unique ratio, one-hit share,
top-1%/10% access share, sequential share, first-access share, inter-reference-gap histogram); no cache
simulation. One regressor per policy predicts its gain over ARC on the next window; trained on 105 rows
from 21 TRAIN traces; model chosen by leave-one-trace-out CV (shallow gradient boosting).

| 12 held-out traces, second half vs ARC (points) | mean [95% CI] | positive |
|---|---|---|
| learned selector (features of 500k requests) | +0.46 [+0.11, +0.82] | 7/12 |
| always S3-FIFO | +0.23 [-0.25, +0.66] | 9/12 |
| pick from first half (simulates 9 policies on 1.5M requests) | +0.70 [+0.34, +1.04] | 10/12 |
| oracle per trace | +0.78 [+0.49, +1.07] | 11/12 |

Learned minus always-S3-FIFO: +0.23 [-0.36, +0.94] (not significant). Cost per decision: 0.56 s of
Python feature extraction vs 10.4 s for 9 native libCacheSim simulations (about 18x; not a like-for-like
comparison, different languages and window lengths).
**Reading, honestly:** the learned selector beats ARC on held-out traces with a CI above zero at a fraction
of the decision cost, and recovers about 60% of the simulation selector's gain. It is NOT reliably better
than always using S3-FIFO, and in leave-one-trace-out CV on TRAIN both learned models were below ARC
(ridge -0.16, GBR -0.08) while always-S3-FIFO was +0.05. Evidence is mixed and data-limited (105 rows).

## Policy selection across four real datasets (`scripts/policy_selection_headroom.py`, fig5)
Data: 37 real block traces, first 3M requests each: MSR-Cambridge 13, CloudPhysics 12, Alibaba 8,
Meta 4 (`results/real/baselines_*.csv`). Our ARC equals libCacheSim's ARC on all 37 (max difference
6e-17). Excluded by the fixed >90% one-hit rule: msr:proj_1, msr:proj_2, msr:src1_1, msr:usr_1. 33 traces remain.
Split fixed in advance: within each dataset traces are sorted by name and assigned to train/test
alternately (related neighbours cp w100-w106 and Alibaba 165/166 grouped): 21 train, 12 test.
Each trace is cut in two halves; all numbers are hit rate on the SECOND half, cache = 5% of unique
objects, libCacheSim native, 9 online policies.

| on 12 held-out test traces (points vs ARC) | mean [95% CI over traces] | positive |
|---|---|---|
| always S3-FIFO (best single policy on TRAIN) | +0.23 [-0.25, +0.66] | 9/12 |
| pick the policy that was best on the first half (causal, no ML) | **+0.70 [+0.34, +1.04]** | 10/12 |
| oracle: best policy per trace (ceiling) | +0.78 [+0.49, +1.07] | 11/12 |

Pick-from-first-half vs always-S3-FIFO: +0.47 [-0.02, +1.09]. Per dataset (all traces), pick vs
always-S3-FIFO: Meta +2.59, CloudPhysics +0.08, MSR +0.03, Alibaba -0.06.

**Reading:** choosing the policy per trace from its own recent history beats ARC on held-out traces from
four companies, and the simple causal rule already reaches about 90% of the per-trace ceiling. So a
learned per-trace selector could add at most about 0.08 points on these traces. Selecting caches with
simulations of candidate configurations is known (e.g. miniature simulations, Waldspurger et al., USENIX
ATC 2017), so the mechanism is not new. Caveats: 12 test traces; the selector here simulates all 9
policies on 1.5M requests (its cost is not measured; sampled simulation is the known way to make it cheap);
one decision per trace, no re-selection over time.

## Real MSR-Cambridge results (final for this run; see results/real/)
Data: 13 MSR-Cambridge volumes (libCacheSim `oracleGeneral` .zst). Unit-size objects, cache = 5% of
unique objects. Exclusion rule fixed before any result: volumes whose slice is more than 90% one-hit
objects (pure scans) are dropped from train and test (`src1_0`, `web_2`, `proj_1`, `proj_2`, `usr_1`),
leaving 4 train volumes (hm_0, prn_0, prn_1, src1_1) and 4 test volumes (proj_0, proj_4, prxy_0, usr_2),
split by server family. Detectors and the re-ranker are trained on TRAIN splices only.

1. **Baselines (3M requests per volume, `results/real/baselines_msr.csv`).** Our ARC equals libCacheSim's
   ARC on all 13 volumes (difference <= 7e-17). No policy wins everywhere (5% cache: S3-FIFO best on
   prn_1, W-TinyLFU on prn_0, ARC on proj_4). Six volumes are scan-dominated, hit rate < 7% for every policy.
2. **Spliced pairs with a known shift (12 pairs, 4 destination volumes).** Points of hit rate vs ARC:

   | gate | model | after shift | rest | overall |
   |---|---|---|---|---|
   | known shift | learned (GBT) | +2.64 | +0.03 | +0.16 [+0.05,+0.33] |
   | known shift | none (LRU-like) | +2.56 | -0.01 | +0.12 [+0.09,+0.16] |
   | detector | learned (GBT) | +1.21 | -0.95 | -0.84 [-1.27,-0.60] |
   | detector | none (LRU-like) | +1.89 | -1.87 | -1.68 [-2.58,-1.19] |
   | random, same lengths | learned (GBT) | +1.96 | -0.32 | -0.21 [-0.44,+0.02] |

   The detector-gated model beats ARC overall on only 4 of 12 pairs.
3. **Natural (unspliced) test volumes (4).** Overall vs ARC: LRU-like gate -1.74 [-3.91,-0.37],
   learned GBT gate -1.88 [-3.91,-0.43]; worse than ARC on all four volumes (proj_4: 0.143 -> 0.094).
4. **Detector on test splices.** Synthetic-tuned fixed: found 2 of 12 shifts, on for 79% of accesses.
   Real-tuned fixed: 1 of 12. Self-calibrating: 5 of 12, precision 0.11, on for 40%. Train F1 only 0.23-0.31.

**Reading:** the benefit exists when the controller knows the shift (about +2.6 points in the 50k
accesses after it, no loss elsewhere), and most of it comes from acting recency-like, not from the model
(+0.08 more). With a detector built from the reuse-gap statistic, the controller loses to ARC on real
traces. The detector is the bottleneck. On splices the model degrades more gently than the model-free
rule when the gate is wrong (-0.84 vs -1.68), but not on natural volumes (-1.88 vs -1.74, overlapping CIs).
The detector-chosen windows were worse places to act than random ones (steady -0.95 vs -0.32).

Caveats: shifts are artificial (a full swap between two real volumes); only 4 destination/test volumes,
so CIs are wide; the gate (fixed detector) was chosen by a rule set beforehand (higher TRAIN F1) even
though the self-calibrating one looks better on test; slices are the first 0.5M-3M requests of each volume.

## Hit-rate-tuned gate and the ceiling (`scripts/real_gate_tuning.py`, `results/real/gate_tuning_*`)
The F1-tuned detector lost to ARC, so the gate was re-tuned for hit rate on TRAIN only (14 traces,
174 configs) with a label-free proxy: gain = sum over gated accesses of (LRU hit - ARC hit) / n.
* **Ceiling of LRU-like switching.** Gating exactly the 1000-access windows where LRU beats ARC would
  gain only +0.74 (train), +0.74 (test splices), +0.68 (test natural) points overall. Always-LRU loses
  0.7 to 3.2 points.
* **Best tuned gate on TRAIN:** proxy +0.10 points, positive on only 36% of train traces.
* **Proxy validity:** correlation 0.98 with the real LRU-gated simulation on 32 TEST runs, mean absolute
  error 0.51 points (the real number is lower: switching eviction behavior costs hit rate).
* **Real LRU-gated ARC on TEST with the hit-rate-tuned gate:** splices -0.81 [-1.27,-0.56] (F1-tuned
  gate: -1.68), natural -1.49 [-3.23,-0.16] (F1-tuned gate: -1.74). Better, still worse than ARC.
So on these MSR traces, an "act like LRU around detected shifts" controller does not beat plain ARC even
with a gate tuned for hit rate; the best any such gate could do is below one point.

## Controlled one-dimension shifts on real volumes (`scripts/real_controlled_shifts.py`, fig4)
First 1M requests of each of the 8 usable MSR volumes; at request 500k ONE change with a controlled
magnitude: popularity (remap 25/50/100% of the post-shift hot set = blocks covering 50% of post-shift
accesses), scan (10/30/50% of requests become a sequential scan of new blocks), tenant (25/50% of
requests interleaved from another real volume). Gate = the hit-rate-tuned one (no tuning here).
Means over the 4 held-out test volumes:
* ARC's p takes 70k-230k accesses to half-swing, yet the MIN - ARC gap right after the shift is not
  larger than in steady state: transition-specific avoidable misses <= 0 in 6 of 8 scenarios (+98 and +44
  misses per 50k accesses in the other two). ARC's slow adaptation does not cause extra misses here.
* LRU-like action with the TRUE change point: -0.5 to -2.0 points after scans and partial hot-set swaps,
  +0.5 / +0.8 for tenant mixes, +0.1 for a full hot-set swap.
* The hit-rate-tuned detector found 1 of 32 test shifts (it is very conservative).
* MIN is 5-10 points above ARC in the 50k accesses after each shift: room exists, but it is not shift-specific.

## Earlier findings (synthetic and the small CloudPhysics sample)
1. **Synthetic, 8 held-out servers.** Adaptive detector (tuned on the 8 train servers only) matches
   the fixed one: precision 1.00, recall 0.97, mean delay 1,667 vs 1,757 accesses, and it is on for
   16.9% of accesses instead of 29.9%.
2. **The CloudPhysics sample cannot validate a shift detector.** It has 48,974 unique objects in
   113,872 requests, so most of it is cold start (share of first-ever accesses falls from 66% to 21%
   with no shift). Its overlap statistic has robust scale 0.17 (synthetic: about 0.01), the lowest
   overlap anywhere is only 2.9 scale units below the median, and a namespace splice changes the
   overlap series by about 1 scale unit. The adaptive detector tuned on synthetic data picked
   z_start = 8, whose threshold is negative on this trace, so it never fires there. That is not
   evidence that it transfers. Real validation needs long MSR traces and splices of different servers.
3. **Real-data baselines (CloudPhysics sample, object-count caches).** Our ARC equals libCacheSim's
   ARC exactly (0.1725 / 0.1886 / 0.2272 at 1% / 5% / 10% of unique objects). ARC is the best online
   policy at 1% and 5%; S3-FIFO is best at 10% (0.2447). CACHEUS 0.1797 < ARC 0.1886 at 5%.
   Clairvoyant MIN reaches 0.2073 / 0.2968 / 0.3710. See `results/baselines/`.

## Known issues in the imported snapshot
* The `OracleTailARC` docstring and the headroom text call the oracle a ceiling no learned
  re-ranker can beat. `gbt[det]` beats `Oracle-X[det]` on all 8 held-out servers, so it is a greedy
  per-decision oracle, not a ceiling.
* The detector-gated GBT gain (+2.50 points in transition windows, `results/full/experiment_ci.txt`)
  is on synthetic servers from one generator family. A model-free control (ARC acting like LRU
  while the gate is on) reaches +1.52, so the learned model's increment is about +0.98 points.

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