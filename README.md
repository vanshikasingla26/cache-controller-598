# cache-controller-598: shift-aware caching and policy selection on real block-storage traces

Can a cache react to workload shifts faster than an adaptive policy like ARC, and if not, where is the
real room to improve? This repository contains the simulator, detectors, learned components,
experiments and results. Unless noted, numbers are on real block-storage traces, in percentage points
of hit rate versus ARC, with 95% bootstrap confidence intervals over traces or volumes.

## Results at a glance
1. **Policy selection, 37 real traces from 4 sources** (MSR-Cambridge, CloudPhysics, Alibaba, Meta; 33
   after excluding pure scans). Two folds, so every trace is a test trace exactly once (fig5):
   | selector (second half of each trace) | vs ARC | positive |
   |---|---|---|
   | pick the policy that was best on the first half (simulates 9 policies) | **+0.63 [+0.28, +1.00]** | 21/33 |
   | always S3-FIFO (best single policy on TRAIN in both folds) | +0.29 [-0.15, +0.74] | 19/33 |
   | learned cheap selector (500k-request features, no simulation) | +0.25 [-0.23, +0.68] | 18/33 |
   | oracle: best policy per trace | +0.93 [+0.62, +1.29] | 26/33 |

   The learned selector decides about 19x faster than simulating, but it is no better than always using
   S3-FIFO (difference -0.04 [-0.70, +0.54]); its first-fold result (+0.45) did not replicate.
2. **Shift-aware controller on real MSR traces** (12 spliced pairs, 4 held-out volumes; fig2, fig3). With
   the true shift time: +2.6 points in the 50k accesses after the shift (+0.16 overall). With a real
   detector it loses to ARC: -0.84 overall (learned re-ranker), -1.68 (model-free), -0.81 with a gate tuned
   for hit rate. Even perfect window-by-window LRU-like switching would gain only about +0.7 overall.
3. **Controlled one-dimension shifts injected into real volumes** (fig4): ARC is not further behind the
   clairvoyant optimum right after a shift than in steady state, and an LRU-like action hurts after scans.
4. **Baselines** (fig1, `results/real/baselines_*.csv`): our ARC is bit-identical to libCacheSim's ARC on
   all 37 traces; no single policy wins everywhere.
5. **Synthetic servers** (one generator family, 8 held-out servers): the detector-gated learned re-ranker
   gains +2.50 points in transition windows; a model-free control gets +1.52 of that.

## Reproducing
```bash
pip install -r requirements.txt
python -m pytest -q                       # 38 tests, about 30 seconds
```
**Data.** Download the `oracleGeneral` files from https://github.com/cacheMon/cache_dataset (rows
"Microsoft Cambridge", "CloudPhysics", "Alibaba Block", "MetaStorage", column "OracleGeneral format")
and place them as:
```
data/msr/msr_<volume>_oracleGeneral.zst      hm_0 prn_0 prn_1 proj_0 proj_1 proj_2 proj_4 prxy_0
                                             src1_0 src1_1 usr_1 usr_2 web_2
data/cp/w<id>.oracleGeneral.bin.zst          w32 w40 w41 w42 w43 w100-w106
data/ali/alibabaBlock_<id>.oracleGeneral.zst 0 11 14 15 100 150 165 166
data/meta/block_traces_<i>.oracleGeneral.bin.zst  1-4
```
`python scripts/make_slice.py <file> <out.bin> --n 3000000` cuts a large trace down if needed.

**Run** (single core; approximate times):
```bash
python -u scripts/real_baselines.py --dir data/msr --out results/real                     # ~15 min
for t in cp ali meta; do python -u scripts/real_baselines.py --dir data/$t --pattern '*.zst' --tag $t --out results/real; done  # ~35 min
python -u scripts/real_experiment.py --dir data/msr --out results/real --side 500000 \
       --natural-n 1500000 --train-pairs 10 --test-pairs 12 --skip-always                  # ~70 min
python -u scripts/real_gate_tuning.py --dir data/msr --out results/real                   # ~7 min
python -u scripts/real_controlled_shifts.py --dir data/msr --out results/real             # ~15 min
python -u scripts/policy_selection_headroom.py --out results/real                         # ~15 min
python -u scripts/learned_selector.py --out results/real                                  # ~15 min
python scripts/make_figures.py --out results/figures
python -u scripts/experiment.py && python -u scripts/lru_gate_control.py                  # synthetic
```
All scripts write their raw CSVs and summaries to `results/`; long runs save after every trace and
resume where they stopped.

## Baseline configuration
libCacheSim 0.3.5 Python bindings with default parameters for every policy: LRU, LFU, ARC, LeCaR, CACHEUS,
S3-FIFO, SIEVE, W-TinyLFU, LHD, and Belady (MIN, offline). Objects are unit size (`ignore_obj_size`), the
cache holds 5% of the unique objects of the trace slice (1% and 10% also reported), caches start cold, and
each trace is cut to its first N requests (3M unless stated). The controller experiments use this
repository's Python ARC / LRU / LFU / Belady; ARC is checked against libCacheSim on every trace.

## Learned cheap policy selector (`scripts/learned_selector.py`, `results/real/selector_*`)
Protocol fixed before results; same 33 traces, splits and second-half hit rates as the policy-selection
section. Features = one pass over the last 500k requests (unique ratio, one-hit share, top-1%/10% access
share, sequential share, first-access share, inter-reference-gap histogram); no cache simulation. One
regressor per policy predicts its gain over ARC on the next window; the model (ridge or shallow gradient
boosting) is chosen per fold by leave-one-trace-out CV on that fold's TRAIN traces.

| second half vs ARC (points) | fold 0 (12 test traces) | both folds (33 test traces) |
|---|---|---|
| learned selector | +0.45 [+0.11, +0.81], 7/12 | +0.25 [-0.23, +0.68], 18/33 |
| always S3-FIFO | +0.23 [-0.25, +0.66], 9/12 | +0.29 [-0.15, +0.74], 19/33 |
| pick from first half (simulation) | +0.70 [+0.34, +1.04], 10/12 | +0.63 [+0.28, +1.00], 21/33 |
| oracle per trace | +0.78 [+0.49, +1.07], 11/12 | +0.93 [+0.62, +1.29], 26/33 |

Learned minus always-S3-FIFO over both folds: -0.04 [-0.70, +0.54]; the learned choice equals the
simulation's choice on 11/33 traces. Leave-one-trace-out CV on TRAIN: fold 0 ridge -0.16, GBR -0.10
(always-S3-FIFO +0.05); fold 1 ridge +0.44, GBR +0.38 (always-S3-FIFO +0.14). The fold-0 test number moves
slightly between runs (+0.46 vs +0.45) because subsampled boosting depends on training-row order.

**Hidden costs.** Training (8 regressors): 1.1 s (fold 0), 0.01 s (fold 1). Inference: 0.5 ms per decision.
Feature extraction: 0.59 s per 500k requests in Python, vs 11.2 s for simulating 9 policies on 1.5M
requests natively (about 19x; different languages and window lengths, so only indicative). State: one
last-seen index per object seen so far, median 0.74M entries at decision time (about 12 MB at 16 bytes per
entry, at most 24 MB; `results/real/selector_state.csv`).

**Reading:** a cheap learned selector is feasible, but with 60-105 training rows it does not beat always
using S3-FIFO. The simulation-based selector is the robust positive result.

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

Two-fold check (every trace tested once, `results/real/policy_selection_2fold.csv`): pick-from-first-half
+0.63 [+0.28, +1.00] (21/33), always-S3-FIFO +0.29 [-0.15, +0.74], oracle +0.93 [+0.62, +1.29];
pick minus always-S3-FIFO +0.34 [-0.01, +0.72]. Per dataset (pick vs ARC): Alibaba +1.03, Meta +1.31,
MSR +0.60, CloudPhysics +0.15.
Pick-from-first-half vs always-S3-FIFO (fold 0 only): +0.47 [-0.02, +1.09]. Per dataset (all traces), pick vs
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

## Always-on vs triggered re-ranker on real data, and its overhead (`scripts/real_always_overhead.py`)
Re-ranker (GBT) trained on TRAIN splices exactly as in `real_experiment.py` (1.56M rows, 11.8 s, 0.93 MB
model, saved as `results/real/models_real.pkl`); one TEST splice per held-out destination volume (4 pairs).

| variant (points vs ARC) | after the shift | overall | cost per access (Python) |
|---|---|---|---|
| ARC | 0 | 0 | 1.6 us |
| LRU-like, detector-triggered | +3.03 | -1.79 | 5.0 us |
| learned re-ranker, detector-triggered | +1.19 | -0.91 | 153 us |
| learned re-ranker, always on | +2.78 | -0.30 | 422 us |

On real data always-on loses less than detector-triggered, the opposite of the synthetic result: the gain
comes from the eviction choices, and the detector (about 5 false starts per trace, on 28% of the time) makes
it worse. Overhead: about 0.8 ms per model call, 0.33 calls per eviction when triggered and 0.97 always on,
i.e. 95-260x ARC's per-access cost in this implementation. A deployable version needs a much cheaper model.

## Combined shifts, and CACHEUS / S3-FIFO inside the shift window (`scripts/real_controlled_combos.py`)
Same setup as the single-dimension controlled shifts (first 1M requests, change at 500k, 4 held-out test
volumes; each dimension in its own id namespace). Points vs ARC, first 50k accesses after the shift / whole
trace:

| scenario | LRU-like, true change point | CACHEUS | S3-FIFO | MIN (after shift) |
|---|---|---|---|---|
| popularity 1.0 | +0.10 / +0.10 | +1.58 / +1.52 | +0.74 / +1.06 | +5.95 |
| scan 0.3 | -1.37 / -0.32 | -0.87 / +0.24 | -0.08 / +0.70 | +7.81 |
| tenant 0.5 | +0.78 / +0.04 | -0.67 / -0.55 | -2.05 / +0.17 | +7.54 |
| popularity + scan | -0.62 / -0.10 | -1.10 / +0.47 | -0.41 / +1.08 | +2.47 |
| popularity + tenant | +1.34 / +0.09 | -0.52 / -0.52 | -2.47 / +0.40 | +3.33 |
| scan + tenant | +1.52 / +0.13 | +0.40 / -1.39 | -1.26 / +0.45 | +5.65 |

An LRU-like action helps when tenant traffic is mixed in and hurts when a scan is involved. No adaptive
policy is uniformly better than ARC right after a shift: S3-FIFO wins overall in every scenario but loses up
to 2.5 points right after tenant-mixing shifts; CACHEUS wins after a popularity shift and loses on tenant mixes.
The detector found at most 1 of 4 shifts per scenario. Which policy is best depends on the kind of shift,
consistent with the policy-selection result. Not covered: request-rate changes (they do not change eviction
order in an object-count cache without time-based expiry) and object-size changes (see byte-level results).

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

## Known issues
* The oracle re-ranker (`shiftcache/policies/oracle.py`) is a greedy per-decision oracle, not a ceiling:
  `gbt[det]` beats `Oracle-X[det]` on all 8 synthetic held-out servers (docstring corrected).
* The detector-gated GBT gain (+2.50 points in transition windows, `results/full/experiment_ci.txt`)
  is on synthetic servers from one generator family. A model-free control (ARC acting like LRU
  while the gate is on) reaches +1.52, so the learned model's increment is about +0.98 points.

---

## Code overview: synthetic pipeline

Trace-driven block-cache simulator for the shift-aware controller project:
a shift detector that reads only the raw request stream, plus a learned
re-ranker that overrides ARC's eviction choice only while a shift is flagged.

### Layout
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

### Setup
```
pip install numpy pandas matplotlib pytest scikit-learn libcachesim
python -m pytest -q tests          # expect 25 passed, 0 skipped
```
Run everything from the repo root (scripts import `shiftcache.*` and `scripts.*`).

### Data
MSR-Cambridge: SNIA IOTTA repository ("MSR Cambridge Traces"). Put files in `data/`
(e.g. `data/web_0.csv.gz`). Columns: Timestamp, Hostname, DiskNumber, Type, Offset, Size, ResponseTime.
CloudPhysics: via the cacheMon collection (libCacheSim format), not yet integrated.

### Run
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

### Pipeline
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

### Design decisions (record in the report)
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

### Status of the synthetic pipeline (as of Sep 29, 2026)
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

## Limitations
* Real results use the first 0.5M-3M requests of each trace, unit-size objects and mostly a 5% cache.
* Spliced shifts are artificial (a full swap between two real volumes); controlled shifts are injected.
* Test sets are small (4 held-out MSR volumes for the controller; 33 traces over two folds for selection).
* The simulation-based selector's cost is measured but not optimised (sampled simulation would cut it).
* The learned selector has 60-105 training rows; more traces are the obvious next step.

## References
* N. Megiddo, D. Modha. ARC: A Self-Tuning, Low Overhead Replacement Cache. FAST 2003.
* Z. Song et al. HALP: Heuristic Aided Learned Preference Eviction Policy for YouTube Content Delivery Network. NSDI 2023.
* D. Yang, D. Berger, K. Li, W. Lloyd. A Learned Cache Eviction Framework with Minimal Overhead (MAT). arXiv 2023.
* Xia et al. Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction. OSDI 2026.
* N. Beckmann, H. Chen, A. Cidon. LHD: Improving Cache Hit Rate by Maximizing Hit Density. NSDI 2018.
* L. Rodriguez et al. Learning Cache Replacement with CACHEUS. FAST 2021.
* G. Vietri et al. Driving Cache Replacement with ML-based LeCaR. HotStorage 2018.
* J. Yang et al. FIFO Queues are All You Need for Cache Eviction (S3-FIFO). SOSP 2023.
* Y. Zhang et al. SIEVE is Simpler than LRU. NSDI 2024.
* G. Einziger, R. Friedman, B. Manes. TinyLFU: A Highly Efficient Cache Admission Policy. ACM TOS 2017.
* C. Waldspurger et al. Cache Modeling and Optimization using Miniature Simulations. USENIX ATC 2017.
* libCacheSim (https://github.com/1a1a11a/libCacheSim) and the cacheMon cache_dataset collection
  (https://github.com/cacheMon/cache_dataset).
* Traces: D. Narayanan, A. Donnelly, A. Rowstron. Write Off-Loading (MSR-Cambridge). FAST 2008;
  C. Waldspurger et al. Efficient MRC Construction with SHARDS (CloudPhysics). FAST 2015;
  J. Li et al. An In-Depth Analysis of Cloud Block Storage Workloads in Large-Scale Production (Alibaba).
  IISWC 2020; Meta storage traces distributed with CacheLib.
