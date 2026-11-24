#!/usr/bin/env bash
# Re-runs every real-trace experiment in dependency order into a fresh folder (default results/repro; the scripts
# resume from raw files already present, so never point OUT at a folder with old results), then compares every
# summary .txt with the committed one in results/real.
# Needs the traces under data/ (see README, "Data") and the pinned versions in requirements.txt.
# Single core takes roughly 20 hours; independent stages of a phase run in parallel with JOBS>1.
#   ./reproduce.sh            all phases
#   ./reproduce.sh 4          start at phase 4 (earlier outputs already in $OUT)
#   OUT=results/repro2 JOBS=2 ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-results/repro}
LOG=$OUT/logs
mkdir -p "$LOG"
START=${1:-1}
JOBS=${JOBS:-1}

run() {  # run <name> <command...>, log to $LOG/<name>.log
  local name=$1; shift
  echo "[$(date +%H:%M)] $name"
  "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see $LOG/$name.log)"; exit 1; }
}
par() {  # run a list of "name|command" jobs, JOBS at a time
  local pids=() spec
  for spec in "$@"; do
    run "${spec%%|*}" bash -c "${spec#*|}" &
    pids+=($!)
    if (( ${#pids[@]} >= JOBS )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

if (( START <= 1 )); then  # 1. baselines on every source
  run baselines_msr python -u scripts/real_baselines.py --dir data/msr --out $OUT
  for t in cp ali meta; do
    run baselines_$t python -u scripts/real_baselines.py --dir data/$t --pattern '*.zst' --tag $t --out $OUT
  done
fi
if (( START <= 2 )); then  # 2. Progress Report 1: shift detector, gate, selection
  run real_experiment python -u scripts/real_experiment.py --dir data/msr --out $OUT --side 500000 \
      --natural-n 1500000 --train-pairs 10 --test-pairs 12 --skip-always
  run gate_tuning python -u scripts/real_gate_tuning.py --dir data/msr --out $OUT
  run controlled_shifts python -u scripts/real_controlled_shifts.py --dir data/msr --out $OUT
  run headroom_5 python -u scripts/policy_selection_headroom.py --out $OUT
  run learned_selector python -u scripts/learned_selector.py --out $OUT
  run hybrid_selector python -u scripts/hybrid_selector.py --out $OUT
  run always_overhead python -u scripts/real_always_overhead.py --dir data/msr --out $OUT
  run controlled_combos python -u scripts/real_controlled_combos.py --dir data/msr --out $OUT
  run headroom_sizes python -u scripts/policy_selection_headroom.py --sizes --out $OUT --tmp /tmp/hb
  par "headroom_1|python -u scripts/policy_selection_headroom.py --fraction 0.01 --out $OUT --tmp /tmp/h01" \
      "headroom_10|python -u scripts/policy_selection_headroom.py --fraction 0.10 --out $OUT --tmp /tmp/h10"
  run loso python -u scripts/loso_eval.py --out $OUT
fi
if (( START <= 3 )); then  # 3. switching, tuning and switchers at 1, 5 and 10%
  par "switching_1|python -u scripts/switching.py --fraction 0.01 --out $OUT" \
      "switching_5|python -u scripts/switching.py --fraction 0.05 --out $OUT" \
      "switching_10|python -u scripts/switching.py --fraction 0.10 --out $OUT"
  par "tuning_1|python -u scripts/s3fifo_tuning.py --fraction 0.01 --out $OUT" \
      "tuning_5|python -u scripts/s3fifo_tuning.py --fraction 0.05 --out $OUT" \
      "tuning_10|python -u scripts/s3fifo_tuning.py --fraction 0.10 --out $OUT"
  run cost_aware python -u scripts/cost_aware_switcher.py --out $OUT
fi
if (( START <= 4 )); then  # 4. Progress Report 2: sampled shadows, overhead, rules, sensitivity
  par "switcher_v2_1|python -u scripts/switcher_v2.py --fraction 0.01 --out $OUT --tmp /tmp/sw2_1" \
      "switcher_v2_5|python -u scripts/switcher_v2.py --fraction 0.05 --out $OUT --tmp /tmp/sw2_5" \
      "switcher_v2_10|python -u scripts/switcher_v2.py --fraction 0.10 --out $OUT --tmp /tmp/sw2_10"
  run shadow_overhead python -u scripts/shadow_overhead.py --out $OUT
  par "margin_1|python -u scripts/margin_switcher.py --fraction 0.01 --out $OUT" \
      "margin_5|python -u scripts/margin_switcher.py --fraction 0.05 --out $OUT" \
      "margin_10|python -u scripts/margin_switcher.py --fraction 0.10 --out $OUT"
  run gate python -u scripts/gate_switcher.py --out $OUT
  run workload python scripts/workload_analysis.py --out $OUT
  par "window_1|python -u scripts/window_sensitivity.py --fraction 0.01 --out $OUT --tmp /tmp/ws" \
      "window_10|python -u scripts/window_sensitivity.py --fraction 0.10 --out $OUT --tmp /tmp/ws" \
      "size_1|python -u scripts/size_aware.py --fraction 0.01 --out $OUT --tmp /tmp/size" \
      "size_10|python -u scripts/size_aware.py --fraction 0.10 --out $OUT --tmp /tmp/size"
fi
run figures python scripts/make_figures.py --out $OUT/figures

echo
echo "Summaries that differ from results/real (CACHEUS, LHD and LeCaR are randomized in libCacheSim and can move"
echo "results that use them; everything else should match exactly):"
for f in results/real/*.txt; do
  g=$OUT/$(basename "$f")
  if [[ ! -f $g ]]; then echo "  missing  $(basename "$f")"; elif ! cmp -s "$f" "$g"; then echo "  differs  $(basename "$f")"; fi
done
