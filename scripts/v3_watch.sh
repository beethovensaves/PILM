#!/usr/bin/env bash
# Live progress dashboard for the v3 extraction orchestrator.
# Refreshes every 2 seconds in-place; exits when the v3 done marker drops or
# the orchestrator process exits.
#
# Usage:
#   bash scripts/v3_watch.sh
#
# Run this in a terminal you can leave open while extraction runs.

set -u
ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"

# Expected total line counts based on prior runs (after Pass-2 filter)
EXP_DEV=1078
EXP_TEST=2569
EXP_TRAIN=9832
EXP_AMI=19279
EXP_TOTAL=$((EXP_DEV + EXP_TEST + EXP_TRAIN + EXP_AMI))

ORCHESTRATOR_PID="${1:-66058}"
REFRESH_SECONDS="${2:-2}"

count_lines() {
  if [ -f "$1" ]; then
    wc -l < "$1" 2>/dev/null | tr -d ' '
  else
    echo 0
  fi
}

# Colour codes (basic ANSI)
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
DIM=$'\033[2m'
BOLD=$'\033[1m'
RESET=$'\033[0m'
CLEAR_SCREEN=$'\033[2J\033[H'

# Render a progress bar of width N, given a current / target pair OR a precomputed pct
# (call as: bar PCT MAX WIDTH where PCT is already-clamped 0..100 and MAX is 100)
bar() {
  local pct=$2 width=${3:-30}
  if [ "$pct" -gt 100 ]; then pct=100; fi
  local filled=$((pct * width / 100))
  local empty=$((width - filled))
  local fill_chars=""
  local empty_chars=""
  for ((i=0; i<filled; i++)); do fill_chars+="█"; done
  for ((i=0; i<empty; i++)); do empty_chars+="░"; done
  printf "%s%s" "$fill_chars" "$empty_chars"
}

format_phase() {
  local name=$1 cur=$2 exp=$3 forced_done=$4
  # Pattern: in a sequential pipeline, mark a phase 100% when the next phase has begun
  # writing — Pass 2 filtering means actual line count caps slightly below prior-run
  # expected, so raw cur/exp under-reports completion. forced_done flag overrides.
  local pct=0
  if [ "$forced_done" = "1" ]; then
    pct=100
  elif [ "$exp" -gt 0 ]; then
    pct=$((100 * cur / exp))
    if [ "$pct" -gt 100 ]; then pct=100; fi
  fi
  local color="$YELLOW"
  if [ "$pct" -ge 100 ]; then color="$GREEN"; fi
  printf "  %-8s  %s%s%s  %4d%%  %s%6d / %d%s\n" \
    "$name" "$color" "$(bar 100 "$pct" 30)" "$RESET" "$pct" "$DIM" "$cur" "$exp" "$RESET"
}

# Extraction start: use the orchestrator log's birthtime if available,
# otherwise fall back to script-launch time. (macOS-specific stat -f %B.)
if [ -f data/v3_extraction.log ]; then
  log_birth=$(stat -f %B data/v3_extraction.log 2>/dev/null || echo "")
  if [ -n "$log_birth" ] && [ "$log_birth" -gt 0 ] 2>/dev/null; then
    start_ts=$log_birth
  else
    start_ts=$(date +%s)
  fi
else
  start_ts=$(date +%s)
fi
prev_total=-1
prev_ts=$(date +%s)

while true; do
  dev=$(count_lines data/meld/parametric_prosody_dev_v3_raw.jsonl)
  test_=$(count_lines data/meld/parametric_prosody_test_v3_raw.jsonl)
  train=$(count_lines data/meld/parametric_prosody_train_v3_raw.jsonl)
  ami=$(count_lines data/ami/parametric_prosody_ami_v3_raw.jsonl)
  total=$((dev + test_ + train + ami))
  overall_pct=0
  if [ "$EXP_TOTAL" -gt 0 ]; then
    overall_pct=$((100 * total / EXP_TOTAL))
    if [ "$overall_pct" -gt 100 ]; then overall_pct=100; fi
  fi

  # Active phase + per-phase done flags. In a sequential orchestrator a phase
  # is complete the moment the next phase starts writing — raw cur/exp would
  # under-report because Pass 2 filtering caps actual write count slightly
  # below prior-run expected.
  active="?"
  dev_done=0; test_done=0; train_done=0; ami_done=0
  if [ -f data/v3_extraction_done.marker ]; then
    active="${GREEN}DONE${RESET}"
    dev_done=1; test_done=1; train_done=1; ami_done=1
    overall_pct=100
  elif kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
    if   [ "$ami"   -gt 0 ]; then
      active="phase 4: AMI per-DA"
      dev_done=1; test_done=1; train_done=1
    elif [ "$train" -gt 0 ]; then
      active="phase 3: MELD train"
      dev_done=1; test_done=1
    elif [ "$test_" -gt 0 ]; then
      active="phase 2: MELD test"
      dev_done=1
    elif [ "$dev"   -gt 0 ]; then
      active="phase 1: MELD dev"
    else
      active="starting (Pass 1 baselines)"
    fi
  else
    active="${YELLOW}orchestrator (PID $ORCHESTRATOR_PID) not running${RESET}"
  fi

  # Throughput / ETA
  now=$(date +%s)
  elapsed=$((now - start_ts))
  rate_str=""
  eta_str=""
  if [ "$prev_total" -ge 0 ] && [ "$total" -gt "$prev_total" ]; then
    delta=$((total - prev_total))
    delta_t=$((now - prev_ts))
    if [ "$delta_t" -gt 0 ]; then
      rate=$(echo "scale=2; $delta / $delta_t" | bc 2>/dev/null || echo "?")
      remaining=$((EXP_TOTAL - total))
      if [ "$delta" -gt 0 ] && [ "$remaining" -gt 0 ]; then
        eta_seconds=$((remaining * delta_t / delta))
        eta_min=$((eta_seconds / 60))
        eta_str="ETA ~${eta_min} min"
      fi
      rate_str="rate ${rate} utt/s"
    fi
  fi
  prev_total=$total
  prev_ts=$now

  printf "%s" "$CLEAR_SCREEN"
  printf "%sPILM v3 extraction — live progress%s\n\n" "$BOLD" "$RESET"
  printf "  %-8s  %-30s  %4s  %s\n" "phase" "" "pct" "lines"
  printf "  %-8s  %-30s  %4s  %s\n" "-----" "------------------------------" "----" "-----"
  format_phase "dev"   "$dev"   "$EXP_DEV"   "$dev_done"
  format_phase "test"  "$test_" "$EXP_TEST"  "$test_done"
  format_phase "train" "$train" "$EXP_TRAIN" "$train_done"
  format_phase "AMI"   "$ami"   "$EXP_AMI"   "$ami_done"
  printf "\n"
  printf "  %-8s  %s%s%s  %s%4d%%%s  %s%6d / %d%s\n" \
    "OVERALL" "$BOLD$GREEN" "$(bar 100 "$overall_pct" 30)" "$RESET" \
    "$BOLD" "$overall_pct" "$RESET" "$DIM" "$total" "$EXP_TOTAL" "$RESET"
  printf "\n"
  printf "  %sactive:%s   %b\n" "$DIM" "$RESET" "$active"
  printf "  %selapsed:%s  %d min %d sec\n" "$DIM" "$RESET" $((elapsed / 60)) $((elapsed % 60))
  if [ -n "$rate_str" ]; then
    printf "  %sthroughput:%s  %s   %s\n" "$DIM" "$RESET" "$rate_str" "$eta_str"
  fi
  printf "\n  %s(refreshing every %ds; Ctrl-C to exit)%s\n" "$DIM" "$REFRESH_SECONDS" "$RESET"

  # Exit conditions
  if [ -f data/v3_extraction_done.marker ]; then
    printf "\n  %s== v3 extraction complete ==%s\n" "$BOLD$GREEN" "$RESET"
    break
  fi
  if ! kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
    printf "\n  %s== orchestrator (PID %s) has exited ==%s\n" "$BOLD$YELLOW" "$ORCHESTRATOR_PID" "$RESET"
    break
  fi
  sleep "$REFRESH_SECONDS"
done
