#!/usr/bin/env bash
# v3 extraction progress snapshot. Run on demand for one-shot status.
# Emits a single line summarising phase completion and overall percentage.
set -u
ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"

# Expected total line counts based on prior runs (after Pass-2 filter)
EXP_DEV=1078
EXP_TEST=2569
EXP_TRAIN=9832
EXP_AMI=19279
EXP_TOTAL=$((EXP_DEV + EXP_TEST + EXP_TRAIN + EXP_AMI))

count_lines() {
  if [ -f "$1" ]; then
    wc -l < "$1" | tr -d ' '
  else
    echo 0
  fi
}

dev=$(count_lines data/meld/parametric_prosody_dev_v3_raw.jsonl)
test_=$(count_lines data/meld/parametric_prosody_test_v3_raw.jsonl)
train=$(count_lines data/meld/parametric_prosody_train_v3_raw.jsonl)
ami=$(count_lines data/ami/parametric_prosody_ami_v3_raw.jsonl)
total=$((dev + test_ + train + ami))

# Percent helper
pct() {
  local cur=$1 exp=$2
  if [ "$exp" -le 0 ]; then echo 0; return; fi
  echo $((100 * cur / exp))
}

dev_pct=$(pct "$dev" "$EXP_DEV")
test_pct=$(pct "$test_" "$EXP_TEST")
train_pct=$(pct "$train" "$EXP_TRAIN")
ami_pct=$(pct "$ami" "$EXP_AMI")
overall_pct=$(pct "$total" "$EXP_TOTAL")

# Detect which phase is active and which are DONE.
# Sequential orchestrator: a phase is complete once the NEXT phase has started writing.
active="?"
dev_state="pending"
test_state="pending"
train_state="pending"
ami_state="pending"
if [ -f data/v3_extraction_done.marker ]; then
  active="DONE"
  dev_state=test_state=train_state=ami_state=done
  dev_pct=100; test_pct=100; train_pct=100; ami_pct=100
  overall_pct=100
elif kill -0 66058 2>/dev/null; then
  if [ "$ami" -gt 0 ]; then
    active="phase4-AMI"; dev_state=done; test_state=done; train_state=done
    dev_pct=100; test_pct=100; train_pct=100
  elif [ "$train" -gt 0 ]; then
    active="phase3-train"; dev_state=done; test_state=done
    dev_pct=100; test_pct=100
  elif [ "$test_" -gt 0 ]; then
    active="phase2-test"; dev_state=done
    dev_pct=100
  elif [ "$dev" -gt 0 ]; then
    active="phase1-dev"
  else
    active="starting"
  fi
else
  active="DEAD (orchestrator not running)"
fi

# Single-line status
ts=$(date +%H:%M:%S)
echo "[$ts] v3 progress: dev ${dev}/${EXP_DEV} (${dev_pct}%) | test ${test_}/${EXP_TEST} (${test_pct}%) | train ${train}/${EXP_TRAIN} (${train_pct}%) | AMI ${ami}/${EXP_AMI} (${ami_pct}%) | overall ${total}/${EXP_TOTAL} (${overall_pct}%) | active: ${active}"
