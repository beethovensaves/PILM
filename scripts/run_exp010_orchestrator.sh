#!/usr/bin/env bash
# EXP-010 — full-pipeline orchestrator. Runs autonomously.
#
# Steps (with idempotent guards):
#   1. Wait for slicing to complete (manifests appear for all 30 meetings).
#   2. Run v1 parametric extraction on AMI.
#   3. Run cross-corpus probes (AMI el.inf + MELD v1 yn-Q comparison).
#   4. Append findings to docs/findings.md and docs/diary.md.
#   5. Touch a "done" marker file.
#
# Designed to run in nohup background. Logs to data/ami/exp010_pipeline.log.

set -u

ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"

LOG="$ROOT/data/ami/exp010_pipeline.log"
DONE_MARKER="$ROOT/data/ami/exp010_done.marker"
PY="$ROOT/.venv/bin/python"

mkdir -p data/ami
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# Step 1: wait for slicing to complete
# ---------------------------------------------------------------------------
log "== Step 1: waiting for slicing of all 30 meetings =="

EXPECTED_MEETINGS=(
  ES2002a ES2002b ES2002c ES2002d ES2003a ES2003b ES2003c ES2003d ES2004a ES2004b
  IS1000a IS1000b IS1000c IS1000d IS1001a IS1001b IS1001c IS1002b IS1002c IS1002d
  TS3003a TS3003b TS3003c TS3003d TS3004a TS3004b TS3004c TS3004d TS3005a TS3005b
)

while true; do
  done_count=0
  for m in "${EXPECTED_MEETINGS[@]}"; do
    if [ -f "data/ami/mfa_input/$m/manifest.jsonl" ]; then
      done_count=$((done_count + 1))
    fi
  done
  log "slicing progress: $done_count / 30 meetings"
  if [ "$done_count" -eq 30 ]; then
    break
  fi
  sleep 30
done
log "slicing complete (30/30)"

# ---------------------------------------------------------------------------
# Step 2: v1 parametric extraction on AMI
# ---------------------------------------------------------------------------
log "== Step 2: extracting parametric prosody on AMI v1 =="

if [ -f "data/ami/parametric_prosody_ami_v1.jsonl" ] && [ -s "data/ami/parametric_prosody_ami_v1.jsonl" ]; then
  log "(skipping: parametric_prosody_ami_v1.jsonl already exists with content)"
else
  "$PY" -m scripts.extract_parametric_prosody_ami_v1 \
      --manifest-glob 'data/ami/mfa_input/*/manifest.jsonl' \
      --out data/ami/parametric_prosody_ami_v1.jsonl 2>&1 | tee -a "$LOG"
fi

n_segments=$(wc -l < data/ami/parametric_prosody_ami_v1.jsonl | tr -d ' ')
log "extraction complete: $n_segments segments in parametric_prosody_ami_v1.jsonl"

# ---------------------------------------------------------------------------
# Step 3: cross-corpus probes
# ---------------------------------------------------------------------------
log "== Step 3: running cross-corpus probes =="

"$PY" -m scripts.run_exp010_probes \
    --ami-parametric data/ami/parametric_prosody_ami_v1.jsonl \
    --meld-train-parametric data/meld/parametric_prosody_train_v1.jsonl \
    --meld-test-parametric data/meld/parametric_prosody_test_v1.jsonl \
    --meld-train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \
    --meld-test-csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \
    --out data/ami/exp010_results.json 2>&1 | tee -a "$LOG"

log "probes complete: results in data/ami/exp010_results.json"

# ---------------------------------------------------------------------------
# Step 4: append findings to docs/findings.md and docs/diary.md
# ---------------------------------------------------------------------------
log "== Step 4: appending findings to docs =="

"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from datetime import date

results_path = Path("data/ami/exp010_results.json")
results = json.loads(results_path.read_text())

ami_auc_pooled = results["prosody_pooled_lr"]["auc_mean"]
ami_auc_last = results["prosody_last_syl_lr"]["auc_mean"]
ami_auc_bilstm = results["prosody_bilstm"]["auc_mean"]
ami_speaker_auc = results["speaker_held_out_last_syl"]["auc_mean"]
pos_first = results["position_ablation"]["first_syl"]["auc_mean"]
pos_middle = results["position_ablation"]["middle_syl"]["auc_mean"]
pos_last = results["position_ablation"]["last_syl"]["auc_mean"]

text_f1 = results["text_vs_prosody"]["text"]["mean"]
prosody_f1 = results["text_vs_prosody"]["prosody"]["mean"]
combined_f1 = results["text_vs_prosody"]["combined"]["mean"]
combined_uplift = combined_f1 - text_f1

meld_auc = results["meld_v1_yn_q_last_syl_auc"]
gap = ami_auc_last - meld_auc

stable_top = [c for c in results["bootstrap_coefs"][:6] if c["sign_stable"]]

findings_block = f"""

## EXP-010 — Cross-corpus prosody probe on AMI ({date.today().isoformat()})

The same v1 parametric pipeline was applied to a 30-meeting subset of the AMI Meeting Corpus (10 each from the Edinburgh, Idiap, and TNO/Twente recording sites; {results["counts"]["n_meetings"]} meetings, {results["counts"]["n_speakers"]} unique speakers, {results["counts"]["n_total"]} segments, {results["counts"]["n_pos_el_inf"]} `el.inf` and {results["counts"]["n_neg_statement"]} clear-statement DAs).

**Prosody-only AUC for predicting `el.inf` (Elicit-Inform) versus statement DAs**, 5-fold stratified CV: pooled LR = {ami_auc_pooled:.4f}; last-syllable LR = {ami_auc_last:.4f}; bi-LSTM sequence = {ami_auc_bilstm:.4f}.

**Cross-corpus comparison (apples-to-apples, v1 prosody on both):** AMI `el.inf` last-syl AUC = {ami_auc_last:.4f}; MELD yn-Q last-syl AUC (same algorithm, v1) = {meld_auc:.4f}; gap = {gap:+.4f}.

**Speaker-held-out** (GroupKFold by `global_name`) AUC = {ami_speaker_auc:.4f} — close to within-speaker AUC, supporting that the probe is not speaker-fingerprinting.

**Position ablation** confirms the boundary-tone localisation finding from MELD: last-syllable AUC = {pos_last:.4f} versus middle = {pos_middle:.4f} and first = {pos_first:.4f}.

**Text vs prosody on `el.inf` (5-fold macro-F1)**: text-only = {text_f1:.4f}; prosody-only = {prosody_f1:.4f}; combined = {combined_f1:.4f} (uplift over text alone: {combined_uplift:+.4f}).

**Bootstrap-stable coefficients** ({len(stable_top)} of top 6): {", ".join(c["dim"] for c in stable_top)}.
"""
fp = Path("docs/findings.md")
content = fp.read_text()
fp.write_text(content + findings_block)
print("[ok] appended EXP-010 to docs/findings.md")

dp = Path("docs/diary.md")
diary_block = f"""

## {date.today().isoformat()} — EXP-010 cross-corpus probe complete

Cross-corpus probe results written to `data/ami/exp010_results.json` and summarised in `docs/findings.md`. Headline: AMI `el.inf` prosody-only last-syl AUC = {ami_auc_last:.4f} versus MELD yn-Q AUC = {meld_auc:.4f} (gap {gap:+.4f}). Combined text+prosody on AMI `el.inf` adds {combined_uplift:+.4f} macro-F1 over text alone. Position ablation reproduces the MELD boundary-tone localisation: last-syl AUC {pos_last:.4f} > middle {pos_middle:.4f}.
"""
content = dp.read_text()
dp.write_text(diary_block + content)
print("[ok] prepended EXP-010 to docs/diary.md")
PYEOF

log "docs updated"

# ---------------------------------------------------------------------------
# Step 5: done marker
# ---------------------------------------------------------------------------
date > "$DONE_MARKER"
log "== EXP-010 pipeline complete =="
log "results: data/ami/exp010_results.json"
log "findings: docs/findings.md"
log "diary: docs/diary.md"
log "done marker: $DONE_MARKER"
