#!/usr/bin/env bash
# EXP-010 follow-ups (Q2 + Q3): per-DA AMI extraction + rich-probe re-run on
# both per-segment and per-DA data.
#
# Steps:
#   1. Slice AMI audio per-DA into data/ami/per_da_input/<meeting>/.
#   2. Extract v1 parametric prosody on per-DA wavs → data/ami/parametric_prosody_ami_v1_perDA.jsonl.
#   3. Run probes (incl. rich bi-LSTM) on per-DA data → data/ami/exp010_results_perDA.json.
#   4. Re-run probes (now with rich bi-LSTM) on per-segment data → data/ami/exp010_results_richBiLSTM.json.
#   5. Append updated findings to docs/findings.md and docs/diary.md.
#   6. Touch data/ami/exp010_followups_done.marker.
#
# Logs: data/ami/exp010_followups.log

set -u
ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/data/ami/exp010_followups.log"
DONE_MARKER="$ROOT/data/ami/exp010_followups_done.marker"
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Step 1 — per-DA slicing
log "== Step 1: per-DA slicing of all 30 meetings =="
"$PY" -m scripts.prepare_ami_per_da \
    --meeting all-local \
    --audio-root data/ami/audio \
    --ami-root data/ami/ami_annotations \
    --out data/ami/per_da_input 2>&1 | tee -a "$LOG"
n_da=$(wc -l data/ami/per_da_input/*/manifest.jsonl | tail -1 | awk '{print $1}')
log "per-DA slicing complete: ~$n_da DAs across 30 meetings"

# Step 2 — extraction on per-DA
log "== Step 2: extracting v1 parametric prosody on per-DA wavs =="
"$PY" -m scripts.extract_parametric_prosody_ami_v1 \
    --manifest-glob 'data/ami/per_da_input/*/manifest.jsonl' \
    --out data/ami/parametric_prosody_ami_v1_perDA.jsonl 2>&1 | tee -a "$LOG"

# Step 3 — probes on per-DA
log "== Step 3: probes on per-DA parametric vectors =="
"$PY" -m scripts.run_exp010_probes \
    --ami-parametric data/ami/parametric_prosody_ami_v1_perDA.jsonl \
    --meld-train-parametric data/meld/parametric_prosody_train_v1.jsonl \
    --meld-test-parametric data/meld/parametric_prosody_test_v1.jsonl \
    --meld-train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \
    --meld-test-csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \
    --out data/ami/exp010_results_perDA.json 2>&1 | tee -a "$LOG"

# Step 4 — re-run probes (now with rich bi-LSTM) on per-segment data for parity
log "== Step 4: re-running probes on per-segment data with rich bi-LSTM =="
"$PY" -m scripts.run_exp010_probes \
    --ami-parametric data/ami/parametric_prosody_ami_v1.jsonl \
    --meld-train-parametric data/meld/parametric_prosody_train_v1.jsonl \
    --meld-test-parametric data/meld/parametric_prosody_test_v1.jsonl \
    --meld-train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \
    --meld-test-csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \
    --out data/ami/exp010_results_richBiLSTM.json 2>&1 | tee -a "$LOG"

# Step 5 — append findings
log "== Step 5: appending follow-up findings to docs =="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from datetime import date

per_da = json.loads(Path("data/ami/exp010_results_perDA.json").read_text())
per_seg_rich = json.loads(Path("data/ami/exp010_results_richBiLSTM.json").read_text())

# Q1 results (MELD v1 bootstrap from validate_exp007_v1.json)
v1_path = Path("data/meld/validate_exp007_v1.json")
if v1_path.exists():
    v1 = json.loads(v1_path.read_text())
    v1_baseline_auc = v1.get("baseline", {}).get("auc")
    v1_e = v1.get("e_bootstrap", {})
    v1_coefs = v1_e.get("coefficients", [])
    n_stable_top5 = sum(1 for c in v1_coefs[:5] if c.get("sign_stable"))
    v1_summary = (f"MELD v1 reproduce: baseline yn-only last-syl AUC = {v1_baseline_auc:.4f}; "
                   f"top-5 last-syl coefficients sign-stable: {n_stable_top5}/5")
else:
    v1_summary = "MELD v1 bootstrap result not yet on disk"

per_da_pooled = per_da["prosody_pooled_lr"]["auc_mean"]
per_da_last = per_da["prosody_last_syl_lr"]["auc_mean"]
per_da_pos = per_da["position_ablation"]
per_da_first = per_da_pos["first_syl"]["auc_mean"]
per_da_middle = per_da_pos["middle_syl"]["auc_mean"]
per_da_last_pos = per_da_pos["last_syl"]["auc_mean"]
per_da_text_uplift = per_da["text_vs_prosody"]["combined"]["mean"] - per_da["text_vs_prosody"]["text"]["mean"]
per_da_n = per_da["counts"]["n_total"]
per_da_pos_n = per_da["counts"]["n_pos_el_inf"]

ps_rich = per_seg_rich["prosody_bilstm_rich"]["auc_mean"]
ps_small = per_seg_rich["prosody_bilstm_small"]["auc_mean"]

pd_rich = per_da["prosody_bilstm_rich"]["auc_mean"]
pd_small = per_da["prosody_bilstm_small"]["auc_mean"]

block = f"""

## EXP-010 follow-ups: per-DA extraction, MELD v1 bootstrap, richer bi-LSTM ({date.today().isoformat()})

Three follow-up studies addressed surprises in the initial cross-corpus result.

**MELD v1 bootstrap (Q1).** {v1_summary}.

**Per-DA AMI re-extraction (Q2).** Re-slicing AMI audio per dialogue act (rather than per transcriber-marked segment) produced {per_da_n} DA-level utterances ({per_da_pos_n} `el.inf` positives). Prosody-only AUC: pooled LR = {per_da_pooled:.4f}; last-syllable LR = {per_da_last:.4f}. Position ablation on per-DA data: first = {per_da_first:.4f}, middle = {per_da_middle:.4f}, last = {per_da_last_pos:.4f}. Combined text+prosody uplift over text alone: {per_da_text_uplift:+.4f} macro-F1.

**Richer bi-LSTM probe (Q3).** A 4-layer × 128-hidden bi-LSTM with attention pooling and 30 training epochs reached AUC {ps_rich:.4f} on per-segment AMI data and {pd_rich:.4f} on per-DA data, compared to the original 2-layer × 64-hidden mean-pooled probe at {ps_small:.4f} (per-segment) and {pd_small:.4f} (per-DA).
"""
fp = Path("docs/findings.md")
fp.write_text(fp.read_text() + block)
print("[ok] appended follow-up block to docs/findings.md")

dp = Path("docs/diary.md")
diary_block = f"""

## {date.today().isoformat()} — EXP-010 follow-ups (Q1 / Q2 / Q3) complete

Three follow-up studies on the cross-corpus probe:
- **Q1 (MELD v1 bootstrap):** {v1_summary}
- **Q2 (per-DA re-extraction):** AMI per-DA prosody-only last-syl AUC = {per_da_last:.4f}; pooled = {per_da_pooled:.4f}; position ablation first/middle/last = {per_da_first:.4f}/{per_da_middle:.4f}/{per_da_last_pos:.4f}; text+prosody uplift = {per_da_text_uplift:+.4f}.
- **Q3 (richer bi-LSTM):** rich (4x128, attn, 30ep) AUC = {ps_rich:.4f} per-segment / {pd_rich:.4f} per-DA, vs small (2x64, mean, 10ep) at {ps_small:.4f} / {pd_small:.4f}.

Results: data/ami/exp010_results_perDA.json, data/ami/exp010_results_richBiLSTM.json, data/meld/validate_exp007_v1.json.
"""
dp.write_text(diary_block + dp.read_text())
print("[ok] prepended follow-up block to docs/diary.md")
PYEOF

# Step 6 — done marker
date > "$DONE_MARKER"
log "== EXP-010 follow-ups complete =="
log "results: per-DA → data/ami/exp010_results_perDA.json"
log "results: rich bi-LSTM (per-segment) → data/ami/exp010_results_richBiLSTM.json"
log "results: MELD v1 bootstrap → data/meld/validate_exp007_v1.json"
log "findings: docs/findings.md"
log "diary: docs/diary.md"
log "done marker: $DONE_MARKER"
