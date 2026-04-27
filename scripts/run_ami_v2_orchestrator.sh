#!/usr/bin/env bash
# AMI v2 (MFA-aligned) cross-corpus probe orchestrator.
#
# Runs the MFA-aligned analogue of MELD's EXP-007 question-prediction probe on
# AMI's el.inf vs statement task. This is the apples-to-apples comparison
# Felipe is after: same v2 syllabification quality, same probes, both corpora.
#
# Steps:
#   1. Wait for the per-DA followups orchestrator to land its done marker
#      (it produces the per-DA wav + .lab pairs and the per-DA manifests).
#   2. Run MFA align on the AMI per-DA inputs.
#   3. Run the v2 AMI extractor → parametric_prosody_ami_v2.jsonl.
#   4. Run the cross-corpus probes (run_exp010_probes.py) on v2 input
#      → data/ami/exp010_results_v2.json.
#   5. Append the v2 results to docs/findings.md and docs/diary.md.
#   6. Touch data/ami/exp010_v2_done.marker.
#
# Logs to data/ami/exp010_v2.log. Designed for nohup background execution.

set -u

ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/data/ami/exp010_v2.log"
DONE="$ROOT/data/ami/exp010_v2_done.marker"
PRE_DONE="$ROOT/data/ami/exp010_followups_done.marker"

mkdir -p "$ROOT/data/ami"
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# Step 1 — wait for Q2/Q3 followups orchestrator
# ---------------------------------------------------------------------------
log "== Step 1: waiting for per-DA followups orchestrator =="
while [ ! -f "$PRE_DONE" ]; do
  log "waiting for $PRE_DONE ..."
  sleep 60
done
log "per-DA followups marker present; proceeding"

# Sanity: we need per-DA wavs + labs ready (use find — `ls *` exceeds argument-list-too-long on 20k+ files)
n_da_wav=$(find data/ami/per_da_input -name "*.wav" -type f 2>/dev/null | wc -l | tr -d ' ')
n_da_lab=$(find data/ami/per_da_input -name "*.lab" -type f 2>/dev/null | wc -l | tr -d ' ')
log "per-DA inputs: ${n_da_wav} wavs, ${n_da_lab} labs"
if [ "$n_da_wav" -lt 1000 ] || [ "$n_da_lab" -lt 1000 ]; then
  log "FATAL: too few per-DA inputs to align"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 2 — MFA align
# ---------------------------------------------------------------------------
log "== Step 2: running MFA align on per-DA AMI inputs =="

mkdir -p data/ami/per_da_textgrids_mfa

# Run MFA via the conda environment, mirroring process_meld_split.sh
# Note: MFA align scans subdirectories of the input root recursively.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mfa-env

mfa align \
    "$ROOT/data/ami/per_da_input" \
    english_us_arpa \
    english_us_arpa \
    "$ROOT/data/ami/per_da_textgrids_mfa" \
    --num_jobs 8 --clean 2>&1 | tee -a "$LOG"

# Some MFA versions emit TextGrids inside the source dir instead of dest.
# Walk both and consolidate into per_da_textgrids_mfa/<meeting>/<utt>.TextGrid.
shopt -s nullglob
moved=0
for tg in data/ami/per_da_input/*/*.TextGrid; do
  meeting=$(basename "$(dirname "$tg")")
  mkdir -p "data/ami/per_da_textgrids_mfa/$meeting"
  mv "$tg" "data/ami/per_da_textgrids_mfa/$meeting/"
  moved=$((moved + 1))
done
shopt -u nullglob
log "  consolidated $moved stray TextGrids"

n_tg=$(ls data/ami/per_da_textgrids_mfa/*/*.TextGrid 2>/dev/null | wc -l | tr -d ' ')
log "MFA produced $n_tg TextGrids (target was ~${n_da_wav})"

conda deactivate

# ---------------------------------------------------------------------------
# Step 3 — v2 extraction
# ---------------------------------------------------------------------------
log "== Step 3: extracting v2 parametric prosody on AMI =="

"$PY" -m scripts.extract_parametric_prosody_ami_v2 \
    --manifest-glob 'data/ami/per_da_input/*/manifest.jsonl' \
    --textgrids-root data/ami/per_da_textgrids_mfa \
    --out data/ami/parametric_prosody_ami_v2.jsonl 2>&1 | tee -a "$LOG"

n_v2=$(wc -l < data/ami/parametric_prosody_ami_v2.jsonl | tr -d ' ')
log "v2 extraction complete: $n_v2 segments in parametric_prosody_ami_v2.jsonl"

# ---------------------------------------------------------------------------
# Step 4 — cross-corpus probes at v2 quality
# ---------------------------------------------------------------------------
log "== Step 4: running cross-corpus probes on AMI v2 vs MELD v2 =="

"$PY" -m scripts.run_exp010_probes \
    --ami-parametric data/ami/parametric_prosody_ami_v2.jsonl \
    --meld-train-parametric data/meld/parametric_prosody_train_mfa.jsonl \
    --meld-test-parametric data/meld/parametric_prosody_test_mfa.jsonl \
    --meld-train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \
    --meld-test-csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \
    --out data/ami/exp010_results_v2.json 2>&1 | tee -a "$LOG"

log "v2 probes complete: results in data/ami/exp010_results_v2.json"

# ---------------------------------------------------------------------------
# Step 5 — append findings to docs
# ---------------------------------------------------------------------------
log "== Step 5: appending v2 findings to docs =="

"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from datetime import date

r = json.loads(Path("data/ami/exp010_results_v2.json").read_text())

ami_pooled = r["prosody_pooled_lr"]["auc_mean"]
ami_last = r["prosody_last_syl_lr"]["auc_mean"]
ami_bilstm_small = r["prosody_bilstm_small"]["auc_mean"]
ami_bilstm_rich = r["prosody_bilstm_rich"]["auc_mean"]
ami_speaker = r["speaker_held_out_last_syl"]["auc_mean"]
pos = r["position_ablation"]
pos_first = pos["first_syl"]["auc_mean"]
pos_middle = pos["middle_syl"]["auc_mean"]
pos_last = pos["last_syl"]["auc_mean"]
text_uplift = r["text_vs_prosody"]["combined"]["mean"] - r["text_vs_prosody"]["text"]["mean"]

# MELD v2 baseline (from the comparison block in the probe script — reuses MFA paths now)
meld_v2_last = r["meld_v1_yn_q_last_syl_auc"]  # field name predates v2; key is the last-syl AUC value
meld_v2_pooled = r["meld_v1_yn_q_pooled_auc"]
gap_last = ami_last - meld_v2_last

stable_top = [c for c in r["bootstrap_coefs"][:6] if c["sign_stable"]]
n_stable = len(stable_top)

block = f"""

## EXP-010 v2 — MFA-aligned cross-corpus question prediction on AMI ({date.today().isoformat()})

The MFA-aligned (v2) parametric pipeline was applied to the per-DA AMI subset, enabling apples-to-apples comparison with MELD's EXP-007 v2 numbers. The probe target on AMI is the `Elicit-Inform` dialogue act vs clear-statement DAs; on MELD it is utterance-final question-mark presence (yes/no questions only).

**Prosody-only AUC for predicting `el.inf` versus statement DAs**, 5-fold stratified CV, AMI v2: pooled LR = {ami_pooled:.4f}; last-syllable LR = {ami_last:.4f}; bi-LSTM (mean-pool, small) = {ami_bilstm_small:.4f}; bi-LSTM (attention-pool, large) = {ami_bilstm_rich:.4f}.

**Cross-corpus comparison at v2 quality.** AMI `el.inf` last-syl AUC = {ami_last:.4f}; MELD yn-Q last-syl AUC = {meld_v2_last:.4f}; gap = {gap_last:+.4f}.

**Speaker-held-out** (GroupKFold by `global_name`) AUC = {ami_speaker:.4f}.

**Position ablation on AMI v2:** first = {pos_first:.4f}; middle = {pos_middle:.4f}; last = {pos_last:.4f}.

**Text vs prosody on AMI v2:** combined uplift over text alone = {text_uplift:+.4f} macro-F1.

**Bootstrap-stable coefficients in top 6:** {n_stable}/6 sign-stable {{{", ".join(c["dim"] for c in stable_top)}}}.
"""
fp = Path("docs/findings.md")
fp.write_text(fp.read_text() + block)
print("[ok] appended v2 block to docs/findings.md")

dp = Path("docs/diary.md")
diary = f"""

## {date.today().isoformat()} — EXP-010 v2 (MFA on AMI) complete

MFA-aligned cross-corpus question-prediction probe on AMI per-DA data finished.
- AMI el.inf last-syl AUC v2 = {ami_last:.4f}; pooled = {ami_pooled:.4f}; bi-LSTM rich = {ami_bilstm_rich:.4f}.
- MELD yn-Q last-syl AUC v2 (apples-to-apples) = {meld_v2_last:.4f}.
- Gap = {gap_last:+.4f}.
- AMI v2 position ablation: first/middle/last = {pos_first:.4f}/{pos_middle:.4f}/{pos_last:.4f}.
- AMI v2 text+prosody uplift = {text_uplift:+.4f}.
- AMI v2 bootstrap-stable coefficients in top 6: {n_stable}/6.

Results in data/ami/exp010_results_v2.json.
"""
dp.write_text(diary + dp.read_text())
print("[ok] prepended v2 entry to docs/diary.md")
PYEOF

# ---------------------------------------------------------------------------
# Step 6 — done marker
# ---------------------------------------------------------------------------
date > "$DONE"
log "== AMI v2 pipeline complete =="
log "results: data/ami/exp010_results_v2.json"
log "findings: docs/findings.md"
log "diary: docs/diary.md"
log "done marker: $DONE"
