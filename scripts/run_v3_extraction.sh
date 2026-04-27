#!/usr/bin/env bash
# Full v3 extraction orchestrator: re-extract MELD (train/test/dev) + AMI per-DA
# with the 22-dim spec (18 vec + 4 vq), then post-process to add phrase digest
# (final 26-dim view: 18 vec + 4 vq + 4 phrase_digest).

set -u
ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/data/v3_extraction.log"
DONE="$ROOT/data/v3_extraction_done.marker"

: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "== Phase 1: MELD v3 extraction (sequential: dev → test → train) =="

# MELD dev
log "extracting MELD dev v3..."
"$PY" -m scripts.extract_parametric_prosody_mfa \
    --in-dir data/meld/MELD.Raw/dev_splits_complete \
    --textgrids-dir data/meld/dev_textgrids_mfa \
    --metadata data/meld/MELD.Raw/dev_sent_emo_cleaned.csv \
    --workers 8 \
    --out data/meld/parametric_prosody_dev_v3_raw.jsonl 2>&1 | tail -5 | tee -a "$LOG"
log "MELD dev v3 raw extraction done"

# MELD test
log "extracting MELD test v3..."
"$PY" -m scripts.extract_parametric_prosody_mfa \
    --in-dir data/meld/MELD.Raw/output_repeated_splits_test \
    --textgrids-dir data/meld/test_textgrids_mfa \
    --metadata data/meld/MELD.Raw/test_sent_emo_cleaned.csv \
    --workers 8 \
    --out data/meld/parametric_prosody_test_v3_raw.jsonl 2>&1 | tail -5 | tee -a "$LOG"
log "MELD test v3 raw extraction done"

# MELD train
log "extracting MELD train v3..."
"$PY" -m scripts.extract_parametric_prosody_mfa \
    --in-dir data/meld/MELD.Raw/train_splits \
    --textgrids-dir data/meld/train_textgrids_mfa \
    --metadata data/meld/MELD.Raw/train_sent_emo_cleaned.csv \
    --workers 8 \
    --out data/meld/parametric_prosody_train_v3_raw.jsonl 2>&1 | tail -5 | tee -a "$LOG"
log "MELD train v3 raw extraction done"

log "== Phase 2: AMI per-DA v3 extraction =="
"$PY" -m scripts.extract_parametric_prosody_ami_v2 \
    --manifest-glob 'data/ami/per_da_input/*/manifest.jsonl' \
    --textgrids-root data/ami/per_da_textgrids_mfa \
    --workers 8 \
    --out data/ami/parametric_prosody_ami_v3_raw.jsonl 2>&1 | tail -5 | tee -a "$LOG"
log "AMI per-DA v3 raw extraction done"

log "== Phase 3: phrase digest post-processing =="
for split in dev test train; do
  log "phrase digest on MELD $split..."
  "$PY" -m scripts.add_phrase_digest \
      --in  "data/meld/parametric_prosody_${split}_v3_raw.jsonl" \
      --out "data/meld/parametric_prosody_${split}_v3.jsonl" 2>&1 | tail -3 | tee -a "$LOG"
done

log "phrase digest on AMI per-DA..."
"$PY" -m scripts.add_phrase_digest \
    --in  data/ami/parametric_prosody_ami_v3_raw.jsonl \
    --out data/ami/parametric_prosody_ami_v3.jsonl 2>&1 | tail -3 | tee -a "$LOG"

log "== summary =="
for f in data/meld/parametric_prosody_dev_v3.jsonl \
         data/meld/parametric_prosody_test_v3.jsonl \
         data/meld/parametric_prosody_train_v3.jsonl \
         data/ami/parametric_prosody_ami_v3.jsonl; do
  if [ -f "$f" ]; then
    n=$(wc -l < "$f" | tr -d ' ')
    sz=$(ls -la "$f" | awk '{print $5}')
    log "  $f: $n lines, $sz bytes"
  fi
done

date > "$DONE"
log "== v3 extraction pipeline complete =="
log "done marker: $DONE"
