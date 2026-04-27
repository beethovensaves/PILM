#!/usr/bin/env bash
# Wait for v2 extraction to finish, then run probes + write findings + done marker.
# Designed to chain after `extract_parametric_prosody_ami_v2.py` running in background.
set -u
ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/data/ami/exp010_v2.log"
DONE="$ROOT/data/ami/exp010_v2_done.marker"
EXTRACT_PID="${1:?extract pid arg required}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "== post-extraction chain: waiting for extraction PID $EXTRACT_PID =="
while kill -0 "$EXTRACT_PID" 2>/dev/null; do
  sleep 30
done
log "extraction process $EXTRACT_PID exited"

n_v2=$(wc -l < data/ami/parametric_prosody_ami_v2.jsonl 2>/dev/null | tr -d ' ')
log "parametric_prosody_ami_v2.jsonl has $n_v2 lines"
if [ "$n_v2" -lt 1000 ]; then
  log "FATAL: extraction produced too few segments"
  exit 1
fi

log "== running cross-corpus probes (AMI v2 vs MELD v2) =="
"$PY" -m scripts.run_exp010_probes \
    --ami-parametric data/ami/parametric_prosody_ami_v2.jsonl \
    --meld-train-parametric data/meld/parametric_prosody_train_mfa.jsonl \
    --meld-test-parametric data/meld/parametric_prosody_test_mfa.jsonl \
    --meld-train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \
    --meld-test-csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \
    --out data/ami/exp010_results_v2.json 2>&1 | tee -a "$LOG"

log "== appending findings =="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
from datetime import date
r = json.loads(Path("data/ami/exp010_results_v2.json").read_text())

ami_pooled = r["prosody_pooled_lr"]["auc_mean"]
ami_last = r["prosody_last_syl_lr"]["auc_mean"]
ami_small = r["prosody_bilstm_small"]["auc_mean"]
ami_rich = r["prosody_bilstm_rich"]["auc_mean"]
ami_speaker = r["speaker_held_out_last_syl"]["auc_mean"]
pos = r["position_ablation"]
pf, pm, pl = pos["first_syl"]["auc_mean"], pos["middle_syl"]["auc_mean"], pos["last_syl"]["auc_mean"]
text_uplift = r["text_vs_prosody"]["combined"]["mean"] - r["text_vs_prosody"]["text"]["mean"]
meld_v2_last = r["meld_v1_yn_q_last_syl_auc"]
meld_v2_pooled = r["meld_v1_yn_q_pooled_auc"]
gap = ami_last - meld_v2_last
stable = [c for c in r["bootstrap_coefs"][:6] if c["sign_stable"]]

block = f"""

## EXP-010 v2 — MFA-aligned cross-corpus question prediction on AMI ({date.today().isoformat()})

The MFA-aligned (v2) parametric pipeline was applied to the per-DA AMI subset, enabling apples-to-apples comparison with MELD's EXP-007 v2 numbers. AMI probe target is the `Elicit-Inform` dialogue act vs clear-statement DAs; MELD probe target is utterance-final question-mark presence (yes/no questions only).

**Prosody-only AUC for predicting `el.inf` vs statement DAs**, 5-fold CV, AMI v2: pooled LR = {ami_pooled:.4f}; last-syllable LR = {ami_last:.4f}; bi-LSTM (mean-pool, small) = {ami_small:.4f}; bi-LSTM (attention-pool, rich) = {ami_rich:.4f}.

**Cross-corpus comparison at v2 quality.** AMI `el.inf` last-syl AUC = {ami_last:.4f}; MELD yn-Q last-syl AUC (MFA-aligned) = {meld_v2_last:.4f}; gap = {gap:+.4f}.

**Speaker-held-out** (GroupKFold by `global_name`) AUC = {ami_speaker:.4f}.

**Position ablation on AMI v2:** first = {pf:.4f}; middle = {pm:.4f}; last = {pl:.4f}.

**Text vs prosody on AMI v2:** combined uplift over text alone = {text_uplift:+.4f} macro-F1.

**Bootstrap-stable coefficients in top 6:** {len(stable)}/6 sign-stable {{{", ".join(c["dim"] for c in stable)}}}.
"""
fp = Path("docs/findings.md")
fp.write_text(fp.read_text() + block)
print("[ok] appended v2 block to docs/findings.md")

dp = Path("docs/diary.md")
diary = f"""

## {date.today().isoformat()} — EXP-010 v2 (MFA-aligned per-DA AMI) complete

MFA-aligned cross-corpus probe finished: AMI el.inf last-syl AUC v2 = {ami_last:.4f}; pooled = {ami_pooled:.4f}; bi-LSTM rich = {ami_rich:.4f}. MELD yn-Q last-syl AUC v2 (apples-to-apples) = {meld_v2_last:.4f}. Gap = {gap:+.4f}. AMI v2 position ablation: first/middle/last = {pf:.4f}/{pm:.4f}/{pl:.4f}. AMI v2 text+prosody uplift = {text_uplift:+.4f}. AMI v2 bootstrap-stable coefficients in top 6: {len(stable)}/6.
Results in data/ami/exp010_results_v2.json.
"""
dp.write_text(diary + dp.read_text())
print("[ok] prepended v2 entry to docs/diary.md")
PYEOF

date > "$DONE"
log "== AMI v2 pipeline complete (after fix) =="
log "results: data/ami/exp010_results_v2.json"
log "done marker: $DONE"
