#!/usr/bin/env bash
# Process one MELD split end-to-end: mp4→wav → MFA align → v2 parametric extract
# Usage: process_meld_split.sh <split_name> <split_dir> <metadata_csv>
#   split_name:   "train" | "test" | "dev" — used to name output JSONLs / dirs
#   split_dir:    name of the dir under MELD.Raw that holds the wavs/mp4s
#                 (test = "output_repeated_splits_test", train = "train_splits", dev = "dev_splits_complete")
#   metadata_csv: path to the corresponding *_sent_emo.csv

set -euo pipefail

SPLIT="${1:?usage: <split_name> <split_dir> <metadata_csv>}"
SPLIT_DIR="${2:?}"
METADATA="${3:?}"

ROOT="/Users/felipe.casadei/vscode/vsclean/PILM"
PY="$ROOT/.venv/bin/python"
AUDIO_DIR="$ROOT/data/meld/MELD.Raw/$SPLIT_DIR"
TG_DIR="$ROOT/data/meld/${SPLIT}_textgrids_mfa"
OUT_PARAMETRIC="$ROOT/data/meld/parametric_prosody_${SPLIT}_mfa.jsonl"
OUT_TARGETS="$ROOT/data/meld/emotion_probe_targets_${SPLIT}.jsonl"

echo "==[$(date +%T)] $SPLIT: convert mp4→wav (8-parallel)=="
cd "$AUDIO_DIR"
N_MP4=$(ls *.mp4 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N_MP4" -gt 0 ]]; then
    # || true so a single corrupt mp4 doesn't kill the rest of the pipeline.
    ls *.mp4 | xargs -P 8 -I {} bash -c '
        f="{}"
        out="${f%.mp4}.wav"
        if [ ! -f "$out" ]; then
            if ffmpeg -y -loglevel error -i "$f" -ac 1 -ar 16000 -vn "$out" 2>/dev/null; then
                rm "$f"
            fi
        fi
    ' || true
    # remove any mp4s that are still around (corrupt — can't convert)
    rm -f *.mp4 || true
fi
N_WAV=$(ls *.wav 2>/dev/null | wc -l | tr -d ' ')
echo "==[$(date +%T)] $SPLIT: wav files now: $N_WAV (was $N_MP4 mp4s)=="

cd "$ROOT"

echo "==[$(date +%T)] $SPLIT: write .lab transcripts=="
"$PY" scripts/prepare_mfa_transcripts.py --in-dir "$AUDIO_DIR" --metadata "$METADATA"

echo "==[$(date +%T)] $SPLIT: MFA force align=="
mkdir -p "$TG_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mfa-env
mfa align "$AUDIO_DIR" english_us_arpa english_us_arpa "$TG_DIR" --num_jobs 8 --clean

echo "==[$(date +%T)] $SPLIT: relocate any TextGrids that landed in source dir=="
shopt -s nullglob
moved=0
for tg in "$AUDIO_DIR"/*.TextGrid; do
    mv "$tg" "$TG_DIR/"
    ((moved++))
done
shopt -u nullglob
echo "  moved $moved"
N_TG=$(ls "$TG_DIR"/*.TextGrid 2>/dev/null | wc -l | tr -d ' ')
echo "  total TextGrids: $N_TG"

conda deactivate

echo "==[$(date +%T)] $SPLIT: run v2 parametric extractor=="
"$PY" scripts/extract_parametric_prosody_mfa.py \
    --in-dir "$AUDIO_DIR" \
    --textgrids-dir "$TG_DIR" \
    --metadata "$METADATA" \
    --out "$OUT_PARAMETRIC"

echo "==[$(date +%T)] $SPLIT: build emotion probe targets=="
"$PY" scripts/build_emotion_probe_targets.py --metadata "$METADATA" --out "$OUT_TARGETS"

echo "==[$(date +%T)] $SPLIT: DONE=="
echo "  parametric: $OUT_PARAMETRIC"
echo "  targets:    $OUT_TARGETS"
echo "  textgrids:  $TG_DIR"
