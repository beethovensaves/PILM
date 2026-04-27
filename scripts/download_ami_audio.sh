#!/usr/bin/env bash
# Download AMI Individual Headset (IH) audio for all 139 meetings to the
# Mac mini network share. IH is the prosody-grade per-speaker audio; we skip
# Mix-Headset (we already have 3 mix files locally), far-field arrays, and video.
#
# Per-meeting: tries Headset-0.wav through Headset-4.wav (most have 4 channels,
# a few 3 or 5). Skips files already downloaded; resumes partial downloads via curl -C -.
#
# Output:  $DEST/audio/<meeting>/Headset-{0..4}.wav  (~22 GB total)
# Log:     $DEST/download_ih.log
#
# Usage:   bash scripts/download_ami_audio.sh
#          (Designed to run in the background; tail the log to monitor.)

set -u
DEST="/Volumes/Macintosh HD-1/Users/dratsi/ami"
ROOT="$DEST/audio"
LOG="$DEST/download_ih.log"
MEETINGS="/tmp/ami_meetings.txt"
BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

mkdir -p "$ROOT"
: > "$LOG"
echo "[start] $(date)" >> "$LOG"
echo "[dest]  $ROOT" >> "$LOG"

n_meetings=0
n_ok=0
n_404=0
n_fail=0
total_bytes=0

while IFS= read -r m; do
  n_meetings=$((n_meetings + 1))
  mdir="$ROOT/$m"
  mkdir -p "$mdir"
  for ch in 0 1 2 3 4; do
    out="$mdir/Headset-${ch}.wav"
    url="$BASE/$m/audio/$m.Headset-${ch}.wav"
    if [ -s "$out" ]; then
      sz=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out")
      if [ "$sz" -gt 1000 ]; then
        echo "SKIP  $m Headset-${ch}  ($sz bytes already present)" >> "$LOG"
        n_ok=$((n_ok + 1))
        total_bytes=$((total_bytes + sz))
        continue
      fi
    fi
    code=$(curl -sL -w "%{http_code}" -o "$out" -C - "$url")
    if [ "$code" = "200" ] || [ "$code" = "206" ]; then
      sz=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out")
      if [ "$sz" -lt 1000 ]; then
        rm -f "$out"
        echo "FAIL  $m Headset-${ch}  http=$code size=$sz (truncated)" >> "$LOG"
        n_fail=$((n_fail + 1))
      else
        echo "OK    $m Headset-${ch}  $sz bytes" >> "$LOG"
        n_ok=$((n_ok + 1))
        total_bytes=$((total_bytes + sz))
      fi
    elif [ "$code" = "404" ]; then
      rm -f "$out"
      echo "404   $m Headset-${ch}  (channel does not exist)" >> "$LOG"
      n_404=$((n_404 + 1))
    else
      echo "FAIL  $m Headset-${ch}  http=$code" >> "$LOG"
      n_fail=$((n_fail + 1))
    fi
  done
  if [ $((n_meetings % 5)) -eq 0 ]; then
    gb=$(echo "scale=2; $total_bytes / 1073741824" | bc 2>/dev/null || echo "?")
    echo "[progress] $n_meetings/139 meetings done, ${n_ok} OK, ${n_404} 404, ${n_fail} fail, ${gb} GB" >> "$LOG"
  fi
done < "$MEETINGS"

gb=$(echo "scale=2; $total_bytes / 1073741824" | bc 2>/dev/null || echo "?")
echo "[done] $(date)" >> "$LOG"
echo "[done] $n_meetings meetings processed, $n_ok OK files, $n_404 channels-not-present, $n_fail failures, ${gb} GB total" >> "$LOG"
