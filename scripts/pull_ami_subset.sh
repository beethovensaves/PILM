#!/usr/bin/env bash
# Pull a 30-meeting subset (10 ES + 10 IS + 10 TS, ~5 GB) from the Mac mini's
# AMI mirror to the laptop. Idempotent — only copies meetings not already
# present locally with valid IH wavs.
#
# Selection: first 10 alphabetically of each site prefix. Spans early-to-late
# of the corpus and gives 30 meetings × ~4 speakers ≈ 120 unique speakers,
# which matches MELD's effective scale for el.inf prediction probes.
#
# Usage:   bash scripts/pull_ami_subset.sh
#          (Run after the Mac mini mirror is at least partly populated.)

set -u
SRC="/Volumes/Macintosh HD-1/Users/dratsi/ami/audio"
DST="data/ami/audio"

# 10 first-alphabetical meetings per site
SUBSET=(
  ES2002a ES2002b ES2002c ES2002d ES2003a ES2003b ES2003c ES2003d ES2004a ES2004b
  IS1000a IS1000b IS1000c IS1000d IS1001a IS1001b IS1001c IS1002b IS1002c IS1002d
  TS3003a TS3003b TS3003c TS3003d TS3004a TS3004b TS3004c TS3004d TS3005a TS3005b
)

mkdir -p "$DST"
n_copied=0
n_skipped=0
n_missing=0
for m in "${SUBSET[@]}"; do
  src_dir="$SRC/$m"
  dst_dir="$DST/$m"
  if [ ! -d "$src_dir" ]; then
    echo "MISSING  $m  (source not yet present on Mac mini)"
    n_missing=$((n_missing + 1))
    continue
  fi
  mkdir -p "$dst_dir"
  for wav in "$src_dir"/Headset-*.wav; do
    [ -f "$wav" ] || continue
    base=$(basename "$wav")
    out="$dst_dir/$base"
    if [ -s "$out" ]; then
      src_sz=$(stat -f%z "$wav" 2>/dev/null || stat -c%s "$wav")
      dst_sz=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out")
      if [ "$src_sz" = "$dst_sz" ]; then
        n_skipped=$((n_skipped + 1))
        continue
      fi
    fi
    cp "$wav" "$out"
    n_copied=$((n_copied + 1))
    echo "COPY  $m/$base"
  done
done
echo ""
echo "[done] $n_copied copied, $n_skipped already present, $n_missing meetings missing on source"
total_bytes=$(du -sb "$DST" 2>/dev/null | awk '{print $1}' || du -sk "$DST" | awk '{print $1*1024}')
gb=$(echo "scale=2; $total_bytes / 1073741824" | bc 2>/dev/null || echo "?")
echo "[size] local subset is now ${gb} GB"
