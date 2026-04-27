"""
Merge per-utterance MELD wavs into per-dialogue wavs.

For each Dialogue_ID, collects all utterance clips, sorts by StartTime,
positions each clip on a common timeline at (utt_start − dialogue_start),
and writes a single dialogue.wav. Silence between utterances is preserved
(matches the original episode dynamics).

Overlap policy: **first-wins.** Where a later clip overlaps a region
already filled by an earlier clip, keep the earlier audio. Handles both:
    - exact-duplicate audio (MELD sometimes extracts the same chunk twice
      for adjacent utterances) — deduped cleanly.
    - true overlapping speech (one clip's bleed from another speaker) —
      keeps the cleaner first clip.

Output also includes a JSON summary with per-dialogue placement stats.

Usage:
    .venv/bin/python scripts/merge_meld_dialogues.py \\
        --in-dir data/meld/MELD.Raw/dev_splits_complete \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --out-dir data/meld/dev_dialogues
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def to_seconds(t: str) -> float:
    """Parse 'HH:MM:SS,mmm' or 'H:MM:SS,mmm' (some MELD rows drop the leading zero)."""
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def merge_dialogue(
    group: pd.DataFrame,
    audio_dir: Path,
    sample_rate: int,
) -> tuple[np.ndarray, dict]:
    g = group.copy()
    g["start_s"] = g["StartTime"].apply(to_seconds)
    g["end_s"] = g["EndTime"].apply(to_seconds)
    g = g.sort_values("start_s").reset_index(drop=True)

    dialogue_start = float(g["start_s"].min())
    dialogue_end = float(g["end_s"].max())
    duration_s = dialogue_end - dialogue_start
    # +1 second safety margin in case a clip extends slightly past EndTime
    n_samples = int((duration_s + 1.0) * sample_rate)

    buf = np.zeros(n_samples, dtype=np.float32)
    occupied = np.zeros(n_samples, dtype=bool)

    n_placed = 0
    n_skipped_missing = 0
    n_skipped_sr = 0
    overlap_dropped_s = 0.0
    placement_log: list[dict] = []

    for row in g.itertuples():
        wav_path = audio_dir / f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav"
        if not wav_path.exists():
            n_skipped_missing += 1
            placement_log.append({
                "utt_id": int(row.Utterance_ID),
                "status": "missing",
            })
            continue
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            n_skipped_missing += 1
            placement_log.append({
                "utt_id": int(row.Utterance_ID),
                "status": f"read_error: {e}",
            })
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # safety; the cache is already mono
        if sr != sample_rate:
            n_skipped_sr += 1
            placement_log.append({
                "utt_id": int(row.Utterance_ID),
                "status": f"unexpected_sr: {sr}",
            })
            continue

        offset_s = float(row.start_s) - dialogue_start
        offset_samples = int(round(offset_s * sample_rate))
        end_samples = offset_samples + len(audio)
        # Extend buffers if a clip runs past the +1s margin
        if end_samples > len(buf):
            grow = end_samples - len(buf)
            buf = np.concatenate([buf, np.zeros(grow, dtype=np.float32)])
            occupied = np.concatenate([occupied, np.zeros(grow, dtype=bool)])

        # First-wins: only fill samples that are not already occupied
        target = buf[offset_samples:end_samples]
        target_occ = occupied[offset_samples:end_samples]
        write_mask = ~target_occ
        # Apply audio only where unoccupied; leave already-filled samples untouched
        target[write_mask] = audio[: len(target)][write_mask]
        # Mark as occupied regardless (so a later clip can't overwrite this region either)
        target_occ[:] = True

        n_overlap_samples = int((~write_mask).sum())
        overlap_dropped_s += n_overlap_samples / sample_rate
        n_placed += 1
        placement_log.append({
            "utt_id": int(row.Utterance_ID),
            "status": "placed",
            "offset_s": round(offset_s, 3),
            "duration_s": round(len(audio) / sample_rate, 3),
            "overlap_dropped_s": round(n_overlap_samples / sample_rate, 3),
        })

    # Trim trailing silence (everything past the last occupied sample)
    if occupied.any():
        last_filled = int(np.flatnonzero(occupied)[-1]) + 1
        buf = buf[:last_filled]

    stats = {
        "n_placed": n_placed,
        "n_skipped_missing": n_skipped_missing,
        "n_skipped_sr": n_skipped_sr,
        "duration_s": round(len(buf) / sample_rate, 3),
        "overlap_dropped_s": round(overlap_dropped_s, 3),
        "placements": placement_log,
    }
    return buf, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="Per-utterance wav dir (e.g. data/meld/MELD.Raw/dev_splits_complete)")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="MELD CSV (e.g. dev_sent_emo.csv)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to write merged dia<N>.wav")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Expected sample rate of input wavs (default 16000 — matches our wav cache).")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for dlg, group in df.groupby("Dialogue_ID"):
        buf, stats = merge_dialogue(group, args.in_dir, args.sample_rate)
        if len(buf) == 0:
            print(f"  dia{dlg}: empty (no clips placed); skipping write")
            continue
        out_path = args.out_dir / f"dia{int(dlg)}.wav"
        sf.write(str(out_path), buf, args.sample_rate)
        record = {
            "dialogue_id": int(dlg),
            "output": str(out_path),
            **{k: v for k, v in stats.items() if k != "placements"},
            "placements": stats["placements"],
        }
        summary.append(record)
        if int(dlg) % 20 == 0 or stats["overlap_dropped_s"] > 1.0:
            print(f"  dia{int(dlg):3d}: placed={stats['n_placed']:2d}  "
                  f"dur={stats['duration_s']:6.2f}s  "
                  f"overlap_dropped={stats['overlap_dropped_s']:5.2f}s"
                  f"{'  (suspicious)' if stats['overlap_dropped_s'] > 3.0 else ''}")

    summary_path = args.out_dir / "merge_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    total_dialogues = len(summary)
    total_placed = sum(s["n_placed"] for s in summary)
    total_dropped = sum(s["overlap_dropped_s"] for s in summary)
    total_duration = sum(s["duration_s"] for s in summary)
    print(f"\nDone. {total_dialogues} dialogues merged.")
    print(f"  total utterances placed: {total_placed}")
    print(f"  total dialogue audio:    {total_duration / 60:.1f} min")
    print(f"  total overlap dropped:   {total_dropped:.1f} s "
          f"({100 * total_dropped / total_duration:.1f}% of total)")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
