"""
Produce a dialogue-aligned version of dev_sent_emo.csv that matches the
per-dialogue merged wavs from `scripts/merge_meld_dialogues.py`.

What changes vs. the original CSV:
  - StartTime / EndTime (episode-relative) → `start_s` / `end_s` (offset
    in seconds within the merged dia<N>.wav).
  - The original episode timestamps are kept as `episode_start` /
    `episode_end` for reference.
  - Adds `overlap_dropped_s` from the merge summary so consumers know
    when the merge dropped audio for that utterance under first-wins.
  - Adds `merge_status` (placed / missing / read_error / etc.).

Outlier handling:
  - A dialogue is an outlier if any inter-utterance gap exceeds
    `--max-gap-s` (default 30s). At that point the dialogue spans a
    scene boundary and the merged wav has multi-minute dead air.
  - Outlier rows are dropped from the new CSV.
  - Outlier merged wavs are moved to <dialogues-dir>/_outliers/ .
  - The outlier set is logged to <out-csv parent>/outliers.json.

Usage:
    .venv/bin/python scripts/reformat_meld_metadata.py \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --merge-summary data/meld/dev_dialogues/merge_summary.json \\
        --dialogues-dir data/meld/dev_dialogues \\
        --out-csv data/meld/dev_sent_emo_dialogue_aligned.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def to_seconds(t: str) -> float:
    """Parse 'HH:MM:SS,mmm' / 'H:MM:SS,mmm' to seconds."""
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def detect_outliers(df: pd.DataFrame, max_gap_s: float) -> tuple[set[int], dict[int, float]]:
    outliers: set[int] = set()
    max_gap_per_dlg: dict[int, float] = {}
    for dlg, group in df.groupby("Dialogue_ID"):
        g = group.copy()
        g["start_s"] = g["StartTime"].apply(to_seconds)
        g["end_s"] = g["EndTime"].apply(to_seconds)
        g = g.sort_values("start_s").reset_index(drop=True)
        if len(g) < 2:
            max_gap_per_dlg[int(dlg)] = 0.0
            continue
        gaps = g["start_s"].values[1:] - g["end_s"].values[:-1]
        max_gap = float(gaps.max())
        max_gap_per_dlg[int(dlg)] = max_gap
        if max_gap > max_gap_s:
            outliers.add(int(dlg))
    return outliers, max_gap_per_dlg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--merge-summary", type=Path, required=True)
    parser.add_argument("--dialogues-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--max-gap-s", type=float, default=30.0,
                        help="Dialogues with any inter-utterance gap above this become outliers.")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"loaded {len(df)} rows ({df['Dialogue_ID'].nunique()} dialogues)")

    # Load merge summary → map dialogue_id → {utt_id: placement_record}
    with args.merge_summary.open() as f:
        merge_data = json.load(f)
    placement_map: dict[int, dict[int, dict]] = {}
    for entry in merge_data:
        dlg = int(entry["dialogue_id"])
        placement_map[dlg] = {int(p["utt_id"]): p for p in entry["placements"]}

    # Outlier detection
    outliers, max_gaps = detect_outliers(df, args.max_gap_s)
    print(f"\noutliers (any inter-utterance gap > {args.max_gap_s}s): {len(outliers)}")
    for d in sorted(outliers):
        n_utts = (df["Dialogue_ID"] == d).sum()
        print(f"  dia{d:3d}: max gap {max_gaps[d]:.1f}s, {n_utts} utterances")

    # Move outlier wav files
    outlier_dir = args.dialogues_dir / "_outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for d in outliers:
        src = args.dialogues_dir / f"dia{d}.wav"
        if src.exists():
            dst = outlier_dir / f"dia{d}.wav"
            shutil.move(str(src), str(dst))
            moved.append(f"dia{d}.wav")
    if moved:
        print(f"  moved {len(moved)} wavs to {outlier_dir}")

    # Per-dialogue: dialogue start in seconds (= min of utterance starts)
    df["abs_start_s"] = df["StartTime"].apply(to_seconds)
    df["abs_end_s"] = df["EndTime"].apply(to_seconds)
    dialogueabs_start_s = df.groupby("Dialogue_ID")["abs_start_s"].min().to_dict()

    # Build the dialogue-aligned rows
    out_rows: list[dict] = []
    n_excluded = 0
    for row in df.itertuples():
        dlg = int(row.Dialogue_ID)
        if dlg in outliers:
            n_excluded += 1
            continue
        utt = int(row.Utterance_ID)
        dlg_start = dialogueabs_start_s[dlg]
        placement = placement_map.get(dlg, {}).get(utt, {})

        out_rows.append({
            "Sr_No":            getattr(row, "_1"),  # 'Sr No.' column auto-renamed to _1
            "Utterance":        row.Utterance,
            "Speaker":          row.Speaker,
            "Emotion":          row.Emotion,
            "Sentiment":        row.Sentiment,
            "Dialogue_ID":      dlg,
            "Utterance_ID":     utt,
            "Season":           int(row.Season),
            "Episode":          int(row.Episode),
            "start_s":          round(row.abs_start_s - dlg_start, 3),
            "end_s":            round(row.abs_end_s - dlg_start, 3),
            "duration_s":       round(row.abs_end_s - row.abs_start_s, 3),
            "overlap_dropped_s": placement.get("overlap_dropped_s", 0.0),
            "merge_status":     placement.get("status", "missing"),
            "audio_path":       f"{args.dialogues_dir.name}/dia{dlg}.wav",
            "episode_start":    row.StartTime,
            "episode_end":      row.EndTime,
        })

    out_df = pd.DataFrame(out_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {len(out_df)} rows to {args.out_csv}")
    print(f"  excluded {n_excluded} outlier-dialogue rows")
    print(f"  unique dialogues kept: {out_df['Dialogue_ID'].nunique()}")
    print(f"  unique speakers: {out_df['Speaker'].nunique()}")
    counts = out_df["Emotion"].value_counts().to_dict()
    print(f"  emotion distribution: {counts}")

    # Side-log: outliers + their utterance counts
    outliers_log = {
        "max_gap_threshold_s": args.max_gap_s,
        "outlier_dialogues": [
            {
                "dialogue_id": d,
                "max_gap_s":   max_gaps[d],
                "n_utterances": int((df["Dialogue_ID"] == d).sum()),
                "wav_relocated_to": str(outlier_dir / f"dia{d}.wav"),
            }
            for d in sorted(outliers)
        ],
        "n_excluded_rows": n_excluded,
    }
    log_path = args.out_csv.parent / "outliers.json"
    with log_path.open("w") as f:
        json.dump(outliers_log, f, indent=2)
    print(f"  outliers log: {log_path}")


if __name__ == "__main__":
    main()
