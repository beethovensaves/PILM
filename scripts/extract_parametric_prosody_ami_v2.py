"""
v2 (MFA-aligned) parametric prosody extractor for AMI per-DA inputs.

Mirrors `extract_parametric_prosody_mfa.py` (MELD v2) but is driven by the
AMI per-DA manifest produced by `prepare_ami_per_da.py`. For each manifest
row it loads the per-DA WAV and its MFA-aligned TextGrid, runs the same
MFA-driven syllabification + 18-dim parametric vector computation, and
emits a JSONL row with the same downstream schema as the MELD v2 output —
plus the AMI-specific fields (das_in_seg, meeting, segment_id, channel,
global_name) that the cross-corpus probes consume.

Speaker baselines are computed per `global_name` (AMI's persistent
cross-meeting speaker identifier), matching how the v1 AMI extractor
treats the same field.

Usage:
    .venv/bin/python scripts/extract_parametric_prosody_ami_v2.py \\
        --manifest-glob 'data/ami/per_da_input/*/manifest.jsonl' \\
        --textgrids-root data/ami/per_da_textgrids_mfa \\
        --out data/ami/parametric_prosody_ami_v2.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import parselmouth

from scripts.extract_parametric_prosody_mfa import (
    INTENSITY_MIN_PITCH_HZ,
    INTENSITY_TIME_STEP_S,
    SpeakerBaseline,
    compute_speaker_baseline,
    extract_pitch,
    load_textgrid,
    parametric_vector,
    syllabify_utterance,
)
from scripts.voice_quality_features import compute_voice_quality

warnings.filterwarnings("ignore")


def textgrid_for(textgrids_root: Path, manifest_row: dict) -> Path:
    """Mirror the per_da_input directory structure under textgrids_root."""
    return textgrids_root / manifest_row["meeting"] / f"{manifest_row['utterance_id']}.TextGrid"


def _process_one_ami(task: dict) -> dict | None:
    """Worker function for parallel AMI v2 extraction."""
    wp = Path(task["wav_path"])
    tg = Path(task["tg_path"])
    baseline = task["baseline"]
    try:
        words = load_textgrid(tg)
        syllables = syllabify_utterance(words)
        if not syllables:
            return None
        sound = parselmouth.Sound(str(wp))
        pitch = extract_pitch(sound)
        intensity = sound.to_intensity(
            minimum_pitch=INTENSITY_MIN_PITCH_HZ,
            time_step=INTENSITY_TIME_STEP_S,
        )
    except Exception:
        return None

    word_syl_durs: dict[int, list[float]] = {}
    for syl in syllables:
        word_syl_durs.setdefault(id(syl.word), []).append(syl.duration_s)

    syl_records = []
    for j, syl in enumerate(syllables):
        next_s = syllables[j + 1] if j + 1 < len(syllables) else None
        vec, vf = parametric_vector(pitch, intensity, syl, next_s,
                                      word_syl_durs, baseline)
        vq = compute_voice_quality(
            sound, pitch, intensity, syl.t_start, syl.t_end,
            baseline.median_f0_hz,
        )
        syl_records.append({
            "nucleus_t_ms": round(syl.nucleus_t_s * 1000, 2),
            "t_start_ms": round(syl.t_start * 1000, 2),
            "t_end_ms": round(syl.t_end * 1000, 2),
            "phones": [p.label for p in syl.phones],
            "nucleus_phone": syl.nucleus.label,
            "word": syl.word.label,
            "vec": [None if math.isnan(x) else round(x, 4) for x in vec.tolist()],
            "vq": [None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(x, 4) for x in vq],
            "voiced_fraction": round(vf, 3),
        })
    return {
        "utterance_id": task["utterance_id"],
        "speaker_id": task["global_name"],
        "speaker_letter": task["speaker_letter"],
        "role": task.get("role", ""),
        "meeting": task["meeting"],
        "segment_id": task["segment_id"],
        "channel": task["channel"],
        "audio_path": str(wp),
        "textgrid_path": str(tg),
        "t_start_s": task["t_start_s"],
        "t_end_s": task["t_end_s"],
        "n_words": task["n_words"],
        "text": task["text"],
        "das_in_seg": task["das_in_seg"],
        "n_syllables": len(syl_records),
        "syllables": syl_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-glob", action="append", required=True,
                        help="Glob expression(s) for AMI per-DA manifest.jsonl files. Repeatable.")
    parser.add_argument("--textgrids-root", type=Path, required=True,
                        help="Root directory of MFA-aligned TextGrids; layout mirrors per_da_input.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many segments (smoke-test).")
    parser.add_argument("--min-utts-per-speaker", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for Pass 2 parallelism.")
    args = parser.parse_args()

    rows = []
    for g in args.manifest_glob:
        for path in glob.glob(g):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    print(f"[load] {len(rows)} per-DA segments")
    if not rows:
        sys.exit("no manifests matched")

    # Filter to rows with TextGrids on disk
    rows_with_tg = []
    for r in rows:
        tg = textgrid_for(args.textgrids_root, r)
        if tg.exists():
            r["_tg"] = tg
            rows_with_tg.append(r)
    print(f"  {len(rows_with_tg)}/{len(rows)} have TextGrids on disk at {args.textgrids_root}")
    if not rows_with_tg:
        sys.exit("no TextGrids found; check --textgrids-root")

    # Group by global_name for speaker baselines
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows_with_tg:
        by_speaker[r["global_name"]].append(r)
    print(f"  unique speakers (global_name): {len(by_speaker)}")

    # Pass 1 — speaker baselines from (wav, textgrid) pairs
    print(f"[pass 1] computing speaker baselines...")
    baselines: dict[str, SpeakerBaseline] = {}
    for spk, segs in by_speaker.items():
        pairs = [(Path(s["wav"]), s["_tg"]) for s in segs
                  if Path(s["wav"]).exists() and s["_tg"].exists()]
        if len(pairs) < args.min_utts_per_speaker:
            print(f"    skip {spk!r}: {len(pairs)} pairs (need >= {args.min_utts_per_speaker})")
            continue
        bl = compute_speaker_baseline(pairs)
        if bl is not None:
            baselines[spk] = bl
            print(f"    {spk!r}: median F0 {bl.median_f0_hz:.1f} Hz, n={bl.n_utterances_seen}")
    if not baselines:
        sys.exit("no usable speaker baselines computed; aborting")

    # Pass 2 — build task list, then process (sequentially or in parallel)
    print(f"[pass 2] extracting v2 parametric vectors for {len(rows_with_tg)} segments (workers={args.workers})...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    n_pre_skipped = 0
    for r in rows_with_tg:
        if args.limit and len(tasks) >= args.limit:
            break
        wp = Path(r["wav"])
        baseline = baselines.get(r["global_name"])
        if baseline is None or not wp.exists():
            n_pre_skipped += 1
            continue
        tasks.append({
            "wav_path": str(wp),
            "tg_path": str(r["_tg"]),
            "utterance_id": r["utterance_id"],
            "global_name": r["global_name"],
            "speaker_letter": r["speaker_letter"],
            "role": r.get("role", ""),
            "meeting": r["meeting"],
            "segment_id": r["segment_id"],
            "channel": r["channel"],
            "t_start_s": r["t_start_s"],
            "t_end_s": r["t_end_s"],
            "n_words": r["n_words"],
            "text": r["text"],
            "das_in_seg": r["das_in_seg"],
            "baseline": baseline,
        })
    print(f"    {len(tasks)} tasks ({n_pre_skipped} pre-skipped)")

    n_written = n_failed = 0
    with args.out.open("w") as f_out:
        if args.workers <= 1:
            for task in tasks:
                row_dict = _process_one_ami(task)
                if row_dict is None:
                    n_failed += 1
                    continue
                f_out.write(json.dumps(row_dict) + "\n")
                n_written += 1
                if n_written % 200 == 0:
                    print(f"    {n_written}/{len(tasks)} written ({n_failed} failed)")
        else:
            with mp.get_context("spawn").Pool(processes=args.workers) as pool:
                for row_dict in pool.imap_unordered(_process_one_ami, tasks, chunksize=8):
                    if row_dict is None:
                        n_failed += 1
                        continue
                    f_out.write(json.dumps(row_dict) + "\n")
                    n_written += 1
                    if n_written % 500 == 0:
                        print(f"    {n_written}/{len(tasks)} written ({n_failed} failed)")

    print(f"Done. {n_written} segments written, {n_pre_skipped + n_failed} skipped/failed.")


if __name__ == "__main__":
    main()
