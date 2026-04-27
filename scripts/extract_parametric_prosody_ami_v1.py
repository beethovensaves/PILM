"""
v1 18-dim parametric prosody extractor for AMI — same algorithm as
`extract_parametric_prosody.py` (Praat de Jong syllable-nucleus heuristic +
two-pass speaker baseline), driven by AMI's per-segment WAV manifests
produced by `prepare_ami_for_mfa.py`.

The point: get a v1 result on AMI tonight that is apples-to-apples
comparable to a v1 rerun on MELD. MFA/v2 upgrade is a follow-up.

Speaker baselining: AMI speakers are identified by their *global_name*
(e.g., MEE006), which persists across meetings. So one baseline per
global_name aggregates every meeting that speaker appears in — a meaningful
speaker baseline rather than a per-meeting one.

Inputs:
    --manifest-glob : glob like 'data/ami/mfa_input/*/manifest.jsonl'
                       (each manifest emitted by prepare_ami_for_mfa.py)

Output: one JSONL line per AMI segment, same dim ordering as MELD v1, plus
AMI-specific fields (das_in_seg, role, meeting, segment_id, channel, etc.)
so downstream probes can directly use DA labels.

Usage:
    .venv/bin/python scripts/extract_parametric_prosody_ami_v1.py \\
        --manifest-glob 'data/ami/mfa_input/*/manifest.jsonl' \\
        --out data/ami/parametric_prosody_ami_v1.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

import parselmouth

from scripts.extract_parametric_prosody import (
    INTENSITY_MIN_PITCH_HZ,
    INTENSITY_TIME_STEP_S,
    SpeakerBaseline,
    compute_speaker_baseline,
    detect_syllable_nuclei,
    extract_pitch,
    parametric_vector,
)

warnings.filterwarnings("ignore")


def load_manifests(globs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for g in globs:
        for path in glob.glob(g):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-glob", action="append", required=True,
                        help="Glob expression(s) for AMI per-meeting manifest.jsonl files. "
                             "Repeatable.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many segments (smoke-test).")
    parser.add_argument("--min-utts-per-speaker", type=int, default=3)
    args = parser.parse_args()

    rows = load_manifests(args.manifest_glob)
    print(f"[load] {len(rows)} segments from {len(args.manifest_glob)} manifest glob(s)")
    if not rows:
        sys.exit("no segments loaded; check --manifest-glob")

    # Group by speaker (global_name across meetings)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[r["global_name"]].append(r)
    print(f"  unique speakers (global_name): {len(by_speaker)}")
    print(f"  unique meetings: {len({r['meeting'] for r in rows})}")

    # Pass 1 — speaker baselines from ALL of that speaker's segment wavs
    print(f"[pass 1] computing speaker baselines...")
    baselines: dict[str, SpeakerBaseline] = {}
    for spk, segs in by_speaker.items():
        wav_paths = [Path(s["wav"]) for s in segs if Path(s["wav"]).exists()]
        if len(wav_paths) < args.min_utts_per_speaker:
            print(f"    skip {spk!r}: {len(wav_paths)} segments (need >= {args.min_utts_per_speaker})")
            continue
        bl = compute_speaker_baseline(wav_paths)
        if bl is not None:
            baselines[spk] = bl
            print(f"    {spk!r}: median F0 {bl.median_f0_hz:.1f} Hz, n={bl.n_utterances_seen} segments")
    if not baselines:
        sys.exit("no usable speaker baselines computed; aborting")

    # Pass 2 — extract parametric vectors
    print(f"[pass 2] extracting parametric vectors for {len(rows)} segments...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    with args.out.open("w") as f_out:
        for r in rows:
            if args.limit and n_written >= args.limit:
                break
            wp = Path(r["wav"])
            if not wp.exists():
                n_skipped += 1
                continue
            baseline = baselines.get(r["global_name"])
            if baseline is None:
                n_skipped += 1
                continue
            try:
                sound = parselmouth.Sound(str(wp))
                pitch = extract_pitch(sound)
                intensity = sound.to_intensity(
                    minimum_pitch=INTENSITY_MIN_PITCH_HZ,
                    time_step=INTENSITY_TIME_STEP_S,
                )
                spans = detect_syllable_nuclei(sound, pitch)
            except Exception as e:
                print(f"    error on {wp.name}: {e}")
                n_skipped += 1
                continue

            syllable_records = []
            for j, span in enumerate(spans):
                next_s = spans[j + 1] if j + 1 < len(spans) else None
                vec, vf = parametric_vector(sound, pitch, intensity, span, next_s, baseline)
                syllable_records.append({
                    "nucleus_t_ms": round(span.nucleus_t_s * 1000, 2),
                    "t_start_ms": round(span.t_start_s * 1000, 2),
                    "t_end_ms": round(span.t_end_s * 1000, 2),
                    "vec": [None if math.isnan(x) else round(x, 4) for x in vec.tolist()],
                    "voiced_fraction": round(vf, 3),
                })
            f_out.write(json.dumps({
                "utterance_id": r["utterance_id"],
                "speaker_id": r["global_name"],
                "speaker_letter": r["speaker_letter"],
                "role": r.get("role", ""),
                "meeting": r["meeting"],
                "segment_id": r["segment_id"],
                "channel": r["channel"],
                "audio_path": str(wp),
                "t_start_s": r["t_start_s"],
                "t_end_s": r["t_end_s"],
                "n_words": r["n_words"],
                "text": r["text"],
                "das_in_seg": r["das_in_seg"],
                "n_syllables": len(syllable_records),
                "syllables": syllable_records,
            }) + "\n")
            n_written += 1
            if n_written % 100 == 0:
                print(f"    {n_written}/{len(rows)} written ({n_skipped} skipped)")

    print(f"Done. {n_written} segments written, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
