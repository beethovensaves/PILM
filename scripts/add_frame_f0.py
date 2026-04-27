"""
Augment a parametric-prosody JSONL with frame-level F0 samples per syllable.

For each syllable we sample F0 at K equally-spaced timepoints between
t_start_ms and t_end_ms, convert to semitones relative to the speaker's
median F0 (computed in pass 1, like the rest of the parametric vector),
and append `frame_f0_st: [...]` (length K) to the syllable record.

This is the D20 ablation feature — gives the encoder access to the
microprosodic shape of the F0 contour that the 18-dim parametric vector
aggregates away. Unvoiced frames and missing F0 → NaN serialized as null.

Usage:
    .venv/bin/python scripts/add_frame_f0.py \\
        --in data/meld/parametric_prosody_dev_mfa.jsonl \\
        --in-dir data/meld/MELD.Raw/dev_splits_complete \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --out data/meld/parametric_prosody_dev_mfa_framef0.jsonl \\
        --k 16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

# Same pitch parameters as extract_parametric_prosody_mfa.py
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 600.0
PITCH_TIME_STEP_S = 0.005


def hz_to_st(hz: float, ref_hz: float) -> float:
    if not (hz and hz > 0 and ref_hz and ref_hz > 0):
        return float("nan")
    return 12.0 * math.log2(hz / ref_hz)


def extract_pitch(sound: parselmouth.Sound) -> parselmouth.Pitch:
    return call(
        sound, "To Pitch (cc)",
        PITCH_TIME_STEP_S, PITCH_FLOOR_HZ, 15, "no",
        0.03, 0.45, 0.01, 0.35, 0.14, PITCH_CEILING_HZ,
    )


def compute_speaker_f0_medians(in_dir: Path, df: pd.DataFrame) -> dict[str, float]:
    """Pass 1: median F0 per speaker."""
    medians: dict[str, float] = {}
    for spk, group in df.groupby("Speaker"):
        f0_pool: list[float] = []
        for r in group.itertuples():
            wp = in_dir / f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.wav"
            if not wp.exists():
                wp = in_dir / f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.mp4"
                if not wp.exists():
                    continue
            try:
                snd = parselmouth.Sound(str(wp))
                pitch = extract_pitch(snd)
                f0_pool.extend(float(v) for v in pitch.selected_array["frequency"] if v > 0)
            except Exception:
                continue
        if f0_pool:
            medians[spk] = float(np.median(f0_pool))
    return medians


def sample_f0_in_window(pitch: parselmouth.Pitch, t_start_s: float, t_end_s: float,
                       k: int, ref_hz: float) -> list[float | None]:
    """Sample F0 at K equally-spaced timepoints between t_start and t_end."""
    if k < 2 or t_end_s <= t_start_s:
        return [None] * k
    times = np.linspace(t_start_s, t_end_s, k)
    out: list[float | None] = []
    for t in times:
        try:
            hz = call(pitch, "Get value at time", float(t), "Hertz", "Linear")
        except Exception:
            hz = None
        if hz is None or (isinstance(hz, float) and math.isnan(hz)):
            out.append(None)
        else:
            st = hz_to_st(float(hz), ref_hz)
            out.append(None if math.isnan(st) else round(st, 4))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",       dest="in_path",   type=Path, required=True)
    parser.add_argument("--in-dir",                     type=Path, required=True)
    parser.add_argument("--metadata",                   type=Path, required=True)
    parser.add_argument("--out",                        type=Path, required=True)
    parser.add_argument("--k", type=int, default=16,
                        help="Number of equally-spaced F0 samples per syllable")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"[pass 1] computing speaker F0 medians...")
    medians = compute_speaker_f0_medians(args.in_dir, df)
    print(f"  speakers with usable F0: {len(medians)}")

    speaker_lookup: dict[str, str] = {}
    for r in df.itertuples():
        speaker_lookup[f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"] = r.Speaker

    print(f"[pass 2] sampling {args.k}-point F0 contour per syllable...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cached_pitch: dict[str, parselmouth.Pitch] = {}
    n = 0
    n_skipped_no_speaker = 0
    n_skipped_no_audio = 0
    with args.in_path.open() as f_in, args.out.open("w") as f_out:
        for line in f_in:
            d = json.loads(line)
            uid = d["utterance_id"]
            speaker = speaker_lookup.get(uid)
            ref_hz = medians.get(speaker)
            if ref_hz is None:
                n_skipped_no_speaker += 1
                # still pass through, but with all-None frame_f0
                for syl in d["syllables"]:
                    syl["frame_f0_st"] = [None] * args.k
                f_out.write(json.dumps(d) + "\n")
                n += 1
                continue
            audio_path = Path(d.get("audio_path", ""))
            if not audio_path.exists():
                n_skipped_no_audio += 1
                for syl in d["syllables"]:
                    syl["frame_f0_st"] = [None] * args.k
                f_out.write(json.dumps(d) + "\n")
                n += 1
                continue
            try:
                sound = parselmouth.Sound(str(audio_path))
                pitch = extract_pitch(sound)
            except Exception:
                for syl in d["syllables"]:
                    syl["frame_f0_st"] = [None] * args.k
                f_out.write(json.dumps(d) + "\n")
                n += 1
                continue
            for syl in d["syllables"]:
                t_start = syl["t_start_ms"] / 1000.0
                t_end   = syl["t_end_ms"]   / 1000.0
                syl["frame_f0_st"] = sample_f0_in_window(pitch, t_start, t_end, args.k, ref_hz)
            f_out.write(json.dumps(d) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  written {n}")
    print(f"Done. {n} utterances. skipped_no_speaker={n_skipped_no_speaker} skipped_no_audio={n_skipped_no_audio}")


if __name__ == "__main__":
    main()
