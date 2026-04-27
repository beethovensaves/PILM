"""
Extract the 18-dim per-syllable parametric prosody vector (D19) from MELD-style
wav files. Output is JSONL, one line per utterance.

Two-pass design:
    Pass 1 — compute per-speaker baselines (median F0, syllable-duration
             distribution, energy distribution) by walking each speaker's wavs.
    Pass 2 — re-walk wavs, detect syllables, compute the 18-dim vector
             per syllable using the speaker's baseline.

The 18 dimensions, in vector order:
    pitch geometry (7):
        0  f0_onset_st           — F0 at syllable onset, semitones rel. speaker median
        1  f0_nucleus_st         — F0 at syllable nucleus
        2  f0_offset_st          — F0 at syllable offset
        3  f0_max_st             — peak F0 within syllable
        4  f0_min_st             — minimum F0 within syllable
        5  f0_range_st           — f0_max_st − f0_min_st
        6  f0_slope_st_per_ms    — (f0_offset_st − f0_onset_st) / duration_ms
    tilt event geometry (4):
        7  f0_peak_position_norm — peak time / syllable duration ∈ [0, 1]
        8  f0_rise_amplitude_st  — f0_max_st − f0_onset_st
        9  f0_fall_amplitude_st  — f0_max_st − f0_offset_st
        10 tilt                  — (rise − fall) / (rise + fall) ∈ [−1, +1]
    energy (2):
        11 rms_max_z             — peak intensity, z-scored against speaker
        12 rms_mean_z            — mean intensity, z-scored against speaker
    duration (2):
        13 syllable_duration_z   — syllable duration vs. speaker syllable distribution
        14 nucleus_duration_z    — vowel-only duration vs. speaker (v1 approx; see writeup)
    boundary (3):
        15 pause_after_ms        — silence to next syllable
        16 final_lengthening_ratio — this duration / mean of preceding word's syllables (v1 placeholder = 1.0)
        17 f0_reset_st           — next syllable's onset minus this syllable's offset

A companion `voiced_fraction ∈ [0, 1]` is reported alongside (not part of the 18).
When voiced_fraction == 0, dims 0–10 are NaN; the consumer must mask them.

Usage:
    .venv/bin/python scripts/extract_parametric_prosody.py \\
        --in-dir data/meld/MELD.Raw/dev_splits_complete \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --out data/meld/parametric_prosody_dev.jsonl

For full design rationale see docs/design_decisions.md (D19) and
docs/writeups/parametric_prosody_pivot.md. For an implementation walkthrough
see docs/writeups/parametric_prosody_extractor.md.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

# ---------------------------------------------------------------------------
# Constants — pitch / intensity extraction parameters
# ---------------------------------------------------------------------------

PITCH_FLOOR_HZ = 75.0          # below typical adult male F0 floor
PITCH_CEILING_HZ = 600.0       # above typical adult female F0 ceiling
PITCH_TIME_STEP_S = 0.005      # 200 Hz frame rate
INTENSITY_MIN_PITCH_HZ = 50.0  # for intensity extraction
INTENSITY_TIME_STEP_S = 0.005

# Syllable-nucleus detection (de Jong & Wempe 2009, simplified)
SILENCE_DB_BELOW_PEAK = 25.0   # peak must be within 25 dB of intensity max
MIN_DIP_DB_BETWEEN = 2.0       # consecutive peaks must be separated by ≥2 dB dip
MIN_PEAK_DISTANCE_S = 0.05     # don't merge syllables closer than 50 ms
SYLLABLE_HALF_WINDOW_S = 0.075 # syllable span = nucleus ± 75 ms (rough; refined later with phone align)

# Output dimensionality
N_DIMS = 18


# ---------------------------------------------------------------------------
# Speaker baseline data
# ---------------------------------------------------------------------------

@dataclass
class SpeakerBaseline:
    median_f0_hz: float
    syl_duration_mean_s: float
    syl_duration_std_s: float
    energy_mean_db: float
    energy_std_db: float
    n_utterances_seen: int


def hz_to_semitones(hz: float, ref_hz: float) -> float:
    """Convert F0 in Hz to semitones relative to a speaker reference. NaN on
    unvoiced or invalid input."""
    if not (hz and hz > 0 and ref_hz and ref_hz > 0):
        return float("nan")
    return 12.0 * math.log2(hz / ref_hz)


# ---------------------------------------------------------------------------
# Syllable-nucleus detection
# ---------------------------------------------------------------------------

@dataclass
class SyllableSpan:
    nucleus_t_s: float
    t_start_s: float
    t_end_s: float


def detect_syllable_nuclei(sound: parselmouth.Sound, pitch: parselmouth.Pitch) -> list[SyllableSpan]:
    """Simplified de Jong & Wempe (2009) syllable-nucleus detection.

    Steps:
        1. Compute intensity contour.
        2. Threshold = intensity_max − SILENCE_DB_BELOW_PEAK.
        3. Find local maxima above threshold.
        4. For each candidate peak, require it to coincide with voiced F0
           (otherwise it's noise/aspiration, not a syllable nucleus).
        5. Enforce minimum distance and minimum dip between consecutive peaks.
        6. Build syllable spans as nucleus ± SYLLABLE_HALF_WINDOW_S, clamped
           to midpoints with neighbors.

    This gives reasonable nucleus times for clean speech. It is approximate;
    Phase 2 will replace it with NXT/MFA phone-aligned syllabification.
    """
    intensity = sound.to_intensity(
        minimum_pitch=INTENSITY_MIN_PITCH_HZ,
        time_step=INTENSITY_TIME_STEP_S,
    )
    times = intensity.xs()
    db = intensity.values[0]
    if len(db) == 0:
        return []
    db_max = float(np.nanmax(db))
    threshold = db_max - SILENCE_DB_BELOW_PEAK

    # Local maxima: db[i] > db[i±1] and above threshold
    is_peak = np.zeros_like(db, dtype=bool)
    is_peak[1:-1] = (db[1:-1] > db[:-2]) & (db[1:-1] > db[2:]) & (db[1:-1] >= threshold)
    peak_idx = np.flatnonzero(is_peak).tolist()

    # F0 voicing check at each peak time
    f0_values = pitch.selected_array["frequency"]
    f0_times = pitch.xs()

    def is_voiced_at(t_s: float) -> bool:
        if len(f0_times) == 0:
            return False
        i = int(np.argmin(np.abs(f0_times - t_s)))
        return f0_values[i] > 0

    voiced_peaks = [i for i in peak_idx if is_voiced_at(times[i])]

    # Minimum distance + minimum dip
    kept: list[int] = []
    for i in voiced_peaks:
        if not kept:
            kept.append(i)
            continue
        prev = kept[-1]
        if times[i] - times[prev] < MIN_PEAK_DISTANCE_S:
            # too close — keep the louder one
            if db[i] > db[prev]:
                kept[-1] = i
            continue
        # require a dip of ≥ MIN_DIP_DB_BETWEEN somewhere between prev and i
        between = db[prev:i + 1]
        dip = max(db[prev], db[i]) - float(np.min(between))
        if dip < MIN_DIP_DB_BETWEEN:
            if db[i] > db[prev]:
                kept[-1] = i
            continue
        kept.append(i)

    # Build syllable spans
    spans: list[SyllableSpan] = []
    for j, i in enumerate(kept):
        nucleus_t = float(times[i])
        # default ±SYLLABLE_HALF_WINDOW_S
        t_start = nucleus_t - SYLLABLE_HALF_WINDOW_S
        t_end = nucleus_t + SYLLABLE_HALF_WINDOW_S
        # clamp to midpoints with neighbors
        if j > 0:
            prev_nucleus = float(times[kept[j - 1]])
            t_start = max(t_start, (prev_nucleus + nucleus_t) / 2.0)
        if j < len(kept) - 1:
            next_nucleus = float(times[kept[j + 1]])
            t_end = min(t_end, (nucleus_t + next_nucleus) / 2.0)
        # clamp to audio bounds
        t_start = max(t_start, 0.0)
        t_end = min(t_end, sound.get_total_duration())
        if t_end > t_start:
            spans.append(SyllableSpan(nucleus_t, t_start, t_end))
    return spans


# ---------------------------------------------------------------------------
# Acoustic primitives — F0, intensity, samplers
# ---------------------------------------------------------------------------

def extract_pitch(sound: parselmouth.Sound) -> parselmouth.Pitch:
    return call(
        sound,
        "To Pitch (cc)",
        PITCH_TIME_STEP_S,
        PITCH_FLOOR_HZ,
        15,           # max number of candidates
        "no",         # very accurate
        0.03,         # silence threshold
        0.45,         # voicing threshold
        0.01,         # octave cost
        0.35,         # octave-jump cost
        0.14,         # voiced/unvoiced cost
        PITCH_CEILING_HZ,
    )


def f0_at_time(pitch: parselmouth.Pitch, t_s: float) -> float:
    """F0 in Hz at given time. Returns 0 (unvoiced) if F0 not detected."""
    f0 = call(pitch, "Get value at time", t_s, "Hertz", "Linear")
    if f0 is None or math.isnan(f0):
        return 0.0
    return float(f0)


# ---------------------------------------------------------------------------
# Pass 1 — speaker baselines
# ---------------------------------------------------------------------------

def compute_speaker_baseline(wav_paths: list[Path]) -> SpeakerBaseline | None:
    f0_hz_pool: list[float] = []
    syl_durations_s: list[float] = []
    energy_db_pool: list[float] = []
    n_seen = 0
    # Praat's intensity analysis needs at least 6.4 / min_pitch_hz seconds.
    # At INTENSITY_MIN_PITCH_HZ = 50 Hz that's 128 ms. Add a safety margin.
    min_duration_s = 6.4 / INTENSITY_MIN_PITCH_HZ + 0.02
    for wp in wav_paths:
        try:
            sound = parselmouth.Sound(str(wp))
            if sound.get_total_duration() < min_duration_s:
                continue
            pitch = extract_pitch(sound)
            intensity = sound.to_intensity(
                minimum_pitch=INTENSITY_MIN_PITCH_HZ,
                time_step=INTENSITY_TIME_STEP_S,
            )
            spans = detect_syllable_nuclei(sound, pitch)
        except Exception:
            # any per-file failure (corrupt audio, Praat refusing the analysis,
            # etc.) — skip and move on so one bad file doesn't kill pass 1.
            continue
        f0_vals = pitch.selected_array["frequency"]
        f0_hz_pool.extend(float(v) for v in f0_vals if v > 0)
        db = intensity.values[0]
        energy_db_pool.extend(float(v) for v in db if not math.isnan(v))
        syl_durations_s.extend(s.t_end_s - s.t_start_s for s in spans)
        n_seen += 1

    if not f0_hz_pool or not syl_durations_s or not energy_db_pool:
        return None
    return SpeakerBaseline(
        median_f0_hz=float(np.median(f0_hz_pool)),
        syl_duration_mean_s=float(np.mean(syl_durations_s)),
        syl_duration_std_s=float(np.std(syl_durations_s) or 1e-6),
        energy_mean_db=float(np.mean(energy_db_pool)),
        energy_std_db=float(np.std(energy_db_pool) or 1e-6),
        n_utterances_seen=n_seen,
    )


# ---------------------------------------------------------------------------
# Pass 2 — per-syllable parametric vector
# ---------------------------------------------------------------------------

def parametric_vector(
    sound: parselmouth.Sound,
    pitch: parselmouth.Pitch,
    intensity: parselmouth.Intensity,
    span: SyllableSpan,
    next_span: SyllableSpan | None,
    baseline: SpeakerBaseline,
) -> tuple[np.ndarray, float]:
    """Compute 18-dim vector + voiced_fraction for one syllable."""
    duration_s = span.t_end_s - span.t_start_s
    duration_ms = duration_s * 1000.0

    # ---- F0 frames within syllable ----
    f0_times = pitch.xs()
    f0_hz = pitch.selected_array["frequency"]
    in_syl = (f0_times >= span.t_start_s) & (f0_times <= span.t_end_s)
    f0_in_hz = f0_hz[in_syl]
    f0_in_times = f0_times[in_syl]
    voiced_mask = f0_in_hz > 0
    voiced_fraction = float(voiced_mask.mean()) if len(f0_in_hz) > 0 else 0.0

    nan = float("nan")
    if voiced_fraction == 0.0:
        f0_onset_st = f0_nucleus_st = f0_offset_st = nan
        f0_max_st = f0_min_st = f0_range_st = f0_slope_st_per_ms = nan
        f0_peak_position_norm = f0_rise_amplitude_st = f0_fall_amplitude_st = tilt = nan
    else:
        # Convert each voiced frame to semitones
        f0_in_st = np.array([
            hz_to_semitones(hz, baseline.median_f0_hz) if v else nan
            for hz, v in zip(f0_in_hz, voiced_mask)
        ])
        # Onset / nucleus / offset: use closest voiced sample to those reference times
        f0_onset_st = _closest_voiced(f0_in_times, f0_in_st, span.t_start_s)
        f0_nucleus_st = _closest_voiced(f0_in_times, f0_in_st, span.nucleus_t_s)
        f0_offset_st = _closest_voiced(f0_in_times, f0_in_st, span.t_end_s)

        voiced_st = f0_in_st[~np.isnan(f0_in_st)]
        if len(voiced_st):
            f0_max_st = float(np.max(voiced_st))
            f0_min_st = float(np.min(voiced_st))
            f0_range_st = f0_max_st - f0_min_st
            peak_idx = int(np.argmax(np.where(np.isnan(f0_in_st), -np.inf, f0_in_st)))
            peak_t_in_syl = float(f0_in_times[peak_idx]) - span.t_start_s
            f0_peak_position_norm = peak_t_in_syl / duration_s if duration_s > 0 else nan
            f0_rise_amplitude_st = (
                f0_max_st - f0_onset_st if not math.isnan(f0_onset_st) else nan
            )
            f0_fall_amplitude_st = (
                f0_max_st - f0_offset_st if not math.isnan(f0_offset_st) else nan
            )
            denom = (f0_rise_amplitude_st or 0) + (f0_fall_amplitude_st or 0)
            tilt = (
                (f0_rise_amplitude_st - f0_fall_amplitude_st) / denom
                if denom and not math.isnan(denom) else nan
            )
        else:
            f0_max_st = f0_min_st = f0_range_st = nan
            f0_peak_position_norm = f0_rise_amplitude_st = f0_fall_amplitude_st = tilt = nan
        # Slope from offset and onset (semitones per ms)
        if not math.isnan(f0_onset_st) and not math.isnan(f0_offset_st) and duration_ms > 0:
            f0_slope_st_per_ms = (f0_offset_st - f0_onset_st) / duration_ms
        else:
            f0_slope_st_per_ms = nan

    # ---- Energy ----
    int_times = intensity.xs()
    int_db = intensity.values[0]
    in_syl_int = (int_times >= span.t_start_s) & (int_times <= span.t_end_s)
    rms_in_db = int_db[in_syl_int]
    if len(rms_in_db) == 0:
        rms_max_z = rms_mean_z = nan
    else:
        rms_max_z = (float(np.max(rms_in_db)) - baseline.energy_mean_db) / baseline.energy_std_db
        rms_mean_z = (float(np.mean(rms_in_db)) - baseline.energy_mean_db) / baseline.energy_std_db

    # ---- Duration ----
    syllable_duration_z = (duration_s - baseline.syl_duration_mean_s) / baseline.syl_duration_std_s
    # nucleus_duration: vowel-only duration. Without phone alignment we approximate
    # as 50% of syllable duration. Phase 2 (NXT phone-aligned) replaces this.
    nucleus_duration_z = ((duration_s * 0.5) - (baseline.syl_duration_mean_s * 0.5)) / (
        baseline.syl_duration_std_s * 0.5 + 1e-9
    )

    # ---- Boundary (right edge) ----
    if next_span is not None:
        pause_after_ms = max(0.0, (next_span.t_start_s - span.t_end_s) * 1000.0)
        next_onset_st = _closest_voiced(
            pitch.xs(),
            np.array([
                hz_to_semitones(hz, baseline.median_f0_hz) if hz > 0 else nan
                for hz in pitch.selected_array["frequency"]
            ]),
            next_span.t_start_s,
        )
        if not math.isnan(next_onset_st) and not math.isnan(f0_offset_st):
            f0_reset_st = next_onset_st - f0_offset_st
        else:
            f0_reset_st = nan
    else:
        pause_after_ms = 0.0
        f0_reset_st = 0.0
    # final_lengthening_ratio: needs word-level grouping. Phase 2 fills this in
    # from forced alignment; for v1 we set the placeholder to 1.0 so the dim is
    # always present and the downstream consumer can mask it.
    final_lengthening_ratio = 1.0

    vec = np.array([
        f0_onset_st,
        f0_nucleus_st,
        f0_offset_st,
        f0_max_st,
        f0_min_st,
        f0_range_st,
        f0_slope_st_per_ms,
        f0_peak_position_norm,
        f0_rise_amplitude_st,
        f0_fall_amplitude_st,
        tilt,
        rms_max_z,
        rms_mean_z,
        syllable_duration_z,
        nucleus_duration_z,
        pause_after_ms,
        final_lengthening_ratio,
        f0_reset_st,
    ], dtype=np.float64)
    assert vec.shape == (N_DIMS,)
    return vec, voiced_fraction


def _closest_voiced(times: np.ndarray, values_st: np.ndarray, t_ref: float) -> float:
    """Closest non-NaN value to t_ref. Returns NaN if no voiced sample found."""
    if len(times) == 0:
        return float("nan")
    voiced_mask = ~np.isnan(values_st)
    if not voiced_mask.any():
        return float("nan")
    times_v = times[voiced_mask]
    vals_v = values_st[voiced_mask]
    i = int(np.argmin(np.abs(times_v - t_ref)))
    return float(vals_v[i])


# ---------------------------------------------------------------------------
# MELD wiring
# ---------------------------------------------------------------------------

def meld_audio_path(in_dir: Path, dialogue_id: int, utterance_id: int) -> Path | None:
    """Return the audio path for a MELD utterance, preferring wav over mp4.
    Returns None if neither exists."""
    base = f"dia{dialogue_id}_utt{utterance_id}"
    for ext in (".wav", ".mp4"):
        p = in_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def ensure_wav(audio_path: Path, cache_dir: Path) -> Path | None:
    """Return a wav path Parselmouth can read. If input is already wav, return
    as-is. If input is mp4, lazily convert via ffmpeg into cache_dir and return
    the wav. Returns None on conversion failure."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    if audio_path.suffix.lower() != ".mp4":
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_out = cache_dir / (audio_path.stem + ".wav")
    if wav_out.exists() and wav_out.stat().st_size > 1024:
        return wav_out
    if shutil.which("ffmpeg") is None:
        print("    ffmpeg not on PATH; cannot convert mp4 to wav", file=sys.stderr)
        return None
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(audio_path),
                "-ac", "1",         # mono — speakers in MELD are tagged per-utterance, no need for stereo
                "-ar", "16000",     # 16 kHz is plenty for prosody work
                "-vn",              # drop video stream
                str(wav_out),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"    ffmpeg failed on {audio_path.name}: {e}", file=sys.stderr)
        return None
    return wav_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="Directory of MELD utterance audio (mp4 or wav).")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="MELD-style CSV with columns Dialogue_ID, Utterance_ID, Speaker.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL.")
    parser.add_argument("--wav-cache", type=Path, default=None,
                        help="Directory for lazy mp4→wav cache. Default: <in-dir>/_wav_cache.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many utterances (smoke-test).")
    parser.add_argument("--min-utts-per-speaker", type=int, default=3,
                        help="Skip speakers with fewer than this many utterances (baseline unstable).")
    args = parser.parse_args()
    wav_cache = args.wav_cache or (args.in_dir / "_wav_cache")

    if not args.metadata.exists():
        sys.exit(f"metadata not found: {args.metadata}")
    if not args.in_dir.exists():
        sys.exit(f"audio dir not found: {args.in_dir}")

    df = pd.read_csv(args.metadata)
    required_cols = {"Dialogue_ID", "Utterance_ID", "Speaker"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"metadata missing required columns: {missing}")

    # Pass 1 — speaker baselines
    print(f"[pass 1] computing speaker baselines from {len(df)} utterances...")
    baselines: dict[str, SpeakerBaseline] = {}
    for spk, group in df.groupby("Speaker"):
        audio_paths_raw = [meld_audio_path(args.in_dir, r.Dialogue_ID, r.Utterance_ID) for r in group.itertuples()]
        audio_paths = [p for p in audio_paths_raw if p is not None]
        wav_paths = [ensure_wav(p, wav_cache) for p in audio_paths]
        wav_paths = [p for p in wav_paths if p is not None]
        if len(wav_paths) < args.min_utts_per_speaker:
            print(f"    skip speaker {spk!r}: {len(wav_paths)} utterances")
            continue
        bl = compute_speaker_baseline(wav_paths)
        if bl is not None:
            baselines[spk] = bl
            print(f"    {spk!r}: median F0 {bl.median_f0_hz:.1f} Hz, n={bl.n_utterances_seen}")
    if not baselines:
        sys.exit("no usable speaker baselines computed; aborting")

    # Pass 2 — extract per-syllable vectors
    print(f"[pass 2] extracting parametric vectors for {len(df)} utterances...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with args.out.open("w") as f_out:
        for i, row in enumerate(df.itertuples()):
            if args.limit and n_written >= args.limit:
                break
            audio_path = meld_audio_path(args.in_dir, row.Dialogue_ID, row.Utterance_ID)
            if audio_path is None:
                n_skipped += 1
                continue
            wp = ensure_wav(audio_path, wav_cache)
            if wp is None:
                n_skipped += 1
                continue
            baseline = baselines.get(row.Speaker)
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
                "utterance_id": f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}",
                "speaker_id": row.Speaker,
                "audio_path": str(audio_path),
                "n_syllables": len(syllable_records),
                "syllables": syllable_records,
            }) + "\n")
            n_written += 1
            if n_written % 50 == 0:
                print(f"    written {n_written}/{len(df)} (skipped {n_skipped})")

    print(f"Done. {n_written} utterances written, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
