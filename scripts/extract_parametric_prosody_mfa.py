"""
v2 of the D19 18-dim parametric prosody extractor — driven by MFA TextGrids
instead of intensity-peak heuristics.

What changes from `extract_parametric_prosody.py` (v1):
    - Syllable boundaries come from MFA's phone alignment + max-onset
      syllabification (not Praat's de Jong intensity-peak detector).
    - `nucleus_duration_z` uses the *real* vowel duration from the phone
      tier (not 0.5 × syllable as in v1).
    - `final_lengthening_ratio` is the syllable's duration divided by the
      mean of its containing word's syllable durations (not the v1 1.0
      placeholder).
    - Output records carry the phone sequence + word label per syllable
      for downstream interpretability / spot-checking.

Inputs:
    - per-utterance wavs (16 kHz mono recommended; ours match)
    - per-utterance MFA TextGrids with `words` and `phones` interval tiers
    - MELD CSV with Dialogue_ID, Utterance_ID, Speaker

Output: one JSONL line per utterance, same schema as v1 with two added
fields per syllable: `phones` (list of ARPABET strings) and `word` (the
containing word's label, possibly `<unk>` for OOV).

Usage:
    .venv/bin/python scripts/extract_parametric_prosody_mfa.py \\
        --in-dir data/meld/MELD.Raw/dev_splits_complete \\
        --textgrids-dir data/meld/dev_textgrids_mfa \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --out data/meld/parametric_prosody_dev_mfa.jsonl

For full design rationale see docs/design_decisions.md (D4, D6, D19) and
docs/writeups/parametric_prosody_pivot.md.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

from scripts.voice_quality_features import compute_voice_quality

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 600.0
PITCH_TIME_STEP_S = 0.005
INTENSITY_MIN_PITCH_HZ = 50.0
INTENSITY_TIME_STEP_S = 0.005
N_DIMS = 18

# ARPABET vowel symbols (CMUdict). Stress digit (0/1/2) is suffixed and
# stripped before lookup.
ARPABET_VOWELS = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
})

# Phone labels we treat as non-speech and skip during syllabification.
NON_SPEECH_LABELS = frozenset({"", "spn", "sil", "sp", "oov", "<unk>"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Phone:
    label: str
    t_start: float
    t_end: float


@dataclass
class Word:
    label: str
    t_start: float
    t_end: float
    phones: list[Phone] = field(default_factory=list)


@dataclass
class Syllable:
    phones: list[Phone]
    nucleus: Phone
    word: Word
    t_start: float
    t_end: float

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def nucleus_duration_s(self) -> float:
        return self.nucleus.t_end - self.nucleus.t_start

    @property
    def nucleus_t_s(self) -> float:
        return (self.nucleus.t_start + self.nucleus.t_end) / 2.0


@dataclass
class SpeakerBaseline:
    median_f0_hz: float
    syl_duration_mean_s: float
    syl_duration_std_s: float
    nuc_duration_mean_s: float
    nuc_duration_std_s: float
    energy_mean_db: float
    energy_std_db: float
    n_utterances_seen: int


# ---------------------------------------------------------------------------
# ARPABET helpers
# ---------------------------------------------------------------------------

def phone_base(p: str) -> str:
    """Strip stress digit suffix from an ARPABET phone label."""
    return p.rstrip("012")


def is_vowel(p: str) -> bool:
    return phone_base(p) in ARPABET_VOWELS


def is_speech_phone(label: str) -> bool:
    return bool(label) and label not in NON_SPEECH_LABELS


# ---------------------------------------------------------------------------
# TextGrid loading
# ---------------------------------------------------------------------------

def load_textgrid(path: Path) -> list[Word]:
    """Read an MFA TextGrid and return Words with their Phones attached.

    Phones are assigned to a Word if the phone interval falls inside the
    word interval (with a small ±1 ms slack to absorb floating-point noise).
    """
    tg = parselmouth.Data.read(str(path))
    n_tiers = int(call(tg, "Get number of tiers"))

    word_intervals: list[tuple[float, float, str]] = []
    phone_intervals: list[tuple[float, float, str]] = []
    for i in range(1, n_tiers + 1):
        name = call(tg, "Get tier name", i)
        n_int = int(call(tg, "Get number of intervals", i))
        intervals = []
        for j in range(1, n_int + 1):
            t_start = float(call(tg, "Get start time of interval", i, j))
            t_end = float(call(tg, "Get end time of interval", i, j))
            label = call(tg, "Get label of interval", i, j) or ""
            intervals.append((t_start, t_end, label))
        if name == "words":
            word_intervals = intervals
        elif name == "phones":
            phone_intervals = intervals

    words: list[Word] = []
    for w_start, w_end, w_label in word_intervals:
        word = Word(label=w_label, t_start=w_start, t_end=w_end)
        # Attach phones whose interval is fully within this word's interval
        for p_start, p_end, p_label in phone_intervals:
            if p_start >= w_start - 0.001 and p_end <= w_end + 0.001:
                word.phones.append(Phone(label=p_label, t_start=p_start, t_end=p_end))
        words.append(word)
    return words


# ---------------------------------------------------------------------------
# Syllabification (max-onset principle, simple split rule)
# ---------------------------------------------------------------------------

def syllabify_word(word: Word) -> list[Syllable]:
    """Group a word's phones into syllables using max-onset splitting.

    Rule for intervocalic clusters:
        - 1 consonant: all to next syllable's onset (max onset)
        - 2 consonants: 1+1 split (first to coda, second to onset)
        - 3 consonants: 1+2 split (English allows 2-consonant onsets in many cases)
        - n consonants: floor(n/2) to coda of left syllable, rest to onset of right.

    This is approximate — a phonotactic-aware splitter (sonority sequencing)
    would do "factor"=fac-tor correctly while avoiding "extra"=ex-tra. The
    floor-half rule gets common English cases right and is simple.
    """
    speech_phones = [p for p in word.phones if is_speech_phone(p.label)]
    if not speech_phones:
        return []

    vowel_idxs = [i for i, p in enumerate(speech_phones) if is_vowel(p.label)]
    if not vowel_idxs:
        return []

    syllables: list[Syllable] = []
    for k, v_idx in enumerate(vowel_idxs):
        # Onset start
        if k == 0:
            onset_start = 0
        else:
            prev_v = vowel_idxs[k - 1]
            cluster_len = v_idx - prev_v - 1
            coda_count = cluster_len // 2
            onset_start = prev_v + 1 + coda_count

        # Coda end
        if k == len(vowel_idxs) - 1:
            coda_end = len(speech_phones)
        else:
            next_v = vowel_idxs[k + 1]
            cluster_len = next_v - v_idx - 1
            coda_count = cluster_len // 2
            coda_end = v_idx + 1 + coda_count

        syl_phones = speech_phones[onset_start:coda_end]
        syllables.append(Syllable(
            phones=syl_phones,
            nucleus=speech_phones[v_idx],
            word=word,
            t_start=syl_phones[0].t_start,
            t_end=syl_phones[-1].t_end,
        ))
    return syllables


def syllabify_utterance(words: list[Word]) -> list[Syllable]:
    """All syllables across all words, in temporal order."""
    out: list[Syllable] = []
    for w in words:
        if not is_speech_phone(w.label):
            continue
        out.extend(syllabify_word(w))
    out.sort(key=lambda s: s.t_start)
    return out


# ---------------------------------------------------------------------------
# Acoustic primitives
# ---------------------------------------------------------------------------

def hz_to_semitones(hz: float, ref_hz: float) -> float:
    if not (hz and hz > 0 and ref_hz and ref_hz > 0):
        return float("nan")
    return 12.0 * math.log2(hz / ref_hz)


def extract_pitch(sound: parselmouth.Sound) -> parselmouth.Pitch:
    return call(
        sound, "To Pitch (cc)",
        PITCH_TIME_STEP_S, PITCH_FLOOR_HZ, 15, "no",
        0.03, 0.45, 0.01, 0.35, 0.14, PITCH_CEILING_HZ,
    )


def _closest_voiced(times: np.ndarray, values_st: np.ndarray, t_ref: float) -> float:
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
# Pass 1 — speaker baseline
# ---------------------------------------------------------------------------

def compute_speaker_baseline(wav_tg_pairs: list[tuple[Path, Path]]) -> SpeakerBaseline | None:
    f0_hz_pool: list[float] = []
    syl_durations_s: list[float] = []
    nuc_durations_s: list[float] = []
    energy_db_pool: list[float] = []
    n_seen = 0

    min_dur = 6.4 / INTENSITY_MIN_PITCH_HZ + 0.02

    for wav, tg in wav_tg_pairs:
        try:
            sound = parselmouth.Sound(str(wav))
            if sound.get_total_duration() < min_dur:
                continue
            pitch = extract_pitch(sound)
            intensity = sound.to_intensity(
                minimum_pitch=INTENSITY_MIN_PITCH_HZ,
                time_step=INTENSITY_TIME_STEP_S,
            )
            words = load_textgrid(tg)
            syllables = syllabify_utterance(words)
        except Exception:
            continue
        if not syllables:
            continue

        f0_vals = pitch.selected_array["frequency"]
        f0_hz_pool.extend(float(v) for v in f0_vals if v > 0)
        db = intensity.values[0]
        energy_db_pool.extend(float(v) for v in db if not math.isnan(v))
        for syl in syllables:
            syl_durations_s.append(syl.duration_s)
            nuc_durations_s.append(syl.nucleus_duration_s)
        n_seen += 1

    if not (f0_hz_pool and syl_durations_s and nuc_durations_s and energy_db_pool):
        return None
    return SpeakerBaseline(
        median_f0_hz=float(np.median(f0_hz_pool)),
        syl_duration_mean_s=float(np.mean(syl_durations_s)),
        syl_duration_std_s=float(np.std(syl_durations_s) or 1e-6),
        nuc_duration_mean_s=float(np.mean(nuc_durations_s)),
        nuc_duration_std_s=float(np.std(nuc_durations_s) or 1e-6),
        energy_mean_db=float(np.mean(energy_db_pool)),
        energy_std_db=float(np.std(energy_db_pool) or 1e-6),
        n_utterances_seen=n_seen,
    )


# ---------------------------------------------------------------------------
# Pass 2 — per-syllable parametric vector
# ---------------------------------------------------------------------------

def parametric_vector(
    pitch: parselmouth.Pitch,
    intensity: parselmouth.Intensity,
    syl: Syllable,
    next_syl: Syllable | None,
    word_syl_durs: dict[int, list[float]],
    baseline: SpeakerBaseline,
) -> tuple[np.ndarray, float]:
    duration_s = syl.duration_s
    duration_ms = duration_s * 1000.0
    nan = float("nan")

    # ---- F0 within syllable ----
    f0_times = pitch.xs()
    f0_hz = pitch.selected_array["frequency"]
    in_syl = (f0_times >= syl.t_start) & (f0_times <= syl.t_end)
    f0_in_hz = f0_hz[in_syl]
    f0_in_times = f0_times[in_syl]
    voiced_mask = f0_in_hz > 0
    voiced_fraction = float(voiced_mask.mean()) if len(f0_in_hz) > 0 else 0.0

    if voiced_fraction == 0.0:
        f0_onset_st = f0_nucleus_st = f0_offset_st = nan
        f0_max_st = f0_min_st = f0_range_st = f0_slope_st_per_ms = nan
        f0_peak_position_norm = f0_rise_amplitude_st = f0_fall_amplitude_st = tilt = nan
    else:
        f0_in_st = np.array([
            hz_to_semitones(hz, baseline.median_f0_hz) if v else nan
            for hz, v in zip(f0_in_hz, voiced_mask)
        ])
        f0_onset_st = _closest_voiced(f0_in_times, f0_in_st, syl.t_start)
        f0_nucleus_st = _closest_voiced(f0_in_times, f0_in_st, syl.nucleus_t_s)
        f0_offset_st = _closest_voiced(f0_in_times, f0_in_st, syl.t_end)

        voiced_st = f0_in_st[~np.isnan(f0_in_st)]
        if len(voiced_st):
            f0_max_st = float(np.max(voiced_st))
            f0_min_st = float(np.min(voiced_st))
            f0_range_st = f0_max_st - f0_min_st
            peak_idx = int(np.argmax(np.where(np.isnan(f0_in_st), -np.inf, f0_in_st)))
            peak_t_in_syl = float(f0_in_times[peak_idx]) - syl.t_start
            f0_peak_position_norm = peak_t_in_syl / duration_s if duration_s > 0 else nan
            f0_rise_amplitude_st = (
                f0_max_st - f0_onset_st if not math.isnan(f0_onset_st) else nan
            )
            f0_fall_amplitude_st = (
                f0_max_st - f0_offset_st if not math.isnan(f0_offset_st) else nan
            )
            denom = ((f0_rise_amplitude_st if not math.isnan(f0_rise_amplitude_st) else 0)
                   + (f0_fall_amplitude_st if not math.isnan(f0_fall_amplitude_st) else 0))
            if denom and not math.isnan(denom):
                tilt = (f0_rise_amplitude_st - f0_fall_amplitude_st) / denom
            else:
                tilt = nan
        else:
            f0_max_st = f0_min_st = f0_range_st = nan
            f0_peak_position_norm = f0_rise_amplitude_st = f0_fall_amplitude_st = tilt = nan

        if not math.isnan(f0_onset_st) and not math.isnan(f0_offset_st) and duration_ms > 0:
            f0_slope_st_per_ms = (f0_offset_st - f0_onset_st) / duration_ms
        else:
            f0_slope_st_per_ms = nan

    # ---- Energy ----
    int_times = intensity.xs()
    int_db = intensity.values[0]
    in_syl_int = (int_times >= syl.t_start) & (int_times <= syl.t_end)
    rms_in_db = int_db[in_syl_int]
    if len(rms_in_db) == 0:
        rms_max_z = rms_mean_z = nan
    else:
        rms_max_z = (float(np.max(rms_in_db)) - baseline.energy_mean_db) / baseline.energy_std_db
        rms_mean_z = (float(np.mean(rms_in_db)) - baseline.energy_mean_db) / baseline.energy_std_db

    # ---- Duration (now using REAL phone-aligned values) ----
    syllable_duration_z = (duration_s - baseline.syl_duration_mean_s) / baseline.syl_duration_std_s
    nucleus_duration_z = (
        (syl.nucleus_duration_s - baseline.nuc_duration_mean_s) / baseline.nuc_duration_std_s
    )

    # ---- Boundary ----
    if next_syl is not None:
        pause_after_ms = max(0.0, (next_syl.t_start - syl.t_end) * 1000.0)
        next_onset_st = _closest_voiced(
            pitch.xs(),
            np.array([
                hz_to_semitones(hz, baseline.median_f0_hz) if hz > 0 else nan
                for hz in pitch.selected_array["frequency"]
            ]),
            next_syl.t_start,
        )
        if not math.isnan(next_onset_st) and not math.isnan(f0_offset_st):
            f0_reset_st = next_onset_st - f0_offset_st
        else:
            f0_reset_st = nan
    else:
        pause_after_ms = 0.0
        f0_reset_st = 0.0

    # final_lengthening_ratio = this syllable's duration / mean of word's syllable durations
    word_durs = word_syl_durs.get(id(syl.word), [duration_s])
    word_mean = float(np.mean(word_durs)) if word_durs else duration_s
    final_lengthening_ratio = duration_s / word_mean if word_mean > 0 else 1.0

    vec = np.array([
        f0_onset_st, f0_nucleus_st, f0_offset_st,
        f0_max_st, f0_min_st, f0_range_st, f0_slope_st_per_ms,
        f0_peak_position_norm, f0_rise_amplitude_st, f0_fall_amplitude_st, tilt,
        rms_max_z, rms_mean_z,
        syllable_duration_z, nucleus_duration_z,
        pause_after_ms, final_lengthening_ratio, f0_reset_st,
    ], dtype=np.float64)
    assert vec.shape == (N_DIMS,)
    return vec, voiced_fraction


# ---------------------------------------------------------------------------
# MELD wiring
# ---------------------------------------------------------------------------

def meld_audio_path(in_dir: Path, dialogue_id: int, utterance_id: int) -> Path | None:
    base = f"dia{dialogue_id}_utt{utterance_id}"
    for ext in (".wav", ".mp4"):
        p = in_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def textgrid_path(tg_dir: Path, dialogue_id: int, utterance_id: int) -> Path:
    return tg_dir / f"dia{dialogue_id}_utt{utterance_id}.TextGrid"


def _process_one_meld(task: dict) -> dict | None:
    """Worker function for parallel Pass 2. Takes a task dict with
    wav_path, tg_path, utterance_id, speaker_id, baseline. Returns the
    JSONL row dict or None on failure.
    """
    wp = Path(task["wav_path"])
    tp = Path(task["tg_path"])
    baseline = task["baseline"]
    try:
        sound = parselmouth.Sound(str(wp))
        pitch = extract_pitch(sound)
        intensity = sound.to_intensity(
            minimum_pitch=INTENSITY_MIN_PITCH_HZ,
            time_step=INTENSITY_TIME_STEP_S,
        )
        words = load_textgrid(tp)
        syllables = syllabify_utterance(words)
    except Exception:
        return None
    if not syllables:
        return None

    word_syl_durs: dict[int, list[float]] = {}
    for syl in syllables:
        word_syl_durs.setdefault(id(syl.word), []).append(syl.duration_s)

    syl_records = []
    for j, syl in enumerate(syllables):
        next_syl = syllables[j + 1] if j + 1 < len(syllables) else None
        vec, vf = parametric_vector(
            pitch, intensity, syl, next_syl, word_syl_durs, baseline,
        )
        vq = compute_voice_quality(
            sound, pitch, intensity, syl.t_start, syl.t_end,
            baseline.median_f0_hz,
        )
        syl_records.append({
            "nucleus_t_ms": round(syl.nucleus_t_s * 1000, 2),
            "t_start_ms":   round(syl.t_start * 1000, 2),
            "t_end_ms":     round(syl.t_end * 1000, 2),
            "phones":       [p.label for p in syl.phones],
            "nucleus_phone": syl.nucleus.label,
            "word":         syl.word.label,
            "vec":          [None if math.isnan(x) else round(x, 4) for x in vec.tolist()],
            "vq":           [None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(x, 4) for x in vq],
            "voiced_fraction": round(vf, 3),
        })
    return {
        "utterance_id": task["utterance_id"],
        "speaker_id":   task["speaker_id"],
        "audio_path":   str(wp),
        "textgrid_path": str(tp),
        "n_syllables":  len(syl_records),
        "syllables":    syl_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="Per-utterance wav directory.")
    parser.add_argument("--textgrids-dir", type=Path, required=True,
                        help="MFA-output TextGrid directory.")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="MELD CSV with Dialogue_ID, Utterance_ID, Speaker.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-utts-per-speaker", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for Pass 2 parallelism.")
    args = parser.parse_args()

    if not args.metadata.exists():
        sys.exit(f"metadata not found: {args.metadata}")
    if not args.in_dir.exists():
        sys.exit(f"audio dir not found: {args.in_dir}")
    if not args.textgrids_dir.exists():
        sys.exit(f"textgrids dir not found: {args.textgrids_dir}")

    df = pd.read_csv(args.metadata)
    required = {"Dialogue_ID", "Utterance_ID", "Speaker"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"metadata missing required columns: {missing}")

    # Pass 1
    print(f"[pass 1] computing speaker baselines from {len(df)} utterances...")
    baselines: dict[str, SpeakerBaseline] = {}
    n_skipped_no_tg = 0
    for spk, group in df.groupby("Speaker"):
        wav_tg_pairs: list[tuple[Path, Path]] = []
        for r in group.itertuples():
            wp = meld_audio_path(args.in_dir, r.Dialogue_ID, r.Utterance_ID)
            tp = textgrid_path(args.textgrids_dir, r.Dialogue_ID, r.Utterance_ID)
            if wp and tp.exists():
                wav_tg_pairs.append((wp, tp))
            else:
                n_skipped_no_tg += 1
        if len(wav_tg_pairs) < args.min_utts_per_speaker:
            print(f"    skip speaker {spk!r}: {len(wav_tg_pairs)} aligned utterances")
            continue
        bl = compute_speaker_baseline(wav_tg_pairs)
        if bl is not None:
            baselines[spk] = bl
            print(f"    {spk!r}: F0 {bl.median_f0_hz:.0f}Hz | "
                  f"syl {bl.syl_duration_mean_s*1000:.0f}±{bl.syl_duration_std_s*1000:.0f}ms | "
                  f"vowel {bl.nuc_duration_mean_s*1000:.0f}±{bl.nuc_duration_std_s*1000:.0f}ms | "
                  f"n={bl.n_utterances_seen}")
    if n_skipped_no_tg:
        print(f"    note: {n_skipped_no_tg} utterances missing wav or TextGrid")
    if not baselines:
        sys.exit("no usable speaker baselines computed; aborting")

    # Pass 2 — build task list, then process (sequentially or in parallel)
    print(f"[pass 2] extracting parametric vectors (workers={args.workers})...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    n_skipped = 0
    for row in df.itertuples():
        if args.limit and len(tasks) >= args.limit:
            break
        wp = meld_audio_path(args.in_dir, row.Dialogue_ID, row.Utterance_ID)
        tp = textgrid_path(args.textgrids_dir, row.Dialogue_ID, row.Utterance_ID)
        if not (wp and tp.exists()):
            n_skipped += 1
            continue
        baseline = baselines.get(row.Speaker)
        if baseline is None:
            n_skipped += 1
            continue
        tasks.append({
            "wav_path": str(wp),
            "tg_path": str(tp),
            "utterance_id": f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}",
            "speaker_id": row.Speaker,
            "baseline": baseline,
        })
    print(f"    {len(tasks)} tasks ({n_skipped} pre-skipped)")

    n_written = 0
    n_failed = 0
    with args.out.open("w") as f_out:
        if args.workers <= 1:
            for task in tasks:
                row_dict = _process_one_meld(task)
                if row_dict is None:
                    n_failed += 1
                    continue
                f_out.write(json.dumps(row_dict) + "\n")
                n_written += 1
                if n_written % 100 == 0:
                    print(f"    written {n_written}/{len(tasks)} ({n_failed} failed)")
        else:
            with mp.get_context("spawn").Pool(processes=args.workers) as pool:
                for row_dict in pool.imap_unordered(_process_one_meld, tasks, chunksize=8):
                    if row_dict is None:
                        n_failed += 1
                        continue
                    f_out.write(json.dumps(row_dict) + "\n")
                    n_written += 1
                    if n_written % 200 == 0:
                        print(f"    written {n_written}/{len(tasks)} ({n_failed} failed)")

    print(f"Done. {n_written} utterances written, {n_skipped + n_failed} skipped/failed.")


if __name__ == "__main__":
    main()
