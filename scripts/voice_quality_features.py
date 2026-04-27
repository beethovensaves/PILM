"""
Voice-quality features for the PILM v3 parametric prosody specification.

Adds four dimensions to the per-syllable representation:
  - creak_fraction          fraction of voiced frames where F0 is anomalously
                            low relative to the speaker's median (heuristic
                            creak detector; not as accurate as Drugman's
                            PEAKDET but cheap and deterministic).
  - h1_minus_h2_db          amplitude difference between first two harmonics
                            at the syllable nucleus (breathiness indicator;
                            positive = breathy, near-zero = modal).
  - cpp_db                  cepstral peak prominence within the syllable
                            (overall periodicity / vocal-fold-vibration
                            regularity; lower = more dysphonic).
  - spectral_tilt_db_per_oct slope of the long-term average spectrum from
                            100 Hz to 5 kHz on a log-frequency axis (steeper
                            negative = breathier; flatter or positive =
                            pressed / creaky).

All four are computed via Parselmouth using standard Praat conventions
(Boersma 2001; Hillenbrand & Houde; Hillenbrand et al.\ for CPP). For
multilingual extension the same set is applicable since the underlying
acoustics are language-independent.

Usage:
    from scripts.voice_quality_features import (
        compute_creak_fraction,
        compute_h1_minus_h2_db,
        compute_cpp_db,
        compute_spectral_tilt_db_per_oct,
    )

    creak = compute_creak_fraction(pitch, intensity, t_start, t_end, baseline.median_f0_hz)
    h1h2  = compute_h1_minus_h2_db(sound, pitch, t_start, t_end)
    cpp   = compute_cpp_db(sound, t_start, t_end)
    tilt  = compute_spectral_tilt_db_per_oct(sound, t_start, t_end)
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import parselmouth
from parselmouth.praat import call

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

CREAK_F0_RATIO_THRESHOLD = 0.6   # frames with F0 < 0.6 × speaker median count as candidates
CREAK_INTENSITY_DB_BELOW = 8.0   # AND intensity at least 8 dB below intensity peak
SPECTRAL_TILT_LOW_HZ = 100.0
SPECTRAL_TILT_HIGH_HZ = 5000.0
CPP_PITCH_FLOOR_HZ = 60.0
CPP_PITCH_CEILING_HZ = 330.0


# ---------------------------------------------------------------------------
# Creak fraction
# ---------------------------------------------------------------------------

def compute_creak_fraction(
    pitch: parselmouth.Pitch,
    intensity: parselmouth.Intensity,
    t_start: float,
    t_end: float,
    speaker_median_f0_hz: float,
) -> float:
    """Heuristic creak detector.

    A frame counts as 'creaky' if it is voiced AND F0 is below
    `CREAK_F0_RATIO_THRESHOLD * speaker_median_f0_hz` AND intensity is at
    least `CREAK_INTENSITY_DB_BELOW` dB below the intensity peak in this
    syllable. Returns the fraction of voiced frames that meet the criteria.
    Returns 0.0 if no voiced frames are available.
    """
    if speaker_median_f0_hz <= 0 or t_end <= t_start:
        return 0.0
    f0_threshold = CREAK_F0_RATIO_THRESHOLD * speaker_median_f0_hz

    f0_times = pitch.xs()
    f0_hz = pitch.selected_array["frequency"]
    in_syl = (f0_times >= t_start) & (f0_times <= t_end)
    f0_hz = f0_hz[in_syl]
    f0_times = f0_times[in_syl]
    voiced = f0_hz > 0
    if not voiced.any():
        return 0.0

    # Intensity peak within syllable
    int_times = intensity.xs()
    int_db = intensity.values[0]
    in_syl_int = (int_times >= t_start) & (int_times <= t_end)
    int_db = int_db[in_syl_int]
    if len(int_db) == 0:
        return 0.0
    intensity_peak = float(np.nanmax(int_db))
    intensity_threshold = intensity_peak - CREAK_INTENSITY_DB_BELOW

    # Per voiced F0 frame, look up nearest intensity sample
    int_times_valid = intensity.xs()[in_syl_int]
    if len(int_times_valid) == 0:
        return 0.0

    n_voiced = 0
    n_creak = 0
    for hz, t in zip(f0_hz, f0_times):
        if hz <= 0:
            continue
        n_voiced += 1
        if hz >= f0_threshold:
            continue
        # find nearest intensity sample
        i_int = int(np.argmin(np.abs(int_times_valid - t)))
        if int_db[i_int] <= intensity_threshold:
            n_creak += 1
    if n_voiced == 0:
        return 0.0
    return float(n_creak) / float(n_voiced)


# ---------------------------------------------------------------------------
# H1 - H2
# ---------------------------------------------------------------------------

def _spectrum_to_mag_db(spec: parselmouth.Spectrum) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_hz, magnitude_db) arrays from a Praat Spectrum."""
    freqs = spec.xs()
    vals = spec.values  # shape (2, n_freq) — real and imaginary
    real = vals[0]
    imag = vals[1]
    magnitude = np.sqrt(real * real + imag * imag)
    # Avoid log(0) — clamp very small magnitudes to a floor
    floor = max(1e-12, float(magnitude.max()) * 1e-9)
    mag_db = 20.0 * np.log10(np.maximum(magnitude, floor))
    return freqs, mag_db


def compute_h1_minus_h2_db(
    sound: parselmouth.Sound,
    pitch: parselmouth.Pitch,
    t_start: float,
    t_end: float,
) -> float:
    """Amplitude difference (dB) between the first two harmonics, evaluated
    at the syllable nucleus midpoint. Positive = breathy, near-zero = modal,
    negative = pressed.

    Returns NaN if F0 cannot be reliably estimated at the nucleus midpoint.
    """
    if t_end <= t_start:
        return float("nan")
    nucleus_t = (t_start + t_end) / 2.0

    f0 = call(pitch, "Get value at time", nucleus_t, "Hertz", "Linear")
    if f0 is None or math.isnan(f0) or f0 <= 0:
        return float("nan")

    # Slice a 40 ms window centred on the nucleus for spectral analysis
    # (longer window gives better F0-resolution; ~25 Hz at 40 ms)
    window_half = 0.020
    s_start = max(0.0, nucleus_t - window_half)
    s_end = min(sound.get_total_duration(), nucleus_t + window_half)
    if s_end - s_start < 0.010:
        return float("nan")
    try:
        slice_sound = sound.extract_part(from_time=s_start, to_time=s_end,
                                            preserve_times=False)
        spec = slice_sound.to_spectrum()
        freqs, mag_db = _spectrum_to_mag_db(spec)
    except Exception:
        return float("nan")

    if len(freqs) < 4:
        return float("nan")

    # Find local maxima near H1 (=F0) and H2 (=2*F0), ±15 Hz search window
    def peak_db_near(target_hz: float) -> float:
        lo = max(freqs[0], target_hz - 20.0)
        hi = min(freqs[-1], target_hz + 20.0)
        in_band = (freqs >= lo) & (freqs <= hi)
        if not in_band.any():
            return float("nan")
        return float(np.max(mag_db[in_band]))

    h1 = peak_db_near(f0)
    h2 = peak_db_near(2.0 * f0)
    if math.isnan(h1) or math.isnan(h2):
        return float("nan")
    return float(h1 - h2)


# ---------------------------------------------------------------------------
# CPP (cepstral peak prominence)
# ---------------------------------------------------------------------------

def compute_cpp_db(
    sound: parselmouth.Sound,
    t_start: float,
    t_end: float,
) -> float:
    """Cepstral peak prominence (dB) over the syllable.

    Implementation: compute a PowerCepstrogram on the syllable slice, then
    take the mean CPP across cepstral frames.
    Returns NaN on failure.
    """
    if t_end <= t_start:
        return float("nan")
    # Praat's PowerCepstrogram needs a sound at least ~0.01 s long
    if t_end - t_start < 0.02:
        return float("nan")
    try:
        slice_sound = sound.extract_part(from_time=t_start, to_time=t_end,
                                            preserve_times=False)
        # PowerCepstrogram parameters: pitch_floor (Hz), time_step (s),
        # max_frequency (Hz), pre-emphasis_from (Hz)
        cepstrogram = call(slice_sound, "To PowerCepstrogram",
                            CPP_PITCH_FLOOR_HZ, 0.002, 5000.0, 50.0)
        # Get CPPS (smoothed CPP) over the whole cepstrogram
        cpps = call(cepstrogram, "Get CPPS",
                     "yes",  # subtract trend
                     0.01,   # time smoothing window (s)
                     0.001,  # quefrency smoothing window (s)
                     CPP_PITCH_FLOOR_HZ,
                     CPP_PITCH_CEILING_HZ,
                     0.05,   # tolerance
                     "Parabolic",
                     0.001,  # quefrency range start
                     0.05,   # quefrency range end
                     "Exponential decay",
                     "Robust slow")
        if cpps is None or math.isnan(cpps):
            return float("nan")
        return float(cpps)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Spectral tilt
# ---------------------------------------------------------------------------

def compute_spectral_tilt_db_per_oct(
    sound: parselmouth.Sound,
    t_start: float,
    t_end: float,
) -> float:
    """Slope (dB / octave) of the log-magnitude spectrum from
    `SPECTRAL_TILT_LOW_HZ` to `SPECTRAL_TILT_HIGH_HZ`. Computed via linear
    regression on the log-frequency axis.

    A more negative value indicates a steeper roll-off (breathier voice);
    a flatter or positive slope indicates pressed or creaky voice.
    Returns NaN on failure.
    """
    if t_end <= t_start or t_end - t_start < 0.02:
        return float("nan")
    try:
        slice_sound = sound.extract_part(from_time=t_start, to_time=t_end,
                                            preserve_times=False)
        spec = slice_sound.to_spectrum()
        freqs, mag_db = _spectrum_to_mag_db(spec)
    except Exception:
        return float("nan")

    if len(freqs) < 8:
        return float("nan")

    in_band = (freqs >= SPECTRAL_TILT_LOW_HZ) & (freqs <= SPECTRAL_TILT_HIGH_HZ)
    if in_band.sum() < 8:
        return float("nan")
    band_freqs = freqs[in_band]
    band_levels = mag_db[in_band]

    # Linear regression: dB vs log2-frequency (slope is dB / octave)
    log2_freqs = np.log2(band_freqs)
    A = np.vstack([log2_freqs, np.ones_like(log2_freqs)]).T
    try:
        slope, _ = np.linalg.lstsq(A, band_levels, rcond=None)[0]
        if math.isnan(slope) or math.isinf(slope):
            return float("nan")
        return float(slope)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Convenience: compute all four at once
# ---------------------------------------------------------------------------

def compute_voice_quality(
    sound: parselmouth.Sound,
    pitch: parselmouth.Pitch,
    intensity: parselmouth.Intensity,
    t_start: float,
    t_end: float,
    speaker_median_f0_hz: float,
) -> tuple[float, float, float, float]:
    """Returns (creak_fraction, h1_minus_h2_db, cpp_db, spectral_tilt_db_per_oct)."""
    creak = compute_creak_fraction(pitch, intensity, t_start, t_end,
                                     speaker_median_f0_hz)
    h1h2 = compute_h1_minus_h2_db(sound, pitch, t_start, t_end)
    cpp = compute_cpp_db(sound, t_start, t_end)
    tilt = compute_spectral_tilt_db_per_oct(sound, t_start, t_end)
    return creak, h1h2, cpp, tilt
