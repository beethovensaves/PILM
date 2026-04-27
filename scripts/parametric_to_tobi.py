"""
Rule-based mapper from the D19 18-dim parametric prosody vector to ToBI
categories (pitch accent, break index, boundary tone). Replaces the
supervised auto-ToBI labeler we couldn't train (no BURNC/NXT access).

Decision rules are derived from AM/ToBI signature definitions and the
per-class signature table in docs/writeups/parametric_prosody_extractor.md
§6.5. Thresholds are conservative — meant for a "first pass" that we'll
calibrate against NXT gold once LDC access lands.

Inputs:
    - per-utterance parametric JSONL (output of extract_parametric_prosody_mfa.py)

Outputs:
    - same JSONL, augmented with `tobi_accent`, `tobi_boundary_tone`, and
      `tobi_break_index` fields per syllable
    - a summary count of predicted categories (printed)

Categories:

  Pitch accents (per syllable):
    NONE   - no accent assigned
    H*     - high accent, peak mid-to-early in syllable
    L*     - low accent, low f0 throughout
    L+H*   - late-peak rising accent (most prominent rise pattern)
    L*+H   - very-late peak, rise after nucleus (delayed peak)
    H+!H*  - downstep marker (relies on f0_reset across syllables)

  Boundary tones (at last syllable of intonational phrase only):
    NONE
    H%     - high final, rising offset
    L%     - low final, falling offset

  Break indices (per syllable, at right edge):
    1      - default within-phrase
    3      - intermediate phrase boundary
    4      - intonational phrase boundary

Usage:
    .venv/bin/python scripts/parametric_to_tobi.py \\
        --in data/meld/parametric_prosody_test_mfa.jsonl \\
        --out data/meld/parametric_prosody_test_tobi.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

# Index mapping into the 18-dim vec — keep in sync with D19
F0_ONSET_ST          = 0
F0_NUCLEUS_ST        = 1
F0_OFFSET_ST         = 2
F0_MAX_ST            = 3
F0_MIN_ST            = 4
F0_RANGE_ST          = 5
F0_SLOPE_ST_PER_MS   = 6
F0_PEAK_POS          = 7
F0_RISE_AMP          = 8
F0_FALL_AMP          = 9
TILT                 = 10
RMS_MAX_Z            = 11
RMS_MEAN_Z           = 12
SYL_DURATION_Z       = 13
NUC_DURATION_Z       = 14
PAUSE_AFTER_MS       = 15
FINAL_LENGTHENING    = 16
F0_RESET_ST          = 17

# Thresholds — tuneable, picked from AM-theory guidance + spot-checking on MELD
T = {
    # Accent presence: enough F0 movement OR enough loudness peak
    "min_range_for_accent": 2.0,        # ST
    "min_rms_for_accent":   0.4,        # speaker-z

    # L*: low target, no significant rise
    "lstar_max_st_max":     0.0,        # f0_max stays at/below median
    "lstar_rise_max":       1.5,        # ST

    # L+H*: late-peak rise, big rise, positive tilt
    "lplush_peak_min":      0.55,       # peak in second half of syllable
    "lplush_tilt_min":      0.15,
    "lplush_rise_min":      2.0,        # ST

    # L*+H: very-late peak (rise after nucleus)
    "lstarplush_peak_min":  0.85,
    "lstarplush_tilt_min":  0.5,

    # Boundary tones (last syllable only)
    "high_boundary_offset_min":   1.0,
    "high_boundary_slope_min":    0.004,
    "low_boundary_offset_max":   -1.0,
    "low_boundary_slope_max":    -0.004,

    # Break indices
    "break4_pause_min_ms":        200,
    "break4_reset_min_st":        4.0,
    "break4_lengthen_min":        1.25,
    "break3_pause_min_ms":        80,
    "break3_reset_min_st":        2.5,
    "break3_lengthen_min":        1.10,

    # Downstep (H+!H*): high accent on this syllable PRECEDED by a higher accent
    "downstep_min_drop_st":       2.5,
}


def safe(v) -> float:
    """NaN/None → 0 for classification thresholds."""
    if v is None:
        return 0.0
    if isinstance(v, float) and math.isnan(v):
        return 0.0
    return float(v)


def predict_accent_basic(vec: list, voiced_fraction: float) -> str:
    """Pitch accent prediction from per-syllable vec only (no neighbor context).
    Downstep classification happens in a second pass (predict_accents_with_context)."""
    if voiced_fraction < 0.2:
        return "NONE"  # can't be sure of pitch accent on unvoiced syllable

    range_st  = safe(vec[F0_RANGE_ST])
    max_st    = safe(vec[F0_MAX_ST])
    min_st    = safe(vec[F0_MIN_ST])
    rms_max_z = safe(vec[RMS_MAX_Z])
    peak_pos  = safe(vec[F0_PEAK_POS])
    rise_amp  = safe(vec[F0_RISE_AMP])
    tilt      = safe(vec[TILT])

    # No accent: low movement and low energy
    if range_st < T["min_range_for_accent"] and rms_max_z < T["min_rms_for_accent"]:
        return "NONE"

    # L*: low target, weak rise, max f0 at-or-below speaker median
    if max_st <= T["lstar_max_st_max"] and rise_amp < T["lstar_rise_max"]:
        return "L*"

    # L*+H: very late peak, strong tilt
    if peak_pos >= T["lstarplush_peak_min"] and tilt >= T["lstarplush_tilt_min"]:
        return "L*+H"

    # L+H*: late-peak rising accent
    if (peak_pos >= T["lplush_peak_min"]
        and tilt >= T["lplush_tilt_min"]
        and rise_amp >= T["lplush_rise_min"]):
        return "L+H*"

    # Default high accent
    return "H*"


def upgrade_accent_for_downstep(prev_max_st: float | None, this_vec: list, this_accent: str) -> str:
    """If THIS syllable has H* but with f0_max meaningfully BELOW the previous
    syllable's accent peak, retag as H+!H* (downstep). Standard ToBI logic."""
    if this_accent != "H*" or prev_max_st is None:
        return this_accent
    this_max = safe(this_vec[F0_MAX_ST])
    if (prev_max_st - this_max) >= T["downstep_min_drop_st"]:
        return "H+!H*"
    return this_accent


def predict_break(vec: list, is_last_in_utt: bool) -> int:
    pause_ms  = safe(vec[PAUSE_AFTER_MS])
    reset_st  = abs(safe(vec[F0_RESET_ST]))
    lengthen  = safe(vec[FINAL_LENGTHENING])

    if is_last_in_utt:
        # Utterances usually end at IP boundary
        return 4
    if (pause_ms >= T["break4_pause_min_ms"]
        or (reset_st >= T["break4_reset_min_st"] and lengthen >= T["break4_lengthen_min"])):
        return 4
    if (pause_ms >= T["break3_pause_min_ms"]
        or (reset_st >= T["break3_reset_min_st"] and lengthen >= T["break3_lengthen_min"])):
        return 3
    return 1


def predict_boundary_tone(vec: list, voiced_fraction: float, break_idx: int) -> str:
    """Boundary tones land at break-index-4 (intonational phrase) boundaries."""
    if break_idx != 4 or voiced_fraction < 0.2:
        return "NONE"
    offset_st = safe(vec[F0_OFFSET_ST])
    slope     = safe(vec[F0_SLOPE_ST_PER_MS])
    if offset_st >= T["high_boundary_offset_min"] or slope >= T["high_boundary_slope_min"]:
        return "H%"
    if offset_st <= T["low_boundary_offset_max"] or slope <= T["low_boundary_slope_max"]:
        return "L%"
    return "NONE"


def label_utterance(syllables: list[dict]) -> list[dict]:
    """Apply rules in order, with cross-syllable context for downstep + last-syllable handling."""
    labeled = []
    prev_accent_max_st = None
    n = len(syllables)
    for i, syl in enumerate(syllables):
        vec = syl["vec"]
        vf  = float(syl.get("voiced_fraction", 0.0))
        is_last = (i == n - 1)

        accent = predict_accent_basic(vec, vf)
        if accent in ("H*", "L+H*"):
            accent = upgrade_accent_for_downstep(prev_accent_max_st, vec, accent)

        break_idx = predict_break(vec, is_last)
        boundary  = predict_boundary_tone(vec, vf, break_idx)

        out = dict(syl)
        out["tobi_accent"]         = accent
        out["tobi_break_index"]    = break_idx
        out["tobi_boundary_tone"]  = boundary
        labeled.append(out)

        if accent != "NONE":
            prev_accent_max_st = safe(vec[F0_MAX_ST])
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    accent_counter   = Counter()
    boundary_counter = Counter()
    break_counter    = Counter()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.in_path.open() as f_in, args.out.open("w") as f_out:
        n = 0
        for line in f_in:
            d = json.loads(line)
            d["syllables"] = label_utterance(d["syllables"])
            for syl in d["syllables"]:
                accent_counter[syl["tobi_accent"]] += 1
                boundary_counter[syl["tobi_boundary_tone"]] += 1
                break_counter[syl["tobi_break_index"]] += 1
            f_out.write(json.dumps(d) + "\n")
            n += 1

    total_syl = sum(accent_counter.values())
    print(f"[done] {n} utterances, {total_syl} syllables")
    print(f"\nPitch accent distribution:")
    for k, v in accent_counter.most_common():
        print(f"  {k:<8s}  {v:6d}  ({100*v/total_syl:5.1f}%)")
    print(f"\nBoundary tone distribution:")
    for k, v in boundary_counter.most_common():
        print(f"  {k:<8s}  {v:6d}  ({100*v/total_syl:5.1f}%)")
    print(f"\nBreak index distribution:")
    for k, v in sorted(break_counter.items()):
        print(f"  {k:<8d}  {v:6d}  ({100*v/total_syl:5.1f}%)")


if __name__ == "__main__":
    main()
