"""
Phrase-digest post-processor for the v3 parametric prosody specification.

Reads a parametric prosody JSONL (one utterance per line, with per-syllable
`vec`, `voiced_fraction`, etc.) and adds a 4-dimensional `phrase_digest`
field to each syllable. The digest is computed at the prosodic-phrase level
(heuristically defined; see below) and broadcast to every syllable in the
phrase.

Phrase-boundary heuristic (deterministic, no AuToBI dependency):
    Mark a phrase boundary AFTER syllable k if
        vec[15] (pause_after_ms)            > 200, OR
        vec[16] (final_lengthening_ratio)   > 1.3.
    Boundaries also occur implicitly at the start and end of the utterance.

The four phrase-digest dimensions:
    23. phrase_decl_slope_st_per_s   linear regression slope of voiced F0
                                      across the phrase, in semitones / second
                                      (negative = declining F0 across phrase)
    24. phrase_baseline_drift_st     min F0 in the last 200 ms of the phrase
                                      minus min F0 in the first 200 ms
                                      (negative = falling baseline)
    25. phrase_articulation_rate     syllable count / phrase duration in
                                      seconds (syl/s)
    26. phrase_mean_f0_st            mean voiced F0 across the phrase, in
                                      semitones rel. speaker median

Each phrase-digest dim is broadcast (replicated) to every syllable in the
phrase: the encoder sees the digest at every position rather than only at
the phrase boundary.

Usage:
    .venv/bin/python scripts/add_phrase_digest.py \\
        --in  data/meld/parametric_prosody_dev_mfa.jsonl \\
        --out data/meld/parametric_prosody_dev_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Indices into vec
PAUSE_AFTER_MS_IDX = 15
FINAL_LENGTHENING_IDX = 16

# Heuristic thresholds for phrase boundary detection
PAUSE_THRESHOLD_MS = 200.0
LENGTHENING_THRESHOLD = 1.3


def _f0_nucleus_st_or_nan(syl: dict) -> float:
    """Pull the f0_nucleus_st value (vec index 1) or NaN."""
    v = syl.get("vec", [])
    if len(v) < 2:
        return float("nan")
    val = v[1]
    return float("nan") if val is None else float(val)


def _t_mid_ms(syl: dict) -> float:
    return 0.5 * (syl.get("t_start_ms", 0.0) + syl.get("t_end_ms", 0.0))


def _pause_after_ms(syl: dict) -> float:
    v = syl.get("vec", [])
    if len(v) <= PAUSE_AFTER_MS_IDX:
        return 0.0
    val = v[PAUSE_AFTER_MS_IDX]
    return 0.0 if val is None else float(val)


def _final_lengthening_ratio(syl: dict) -> float:
    v = syl.get("vec", [])
    if len(v) <= FINAL_LENGTHENING_IDX:
        return 1.0
    val = v[FINAL_LENGTHENING_IDX]
    return 1.0 if val is None else float(val)


def find_phrase_boundaries(syllables: list[dict]) -> list[int]:
    """Return list of indices into syllables marking the END of each phrase
    (inclusive). Always includes the last syllable index."""
    if not syllables:
        return []
    boundaries: list[int] = []
    for i, syl in enumerate(syllables[:-1]):
        if _pause_after_ms(syl) > PAUSE_THRESHOLD_MS:
            boundaries.append(i)
        elif _final_lengthening_ratio(syl) > LENGTHENING_THRESHOLD:
            boundaries.append(i)
    boundaries.append(len(syllables) - 1)
    return boundaries


def compute_phrase_digest(syllables: list[dict], end_idx: int, start_idx: int) -> tuple[float, float, float, float]:
    """Compute (decl_slope_st_per_s, baseline_drift_st, articulation_rate, mean_f0_st)
    over syllables[start_idx:end_idx+1]."""
    phrase = syllables[start_idx:end_idx + 1]
    if not phrase:
        return (float("nan"),) * 4

    # F0 trajectory: collect (t_seconds, f0_nucleus_st) pairs from voiced syllables
    pts: list[tuple[float, float]] = []
    for syl in phrase:
        f0_st = _f0_nucleus_st_or_nan(syl)
        if math.isnan(f0_st):
            continue
        t_s = _t_mid_ms(syl) / 1000.0
        pts.append((t_s, f0_st))

    # Declination slope
    if len(pts) >= 2:
        ts = np.array([p[0] for p in pts])
        f0s = np.array([p[1] for p in pts])
        # least squares slope
        A = np.vstack([ts, np.ones_like(ts)]).T
        try:
            slope, _ = np.linalg.lstsq(A, f0s, rcond=None)[0]
            decl_slope = float(slope) if not (math.isnan(slope) or math.isinf(slope)) else float("nan")
        except Exception:
            decl_slope = float("nan")
    else:
        decl_slope = float("nan")

    # Baseline drift: last-200ms-window min - first-200ms-window min on f0_min_st (vec[4])
    def _f0_min_or_nan(syl: dict) -> float:
        v = syl.get("vec", [])
        if len(v) < 5:
            return float("nan")
        val = v[4]
        return float("nan") if val is None else float(val)

    phrase_t_start = phrase[0].get("t_start_ms", 0.0)
    phrase_t_end = phrase[-1].get("t_end_ms", 0.0)
    early_window_end = phrase_t_start + 200.0
    late_window_start = phrase_t_end - 200.0

    early_mins = [_f0_min_or_nan(s) for s in phrase if s.get("t_end_ms", 0.0) <= early_window_end]
    late_mins = [_f0_min_or_nan(s) for s in phrase if s.get("t_start_ms", 0.0) >= late_window_start]
    early_mins = [v for v in early_mins if not math.isnan(v)]
    late_mins = [v for v in late_mins if not math.isnan(v)]
    if early_mins and late_mins:
        baseline_drift = float(min(late_mins) - min(early_mins))
    else:
        baseline_drift = float("nan")

    # Articulation rate: syllables per second over phrase duration (s)
    duration_s = (phrase_t_end - phrase_t_start) / 1000.0
    if duration_s > 0:
        articulation_rate = float(len(phrase) / duration_s)
    else:
        articulation_rate = float("nan")

    # Mean F0 across phrase, in semitones rel. speaker median (vec[1] is already semitones)
    if pts:
        mean_f0_st = float(np.mean([f0 for _, f0 in pts]))
    else:
        mean_f0_st = float("nan")

    return decl_slope, baseline_drift, articulation_rate, mean_f0_st


def process_file(in_path: Path, out_path: Path) -> dict:
    n_lines = 0
    n_phrases_total = 0
    n_phrases_zero_voiced = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            syllables = d.get("syllables", [])
            if not syllables:
                fout.write(line + "\n")
                continue

            boundaries = find_phrase_boundaries(syllables)
            phrase_starts = [0] + [b + 1 for b in boundaries[:-1]]
            phrase_ends = boundaries

            for start_idx, end_idx in zip(phrase_starts, phrase_ends):
                digest = compute_phrase_digest(syllables, end_idx, start_idx)
                if math.isnan(digest[3]):  # mean_f0_st couldn't be computed
                    n_phrases_zero_voiced += 1
                digest_serializable = [None if math.isnan(x) else round(x, 4) for x in digest]
                # Broadcast to every syllable in the phrase
                for j in range(start_idx, end_idx + 1):
                    syllables[j]["phrase_digest"] = digest_serializable
                n_phrases_total += 1

            d["syllables"] = syllables
            d["n_phrases"] = len(boundaries)
            fout.write(json.dumps(d) + "\n")
            n_lines += 1

    return {
        "n_utterances": n_lines,
        "n_phrases_total": n_phrases_total,
        "n_phrases_zero_voiced": n_phrases_zero_voiced,
        "in_path": str(in_path),
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path}")
    result = process_file(args.in_path, args.out)
    print(f"Done: {result['n_utterances']} utterances, "
          f"{result['n_phrases_total']} phrases (of which "
          f"{result['n_phrases_zero_voiced']} had no voiced F0).")
    print(f"  in:  {result['in_path']}")
    print(f"  out: {result['out_path']}")


if __name__ == "__main__":
    main()
