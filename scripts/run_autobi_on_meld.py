"""
Run AuToBI (Rosenberg 2010) on MELD utterance audio and emit per-utterance
JSONL with predicted ToBI events as **probe targets** (per D5).

AuToBI is a Java tool. This script wraps it as a subprocess driver,
caches its TextGrid outputs, and parses them into a JSONL schema that
aligns with the parametric extractor's output (`scripts/extract_parametric_prosody.py`).

⚠ Setup status as of 2026-04-25:
    AuToBI's pre-trained classifier models (.model files) are NOT bundled
    with the GitHub repo and the original CUNY hosting page is offline.
    To use this script you need to provide:
      --autobi-jar    path to a built AuToBI.jar (build via `ant` from
                      https://github.com/AndrewRosenberg/AuToBI)
      --autobi-models directory containing the .model files for the
                      tasks listed in DEFAULT_AUTOBI_TASKS below.

    Until those exist, prefer `scripts/build_emotion_probe_targets.py`
    which uses MELD's existing utterance-level emotion labels as the
    probe target — same Phase 1.5 validation logic, no AuToBI dependency.

Usage:
    .venv/bin/python scripts/run_autobi_on_meld.py \\
        --in-dir data/meld/MELD.Raw/dev_splits_complete \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --autobi-jar deps/AuToBI/AuToBI.jar \\
        --autobi-models deps/AuToBI/models \\
        --out data/meld/autobi_labels_dev.jsonl

Output (one JSON object per line):
    {
      "utterance_id": "dia0_utt0",
      "speaker_id": "Joey",
      "audio_path": "...",
      "events": {
        "pitch_accent_hyp":            [{"time_s": 0.18, "label": "H*"}, ...],   # point tier
        "phrase_accent_hyp":           [{"time_s": 1.42, "label": "L-"}, ...],
        "boundary_tone_hyp":           [{"time_s": 1.42, "label": "L%"}, ...],
        "intermediate_phrase_boundary_hyp": [...],
        "intonational_phrase_boundary_hyp":  [...]
      }
    }

The probe-validation script (scripts/validate_parametric_prosody.py) will
align these AuToBI events to the parametric extractor's syllable nuclei
by nearest-time matching.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import parselmouth
from parselmouth.praat import call

# Reuse the audio-path + ensure_wav helpers from the parametric extractor
from scripts.extract_parametric_prosody import meld_audio_path, ensure_wav


# ---------------------------------------------------------------------------
# AuToBI invocation
# ---------------------------------------------------------------------------

# These flag names map to AuToBI's CLI arguments. Each value is the
# expected filename inside --autobi-models. Adjust if your model bundle
# uses different naming (e.g. SWB.* instead of BURNC.*).
DEFAULT_AUTOBI_TASKS: dict[str, str] = {
    "pitch_accent_detector":                  "BURNC.PitchAccentDetector.model",
    "pitch_accent_classifier":                "BURNC.PitchAccentClassifier.model",
    "intonational_phrase_boundary_detector":  "BURNC.IntonationalPhraseBoundaryDetector.model",
    "intermediate_phrase_boundary_detector":  "BURNC.IntermediatePhraseBoundaryDetector.model",
    "boundary_tone_classifier":               "BURNC.BoundaryToneClassifier.model",
    "phrase_accent_classifier":               "BURNC.PhraseAccentClassifier.model",
}

AUTOBI_TIMEOUT_S = 60


def autobi_command(
    wav_path: Path,
    jar: Path,
    models_dir: Path,
    out_textgrid: Path,
) -> list[str]:
    cmd = ["java", "-jar", str(jar), f"-wav_file={wav_path}"]
    for flag, model_filename in DEFAULT_AUTOBI_TASKS.items():
        model_path = models_dir / model_filename
        cmd.append(f"-{flag}={model_path}")
    cmd.append(f"-out_file={out_textgrid}")
    return cmd


def run_autobi_one(wav_path: Path, jar: Path, models_dir: Path, out_textgrid: Path) -> bool:
    """Invoke AuToBI on one wav. Returns True on success."""
    cmd = autobi_command(wav_path, jar, models_dir, out_textgrid)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=AUTOBI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"    autobi timeout on {wav_path.name}", file=sys.stderr)
        return False
    if result.returncode != 0:
        # Print only the tail of stderr to keep logs readable
        stderr_tail = (result.stderr or b"").decode(errors="replace")[-400:]
        print(f"    autobi rc={result.returncode} on {wav_path.name}: {stderr_tail}", file=sys.stderr)
        return False
    return out_textgrid.exists() and out_textgrid.stat().st_size > 0


# ---------------------------------------------------------------------------
# TextGrid parsing — uses parselmouth/Praat
# ---------------------------------------------------------------------------

def parse_autobi_textgrid(path: Path) -> dict[str, list[dict]]:
    """Read AuToBI's output TextGrid and extract events per tier.

    AuToBI produces:
      - point tiers (named *_hyp) for predicted accents, boundary tones, etc.
      - interval tiers for words / syllables that carry the predictions

    We surface point tiers as [{"time_s", "label"}, ...] and interval
    tiers as [{"t_start_s", "t_end_s", "label"}, ...]. Empty intervals
    (silence) and empty labels are filtered out.
    """
    tg = parselmouth.Data.read(str(path))
    n_tiers = int(call(tg, "Get number of tiers"))
    out: dict[str, list[dict]] = {}
    for i in range(1, n_tiers + 1):
        tier_name = call(tg, "Get tier name", i)
        is_interval = bool(call(tg, "Is interval tier", i))
        if is_interval:
            n_int = int(call(tg, "Get number of intervals", i))
            entries: list[dict] = []
            for j in range(1, n_int + 1):
                t_start = float(call(tg, "Get start time of interval", i, j))
                t_end = float(call(tg, "Get end time of interval", i, j))
                label = call(tg, "Get label of interval", i, j)
                if label and label.strip():
                    entries.append({
                        "t_start_s": round(t_start, 4),
                        "t_end_s": round(t_end, 4),
                        "label": label,
                    })
            out[tier_name] = entries
        else:
            n_pts = int(call(tg, "Get number of points", i))
            entries = []
            for j in range(1, n_pts + 1):
                t = float(call(tg, "Get time of point", i, j))
                label = call(tg, "Get label of point", i, j)
                if label and label.strip():
                    entries.append({"time_s": round(t, 4), "label": label})
            out[tier_name] = entries
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--autobi-jar", type=Path,
                        default=Path(os.environ.get("AUTOBI_JAR", "")) if os.environ.get("AUTOBI_JAR") else None)
    parser.add_argument("--autobi-models", type=Path,
                        default=Path(os.environ.get("AUTOBI_MODELS", "")) if os.environ.get("AUTOBI_MODELS") else None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wav-cache", type=Path, default=None)
    parser.add_argument("--textgrid-cache", type=Path, default=None,
                        help="Where to keep AuToBI TextGrid outputs. Default: <in-dir>/_autobi_cache")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.autobi_jar is None or not args.autobi_jar.exists():
        sys.exit(
            "AuToBI jar not found. Provide --autobi-jar or set AUTOBI_JAR env var.\n"
            "Build AuToBI with `ant` from https://github.com/AndrewRosenberg/AuToBI."
        )
    if args.autobi_models is None or not args.autobi_models.exists():
        sys.exit(
            "AuToBI models dir not found. Provide --autobi-models or set AUTOBI_MODELS.\n"
            "Pre-trained .model files are not bundled with the AuToBI repo and the original CUNY\n"
            "hosting page is offline. You may need to train them yourself or locate a mirror.\n"
            "Until then, see scripts/build_emotion_probe_targets.py for a no-AuToBI alternative."
        )
    # Verify java is available
    if subprocess.run(["which", "java"], capture_output=True).returncode != 0:
        sys.exit("`java` not on PATH. AuToBI requires a JRE.")

    # Verify expected models present
    missing = [
        f for f in DEFAULT_AUTOBI_TASKS.values()
        if not (args.autobi_models / f).exists()
    ]
    if missing:
        print(f"warn: missing model files in {args.autobi_models}: {missing}", file=sys.stderr)

    df = pd.read_csv(args.metadata)
    required_cols = {"Dialogue_ID", "Utterance_ID", "Speaker"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"metadata missing required columns: {required_cols - set(df.columns)}")

    wav_cache = args.wav_cache or (args.in_dir / "_wav_cache")
    tg_cache = args.textgrid_cache or (args.in_dir / "_autobi_cache")
    tg_cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    with args.out.open("w") as f_out:
        for row in df.itertuples():
            if args.limit and n_written >= args.limit:
                break
            audio = meld_audio_path(args.in_dir, row.Dialogue_ID, row.Utterance_ID)
            if audio is None:
                n_skipped += 1
                continue
            wav = ensure_wav(audio, wav_cache)
            if wav is None:
                n_skipped += 1
                continue

            uid = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
            tg_out = tg_cache / f"{uid}.TextGrid"
            if not (tg_out.exists() and tg_out.stat().st_size > 0):
                ok = run_autobi_one(wav, args.autobi_jar, args.autobi_models, tg_out)
                if not ok:
                    n_skipped += 1
                    continue

            try:
                events = parse_autobi_textgrid(tg_out)
            except Exception as e:
                print(f"    parse failed for {uid}: {e}", file=sys.stderr)
                n_skipped += 1
                continue

            f_out.write(json.dumps({
                "utterance_id": uid,
                "speaker_id": row.Speaker,
                "audio_path": str(audio),
                "events": events,
            }) + "\n")
            n_written += 1
            if n_written % 50 == 0:
                print(f"    written {n_written} (skipped {n_skipped})")

    print(f"Done. {n_written} utterances written, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
