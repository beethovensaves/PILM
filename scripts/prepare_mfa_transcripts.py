"""
Write MFA-compatible transcript .lab files alongside each MELD utterance wav.

For each row in the metadata CSV, writes a `dia<N>_utt<M>.lab` next to the
corresponding wav, containing the cleaned utterance text. MFA expects this
filename convention (matching the wav basename) and uses the .lab content
as the alignment target.

Light text cleaning:
  - Strip parenthetical stage directions, e.g. "(He's carrying an issue)"
  - Strip square-bracket annotations, e.g. "[laughing]"
  - Collapse whitespace
  - Drop rows whose cleaned text is empty
  - Skip wavs that aren't on disk (some mp4s never converted)

The cleaning is conservative — MFA's own normalizer handles punctuation,
case, and most apostrophe issues. We only drop content MFA can't speak.

Usage:
    .venv/bin/python scripts/prepare_mfa_transcripts.py \\
        --in-dir data/meld/MELD.Raw/dev_splits_complete \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PARENS = re.compile(r"\([^)]*\)")
BRACKETS = re.compile(r"\[[^\]]*\]")
MULTI_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = str(s)
    s = PARENS.sub(" ", s)
    s = BRACKETS.sub(" ", s)
    s = MULTI_WS.sub(" ", s)
    return s.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="Directory holding the per-utterance wavs.")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="MELD CSV with Dialogue_ID, Utterance_ID, Utterance.")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    n_written = 0
    n_missing_wav = 0
    n_empty_text = 0
    for row in df.itertuples():
        wav = args.in_dir / f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav"
        if not wav.exists():
            n_missing_wav += 1
            continue
        text = clean_text(row.Utterance)
        if not text:
            n_empty_text += 1
            continue
        lab = wav.with_suffix(".lab")
        lab.write_text(text + "\n", encoding="utf-8")
        n_written += 1

    print(f"wrote {n_written} .lab files to {args.in_dir}")
    print(f"  skipped {n_missing_wav} (wav not on disk)")
    print(f"  skipped {n_empty_text} (empty text after cleaning)")


if __name__ == "__main__":
    main()
