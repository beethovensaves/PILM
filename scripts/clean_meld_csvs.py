"""
Clean MELD's *_sent_emo.csv files: fix mojibake (\\xc2\\x92 → ', etc.) and
collapse overpunctuation in the Utterance column.

MELD's released CSVs were saved with a Windows-1252 → UTF-8 misinterpretation
that turned 410+ apostrophes per split into invisible control characters
(U+0092). ftfy reverses this. We then collapse "?!" → "?", "!!" → "!" so
text features don't get a free emotion-leak channel.

Output: written next to the input as `<basename>_cleaned.csv`. Originals
are not overwritten unless --inplace is passed.

Usage:
    .venv/bin/python scripts/clean_meld_csvs.py \\
        --csvs data/meld/MELD.Raw/dev_sent_emo.csv \\
               data/meld/MELD.Raw/test_sent_emo.csv \\
               data/meld/MELD.Raw/train_sent_emo.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import ftfy
import pandas as pd


_PUNCT_RUN = re.compile(r"([!?]+|\.{2,})")


def _collapse(m: re.Match) -> str:
    run = m.group(0)
    if run.startswith("."):
        return "..."
    return "?" if "?" in run else "!"


def clean_text(s: str) -> str:
    """ftfy mojibake fix + overpunctuation collapse + whitespace normalize."""
    s = ftfy.fix_text(str(s))
    s = _PUNCT_RUN.sub(_collapse, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_csv(path: Path, out_path: Path, column: str = "Utterance") -> dict:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"{path.name}: missing required column {column!r}")
    n_changed = 0
    n_total = len(df)
    samples_before: list[str] = []
    samples_after: list[str] = []
    cleaned: list[str] = []
    for s in df[column].astype(str):
        c = clean_text(s)
        if c != s:
            n_changed += 1
            if len(samples_before) < 3:
                samples_before.append(s)
                samples_after.append(c)
        cleaned.append(c)
    df[column] = cleaned
    df.to_csv(out_path, index=False)
    return {
        "in": str(path),
        "out": str(out_path),
        "rows": n_total,
        "rows_changed": n_changed,
        "samples_before": samples_before,
        "samples_after": samples_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the input files (default: write *_cleaned.csv next to each)")
    parser.add_argument("--column", default="Utterance")
    args = parser.parse_args()

    for p in args.csvs:
        if not p.exists():
            print(f"!! missing: {p}")
            continue
        if args.inplace:
            out = p
        else:
            out = p.with_name(p.stem + "_cleaned" + p.suffix)
        result = clean_csv(p, out, args.column)
        print(f"[{p.name}] {result['rows_changed']} of {result['rows']} rows cleaned → {out.name}")
        for b, a in zip(result["samples_before"], result["samples_after"]):
            print(f"    before: {b!r:.100}")
            print(f"    after:  {a!r:.100}")


if __name__ == "__main__":
    main()
