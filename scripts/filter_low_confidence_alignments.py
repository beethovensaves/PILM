"""
Filter a parametric-prosody JSONL by MFA's per-utterance alignment quality.

MFA writes an `alignment_analysis.csv` next to its TextGrid output with per-
file diagnostics. This script reads that, identifies utterances whose
alignment is low-confidence, and writes a new JSONL with those rows
dropped (or marked).

Quality columns we use:
    - overall_log_likelihood: higher = better. We drop the bottom percentile.
    - phone_duration_deviation: higher = phones forced into mismatched durations.
      We drop the top percentile.

Defaults are conservative (drop ~10% per criterion, with overlap allowed —
union ends up ~15-20%). Override via --ll-percentile and --pdd-percentile.

Usage:
    .venv/bin/python scripts/filter_low_confidence_alignments.py \\
        --parametric data/meld/parametric_prosody_dev_mfa.jsonl \\
        --quality-csv data/meld/dev_textgrids_mfa/alignment_analysis.csv \\
        --out data/meld/parametric_prosody_dev_mfa_filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parametric", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, required=True,
                        help="MFA alignment_analysis.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ll-percentile", type=float, default=10.0,
                        help="Drop utterances below this percentile of overall_log_likelihood (default 10).")
    parser.add_argument("--pdd-percentile", type=float, default=90.0,
                        help="Drop utterances above this percentile of phone_duration_deviation (default 90).")
    parser.add_argument("--report", action="store_true",
                        help="Just print stats, don't write filtered file.")
    args = parser.parse_args()

    qdf = pd.read_csv(args.quality_csv)
    n_total = len(qdf)
    print(f"[load] alignment_analysis.csv: {n_total} rows")

    ll = qdf["overall_log_likelihood"].astype(float)
    pdd = qdf["phone_duration_deviation"].astype(float)

    # NaN rows = failed alignments. Drop them outright.
    nan_mask = ll.isna() | pdd.isna()
    n_nan = int(nan_mask.sum())
    if n_nan:
        print(f"  {n_nan} rows have NaN quality columns (failed alignments — flagged)")

    # Compute percentiles only over non-NaN rows
    ll_valid = ll.dropna()
    pdd_valid = pdd.dropna()
    ll_thresh = float(np.percentile(ll_valid, args.ll_percentile))
    pdd_thresh = float(np.percentile(pdd_valid, args.pdd_percentile))
    print(f"  overall_log_likelihood: drop < {ll_thresh:.2f} (P{args.ll_percentile})")
    print(f"  phone_duration_dev:     drop > {pdd_thresh:.2f} (P{args.pdd_percentile})")

    bad_ll = qdf[ll < ll_thresh]
    bad_pdd = qdf[pdd > pdd_thresh]
    bad_nan = qdf[nan_mask]
    bad_files = set(bad_ll["file"]) | set(bad_pdd["file"]) | set(bad_nan["file"])
    print(f"  flagged: {len(bad_ll)} (low log-lik), {len(bad_pdd)} (high duration dev), {len(bad_nan)} (NaN)")
    print(f"  union of bad files: {len(bad_files)} ({100*len(bad_files)/n_total:.1f}% of all)")

    if args.report:
        return

    # Stream filter the parametric JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_kept = 0
    with args.parametric.open() as f_in, args.out.open("w") as f_out:
        for line in f_in:
            n_in += 1
            d = json.loads(line)
            uid = d["utterance_id"]
            if uid in bad_files:
                continue
            f_out.write(line)
            n_kept += 1
    print(f"[write] kept {n_kept} of {n_in} utterances → {args.out}")


if __name__ == "__main__":
    main()
