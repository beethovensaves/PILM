"""
Convert MELD's per-utterance emotion + sentiment labels into the
probe-target JSONL schema used by `scripts/validate_parametric_prosody.py`.

This is the *primary Phase 1.5 probe target* (per pivot 2026-04-25).
AuToBI was the original choice but its pre-trained classifier `.model`
files are not readily available; MELD ships emotion labels for free
and they validate the same scientific question (does the parametric
vector encode prosodically-informed structure?).

Output schema (one JSON object per line, one per utterance):

    {
        "utterance_id":  "dia0_utt0",
        "speaker_id":    "Phoebe",
        "emotion":       "sadness",        # 7-way: anger / disgust / fear / joy / neutral / sadness / surprise
        "sentiment":     "negative",       # 3-way: negative / neutral / positive
        "utterance":     "Oh my God, hes lost it.",
        "season":        4,
        "episode":       7
    }

The validator joins this against the parametric extractor's JSONL on
`utterance_id` and asks: from the per-syllable parametric vectors
(pooled to utterance level), can a linear or MLP probe recover the
emotion / sentiment label better than a chance baseline?

Usage:
    .venv/bin/python scripts/build_emotion_probe_targets.py \\
        --metadata data/meld/MELD.Raw/dev_sent_emo.csv \\
        --out data/meld/emotion_probe_targets_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EMOTION_VOCAB = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
SENTIMENT_VOCAB = {"negative", "neutral", "positive"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True,
                        help="MELD CSV — dev_sent_emo.csv / train_sent_emo.csv / test_sent_emo.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    required = {"Dialogue_ID", "Utterance_ID", "Speaker", "Emotion", "Sentiment", "Utterance"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"metadata missing columns: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_unknown_emotion = 0
    n_unknown_sentiment = 0

    with args.out.open("w") as f_out:
        for row in df.itertuples():
            emotion = str(row.Emotion).strip().lower()
            sentiment = str(row.Sentiment).strip().lower()
            if emotion not in EMOTION_VOCAB:
                n_unknown_emotion += 1
            if sentiment not in SENTIMENT_VOCAB:
                n_unknown_sentiment += 1
            record = {
                "utterance_id": f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}",
                "speaker_id":   row.Speaker,
                "emotion":      emotion,
                "sentiment":    sentiment,
                "utterance":    row.Utterance,
            }
            # Optional metadata if present
            if hasattr(row, "Season"):
                record["season"] = int(row.Season)
            if hasattr(row, "Episode"):
                record["episode"] = int(row.Episode)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} probe targets to {args.out}")
    if n_unknown_emotion:
        print(f"  warning: {n_unknown_emotion} rows had unrecognized emotion labels")
    if n_unknown_sentiment:
        print(f"  warning: {n_unknown_sentiment} rows had unrecognized sentiment labels")


if __name__ == "__main__":
    main()
