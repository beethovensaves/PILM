"""
Find MELD utterances with identical (normalized) text but different emotion
labels — minimal pairs where prosody is the only possible differentiator.

Then evaluate whether prosody-only classification beats text-only on these
pairs. The hypothesis: text is forced to predict the same label for all
group members (it sees the same input), so its in-group accuracy is
upper-bounded by the majority-class share of the group. Prosody can
predict differently for different members, so it has a higher ceiling.

Pipeline:
    1. Concatenate train + dev + test metadata, attach parametric vectors.
    2. Normalize utterance text: lowercase, strip punctuation/whitespace.
    3. Group by normalized text. Keep groups with ≥`min_group_size` members
       AND ≥2 distinct labels.
    4. Train a TEXT-only probe and a PROSODY-only probe on the train split.
    5. For each minimal-pair group, score how often each probe gets the
       member right.
    6. Report headline accuracy + the most informative example pairs.

Usage:
    .venv/bin/python scripts/find_minimal_pairs.py \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-parametric data/meld/parametric_prosody_dev_mfa.jsonl data/meld/parametric_prosody_test_mfa.jsonl \\
        --eval-csv data/meld/MELD.Raw/dev_sent_emo_cleaned.csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --label emotion
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.compare_prosody_text import (
    clean_utterance_text, load_parametric, pool_to_utterance,
)

warnings.filterwarnings("ignore")

PUNCT_STRIP = re.compile(r"[^\w\s']")


def normalize(s: str) -> str:
    s = clean_utterance_text(s).lower()
    s = PUNCT_STRIP.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def assemble(parametric_paths: list[Path], csv_paths: list[Path], label_col: str) -> pd.DataFrame:
    parametric: dict[str, np.ndarray] = {}
    for p in parametric_paths:
        parametric.update(load_parametric(p))
    dfs = [pd.read_csv(c) for c in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    df["normalized"] = df["Utterance"].apply(normalize)
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df = df[df["utterance_id"].isin(parametric)].copy()
    df["pooled_vec"] = df["utterance_id"].map(
        lambda uid: pool_to_utterance(parametric[uid], "mean+max+std")
    )
    return df


def make_text_probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, lowercase=True,
            token_pattern=r"(?u)\b\w+\b|[!?]+|\.{2,}",
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_prosody_probe() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=2000, C=1.0)),
    ])


def vstack_pooled(series: pd.Series) -> np.ndarray:
    arr = np.vstack(series.values)
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path, nargs="+", required=True)
    parser.add_argument("--train-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-parametric", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--label", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--min-group-size", type=int, default=3,
                        help="Minimum number of utterances in a minimal-pair group")
    parser.add_argument("--top-groups", type=int, default=15,
                        help="How many groups to print")
    parser.add_argument("--strict", action="store_true",
                        help="Force text classifier to see the SAME normalized input as the grouping key — "
                             "no punctuation, lowercased, whitespace collapsed. The strict minimal-pair test.")
    args = parser.parse_args()

    label_col = "Emotion" if args.label == "emotion" else "Sentiment"

    print(f"[load] train: {args.train_csv}")
    train_df = assemble(args.train_parametric, args.train_csv, label_col)
    print(f"[load] eval:  {args.eval_csv}")
    eval_df = assemble(args.eval_parametric, args.eval_csv, label_col)
    print(f"  train={len(train_df)}  eval={len(eval_df)}")

    # Find groups within EVAL only (the held-out set)
    grp = eval_df.groupby("normalized")
    minimal: list[dict] = []
    for text_norm, g in grp:
        if not text_norm:
            continue
        if len(g) < args.min_group_size:
            continue
        if g[label_col].nunique() < 2:
            continue
        minimal.append({
            "text": text_norm,
            "n": len(g),
            "n_labels": int(g[label_col].nunique()),
            "label_counts": dict(g[label_col].value_counts()),
            "group_idxs": list(g.index),
        })
    minimal.sort(key=lambda d: (d["n_labels"], d["n"]), reverse=True)
    print(f"\n[groups] {len(minimal)} minimal-pair groups in eval (≥{args.min_group_size} utts, ≥2 labels)")

    # Train probes on train split only
    text_clf = make_text_probe()
    if args.strict:
        text_train = train_df["normalized"].astype(str).tolist()
        text_eval  = eval_df["normalized"].astype(str).tolist()
        print(f"[strict] text classifier sees normalized (no-punct, lowercased) input on both sides")
    else:
        text_train = train_df["Utterance"].astype(str).tolist()
        text_eval  = eval_df["Utterance"].astype(str).tolist()
    text_clf.fit(text_train, train_df[label_col].values)

    pros_clf = make_prosody_probe()
    Xtr = vstack_pooled(train_df["pooled_vec"])
    pros_clf.fit(Xtr, train_df[label_col].values)

    # Predict on the eval set members
    Xte = vstack_pooled(eval_df["pooled_vec"])
    eval_df = eval_df.copy()
    eval_df["text_pred"] = text_clf.predict(text_eval)
    eval_df["pros_pred"] = pros_clf.predict(Xte)

    # Aggregate: across all members of all minimal-pair groups, who's right more often?
    member_idxs = [i for g in minimal for i in g["group_idxs"]]
    sub = eval_df.loc[member_idxs]
    text_correct = (sub["text_pred"] == sub[label_col]).sum()
    pros_correct = (sub["pros_pred"] == sub[label_col]).sum()
    print(f"\n[in minimal-pair groups, n={len(sub)} utterances]")
    print(f"  text-only correct:    {text_correct} ({100*text_correct/len(sub):.1f}%)")
    print(f"  prosody-only correct: {pros_correct} ({100*pros_correct/len(sub):.1f}%)")

    # Per-group sharper test: text gets ONE prediction per group (same input→same output).
    # Its in-group accuracy ceiling is the majority class share. Prosody can do better.
    groups_text_better = 0
    groups_pros_better = 0
    groups_tied = 0
    for d in minimal:
        sub_g = eval_df.loc[d["group_idxs"]]
        t_acc = (sub_g["text_pred"] == sub_g[label_col]).mean()
        p_acc = (sub_g["pros_pred"] == sub_g[label_col]).mean()
        if t_acc > p_acc:
            groups_text_better += 1
        elif p_acc > t_acc:
            groups_pros_better += 1
        else:
            groups_tied += 1
    print(f"\n[per-group winner among {len(minimal)} groups]")
    print(f"  text wins:    {groups_text_better}")
    print(f"  prosody wins: {groups_pros_better}")
    print(f"  tied:         {groups_tied}")

    # Show top groups with predictions per member
    print(f"\n[top {args.top_groups} groups — text→ vs prosody→ vs true]")
    print("=" * 90)
    for d in minimal[: args.top_groups]:
        sub_g = eval_df.loc[d["group_idxs"]]
        true_dist = ", ".join(f"{l}×{c}" for l, c in d["label_counts"].items())
        print(f"\n  '{d['text']}'  (n={d['n']}, {d['n_labels']} labels: {true_dist})")
        # Show every member with predictions, deduping on (true, text_pred, pros_pred)
        seen: set = set()
        examples = []
        for _, row in sub_g.iterrows():
            key = (row[label_col], row["text_pred"], row["pros_pred"])
            if key in seen:
                continue
            seen.add(key)
            examples.append(row)
            if len(examples) >= 8:
                break
        for row in examples:
            mark = ""
            if row["pros_pred"] == row[label_col] and row["text_pred"] != row[label_col]:
                mark = "  ← prosody correct"
            elif row["text_pred"] == row[label_col] and row["pros_pred"] != row[label_col]:
                mark = "  ← text correct"
            print(f"    [{row['Speaker']:<10s}] true={row[label_col]:<10s} "
                  f"text→{row['text_pred']:<10s} prosody→{row['pros_pred']:<10s}{mark}")


if __name__ == "__main__":
    main()
