"""
Text-only baseline for the same MELD question-prediction task that
`predict_question_from_prosody.py` evaluates with the parametric vector.
Provides the comparable text AUC so we can quote a clean text-vs-prosody gap
on the question task and compare it to AMI's text headroom (0.885 words-only).

Three text variants:

    1. text-with-punct       — full utterance, TF-IDF includes `?`/`!`/`.` tokens.
    2. words-only            — overpunctuation collapsed first, then `?`/`!`
                               stripped before TF-IDF. Closest analogue to AMI's
                               words-only (where `?` is mostly absent in the
                               source data and questions are DA-tagged).
    3. punct-only            — control: TF-IDF over the trailing punctuation
                               run only.

Reports train→test AUC + question-F1 on yn-only and (if --include-wh) all-Qs.

Usage:
    .venv/bin/python scripts/predict_question_from_text.py \\
        --train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-csv  data/meld/MELD.Raw/test_sent_emo_cleaned.csv
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from scripts.compare_prosody_text import clean_utterance_text

warnings.filterwarnings("ignore")

WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}


def is_wh_q(text: str) -> bool:
    if "?" not in text:
        return False
    m = re.search(r"[A-Za-z]+", text)
    return bool(m) and m.group(0).lower() in WH_WORDS


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["raw_text"] = df["Utterance"].astype(str)
    df["has_q"] = df["raw_text"].apply(lambda s: "?" in s)
    df["is_wh_q"] = df["raw_text"].apply(is_wh_q)
    df["is_yn_q"] = df["has_q"] & ~df["is_wh_q"]
    # text variants
    df["text_full"] = df["raw_text"].apply(clean_utterance_text)
    df["text_words"] = df["text_full"].apply(lambda s: re.sub(r"[!?\.]+", " ", s)).str.replace(r"\s+", " ", regex=True).str.strip()
    df["punct_only"] = df["text_full"].apply(lambda s: " ".join(re.findall(r"[!?]+|\.{2,}", s)) or "EMPTY")
    return df


def make_text_probe(token_pattern: str = r"(?u)\b\w+\b|[!?]+|\.{2,}") -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95,
            sublinear_tf=True, lowercase=True,
            token_pattern=token_pattern,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_punct_probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, lowercase=False, token_pattern=r"[^\s]+")),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def evaluate(probe, Xtr, ytr, Xte, yte, name: str) -> dict:
    probe.fit(Xtr, ytr)
    proba = probe.predict_proba(Xte)[:, 1]
    pred = probe.predict(Xte)
    auc = roc_auc_score(yte, proba)
    qf1 = f1_score(yte, pred, pos_label=1, zero_division=0)
    print(f"  {name:<28s}  AUC={auc:.4f}  question_F1={qf1:.4f}")
    return {"auc": float(auc), "question_f1": float(qf1)}


def run_task(train: pd.DataFrame, eval_: pd.DataFrame, label_col: str, drop_wh: bool) -> dict:
    if drop_wh:
        tr = train[~train["is_wh_q"]]
        te = eval_[~eval_["is_wh_q"]]
    else:
        tr, te = train, eval_
    ytr = tr[label_col].astype(int).values
    yte = te[label_col].astype(int).values
    print(f"  n_train={len(tr)} ({int(ytr.sum())} pos)  n_eval={len(te)} ({int(yte.sum())} pos)")

    out = {}
    out["text_full"]   = evaluate(make_text_probe(), tr["text_full"].tolist(),  ytr,
                                                       te["text_full"].tolist(),  yte,
                                                       "text-with-punct")
    out["text_words"]  = evaluate(make_text_probe(token_pattern=r"(?u)\b\w+\b"),
                                  tr["text_words"].tolist(), ytr,
                                  te["text_words"].tolist(), yte,
                                  "words-only")
    out["punct_only"]  = evaluate(make_punct_probe(),
                                  tr["punct_only"].tolist(), ytr,
                                  te["punct_only"].tolist(), yte,
                                  "punct-only")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--eval-csv",  type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] train {args.train_csv.name}")
    train = load(args.train_csv)
    print(f"[load] eval  {args.eval_csv.name}")
    eval_ = load(args.eval_csv)
    print(f"  train: {len(train)} utts ({int(train['is_yn_q'].sum())} yn-Q, "
          f"{int(train['is_wh_q'].sum())} wh-Q)")
    print(f"  eval:  {len(eval_)} utts ({int(eval_['is_yn_q'].sum())} yn-Q, "
          f"{int(eval_['is_wh_q'].sum())} wh-Q)")

    out = {}

    print()
    print("=" * 70)
    print("Task A — yes/no question detection (drop wh-Qs)")
    print("=" * 70)
    out["yn_only"] = run_task(train, eval_, "is_yn_q", drop_wh=True)

    print()
    print("=" * 70)
    print("Task B — any-question detection (keep wh-Qs)")
    print("=" * 70)
    out["all_q"] = run_task(train, eval_, "has_q", drop_wh=False)

    print()
    print("=" * 70)
    print("Cross-corpus comparison (anchor numbers)")
    print("=" * 70)
    print(f"  MELD (this run, yn-only):")
    print(f"    text-with-punct AUC: {out['yn_only']['text_full']['auc']:.4f}")
    print(f"    words-only AUC:      {out['yn_only']['text_words']['auc']:.4f}")
    print(f"    prosody (last-syl LR, EXP-007): 0.6503")
    print(f"  AMI (predict_da_from_text_ami.py, el.inf vs statement):")
    print(f"    text-with-punct AUC: 0.9342")
    print(f"    words-only AUC:      0.8854")
    print(f"    prosody:             TBD (needs AMI audio + parametric pipeline)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
