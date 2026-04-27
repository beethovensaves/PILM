"""
POS ablation: which part-of-speech category carries the most predictive
signal for emotion/sentiment, and how does prosody hold up when that
category is removed from text?

Three experiments, all train→test (train on MELD train, eval on test):

    1. Per-POS-only text features. For each POS in [NOUN, VERB, ADJ, ADV,
       INTJ, PRON, PUNCT, ...], train TF-IDF + LR using ONLY tokens of that
       POS. Reveals which POS categories carry the most discriminative info.
    2. Per-POS ablation. For each POS, train TF-IDF + LR on text with all
       tokens of that POS REMOVED. The biggest drop identifies the POS most
       responsible for text's strong performance.
    3. Prosody-vs-ablated-text. Pick the worst-when-removed POS, ablate it
       from text, and re-run the prosody-vs-text-vs-combined comparison.
       If prosody starts winning vs the ablated text, that's the prosody
       channel demonstrating signal that text uniquely loses.

Uses spaCy en_core_web_sm for POS tagging. Tags are cached to disk.

Usage:
    .venv/bin/python scripts/pos_ablation.py \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-parametric data/meld/parametric_prosody_test_mfa.jsonl \\
        --eval-csv data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --label emotion
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.compare_prosody_text import (
    build_dataset, clean_utterance_text, make_prosody_probe,
)

warnings.filterwarnings("ignore")

# Universal POS tags spaCy emits
POS_TAGS = ("NOUN", "VERB", "ADJ", "ADV", "INTJ", "PRON", "PROPN",
            "AUX", "DET", "ADP", "CCONJ", "SCONJ", "NUM", "PART",
            "PUNCT", "SYM", "X")


# ---------------------------------------------------------------------------
# POS tagging (cached)
# ---------------------------------------------------------------------------

def pos_tag_corpus(texts: list[str], cache_path: Path | None) -> list[list[tuple[str, str]]]:
    """Return per-utterance list of (token_text, pos_tag).

    If cache_path is given and exists, loads from cache. Otherwise tags with
    spaCy and writes the cache.
    """
    if cache_path and cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        if len(cached) == len(texts):
            return [[(t, p) for t, p in row] for row in cached]
        else:
            print(f"  cache size mismatch ({len(cached)} vs {len(texts)}), regenerating")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    out: list[list[tuple[str, str]]] = []
    for doc in nlp.pipe(texts, batch_size=256):
        out.append([(tok.text, tok.pos_) for tok in doc])
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(out, f)
    return out


def tokens_of_pos(tagged: list[list[tuple[str, str]]], pos: str | None,
                  exclude: bool = False) -> list[str]:
    """Reconstruct text from tagged tokens, filtering by POS.

    If pos=None, returns the original (joined) text — full vocabulary.
    If exclude=False: keep only tokens of `pos`.
    If exclude=True:  drop tokens of `pos`, keep everything else.
    """
    out = []
    for sent in tagged:
        if pos is None:
            keep = sent
        elif exclude:
            keep = [(t, p) for t, p in sent if p != pos]
        else:
            keep = [(t, p) for t, p in sent if p == pos]
        out.append(" ".join(t for t, _ in keep))
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def make_text_probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b|[!?]+|\.{2,}",
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def evaluate_text(texts_tr: list[str], y_tr, texts_te: list[str], y_te) -> float:
    # Filter empty strings from training (TF-IDF can't fit on all-empty)
    valid = [(t, l) for t, l in zip(texts_tr, y_tr) if t.strip()]
    if not valid:
        return 0.0
    t_tr, y_tr2 = zip(*valid)
    clf = make_text_probe()
    try:
        clf.fit(list(t_tr), list(y_tr2))
        pred = clf.predict(texts_te)
        return float(f1_score(y_te, pred, average="macro", zero_division=0))
    except ValueError:
        return 0.0


def evaluate_prosody(X_tr, y_tr, X_te, y_te) -> float:
    clf = make_prosody_probe()
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_te, clf.predict(X_te), average="macro", zero_division=0))


def evaluate_combined(X_tr, texts_tr, y_tr, X_te, texts_te, y_te) -> float:
    valid = [(x, t, l) for x, t, l in zip(X_tr, texts_tr, y_tr) if t.strip()]
    if not valid:
        return 0.0
    Xt, tt, yt = zip(*valid)
    Xt = np.array(list(Xt))
    yt = np.array(list(yt))
    text_clf = make_text_probe()
    text_clf.fit(list(tt), yt)
    tfidf = text_clf.named_steps["tfidf"]
    txt_tr = tfidf.transform(list(tt)).toarray()
    txt_te = tfidf.transform(texts_te).toarray()
    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt)
    Xte_s = scaler.transform(X_te)
    Xc_tr = np.hstack([txt_tr, Xt_s])
    Xc_te = np.hstack([txt_te, Xte_s])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xc_tr, yt)
    return float(f1_score(y_te, clf.predict(Xc_te), average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--eval-parametric", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--label", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/meld/_pos_cache"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] {args.train_csv.name} → {args.eval_csv.name}, label={args.label}")
    Xtr, texts_tr, ytr = build_dataset(args.train_parametric, args.train_csv, args.label,
                                       "mean+max+std", clean_text=True)
    Xte, texts_te, yte = build_dataset(args.eval_parametric, args.eval_csv, args.label,
                                       "mean+max+std", clean_text=True)
    print(f"  train={len(ytr)}  eval={len(yte)}")

    # POS-tag both splits (cache per file basename + label)
    print("[pos] tagging train...")
    tr_tags = pos_tag_corpus(texts_tr, args.cache_dir / f"train_pos.json")
    print("[pos] tagging eval...")
    te_tags = pos_tag_corpus(texts_te, args.cache_dir / f"eval_pos.json")

    # Tag-frequency report
    counts: dict[str, int] = {}
    for sent in tr_tags:
        for _, p in sent:
            counts[p] = counts.get(p, 0) + 1
    total = sum(counts.values())
    print(f"\n[pos] train token POS distribution (top 10):")
    for p, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {p:6s}  {c:6d}  {100*c/total:5.1f}%")

    # Baselines
    print(f"\n[baseline] full text:   ", end="")
    f1_full = evaluate_text(texts_tr, ytr, texts_te, yte)
    print(f"macro_F1={f1_full:.4f}")
    print(f"[baseline] prosody only:", end="")
    f1_prosody = evaluate_prosody(Xtr, ytr, Xte, yte)
    print(f" macro_F1={f1_prosody:.4f}")
    print(f"[baseline] combined:    ", end="")
    f1_combined = evaluate_combined(Xtr, texts_tr, ytr, Xte, texts_te, yte)
    print(f" macro_F1={f1_combined:.4f}")

    # Per-POS results
    print(f"\n{'POS':<8s}  {'only-this-POS':>14s}  {'all-but-this':>14s}  {'Δ vs full':>10s}")
    print("-" * 55)
    per_pos: dict[str, dict] = {}
    for pos in POS_TAGS:
        if counts.get(pos, 0) < 50:  # too sparse
            continue
        only_tr = tokens_of_pos(tr_tags, pos, exclude=False)
        only_te = tokens_of_pos(te_tags, pos, exclude=False)
        excl_tr = tokens_of_pos(tr_tags, pos, exclude=True)
        excl_te = tokens_of_pos(te_tags, pos, exclude=True)

        f1_only = evaluate_text(only_tr, ytr, only_te, yte)
        f1_excl = evaluate_text(excl_tr, ytr, excl_te, yte)
        delta = f1_excl - f1_full  # how much performance drops when this POS is removed
        per_pos[pos] = {"only": f1_only, "excl": f1_excl, "delta_from_full": delta}
        print(f"{pos:<8s}  {f1_only:14.4f}  {f1_excl:14.4f}  {delta:+10.4f}")

    # Highlight: best individual POS, biggest ablation drop
    best_only = max(per_pos.items(), key=lambda kv: kv[1]["only"])
    biggest_drop = min(per_pos.items(), key=lambda kv: kv[1]["delta_from_full"])
    print()
    print(f"  most informative POS alone: {best_only[0]}  (only={best_only[1]['only']:.4f})")
    print(f"  POS whose removal hurts text most: {biggest_drop[0]}  "
          f"(text drops to {biggest_drop[1]['excl']:.4f}, Δ={biggest_drop[1]['delta_from_full']:+.4f})")

    # Now: prosody+combined evaluation when we ablate the worst-when-removed POS
    abl_pos = biggest_drop[0]
    print(f"\n=== Ablation: remove all {abl_pos} tokens from text, re-evaluate prosody contribution ===")
    abl_tr = tokens_of_pos(tr_tags, abl_pos, exclude=True)
    abl_te = tokens_of_pos(te_tags, abl_pos, exclude=True)
    f1_text_abl = evaluate_text(abl_tr, ytr, abl_te, yte)
    f1_combined_abl = evaluate_combined(Xtr, abl_tr, ytr, Xte, abl_te, yte)
    print(f"  text-only (no {abl_pos}):       macro_F1={f1_text_abl:.4f}  (was {f1_full:.4f})")
    print(f"  prosody-only (unchanged):       macro_F1={f1_prosody:.4f}")
    print(f"  combined (no-{abl_pos} text + prosody): macro_F1={f1_combined_abl:.4f}  (was {f1_combined:.4f})")
    print(f"  Δ combined vs ablated-text:     {f1_combined_abl - f1_text_abl:+.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "label": args.label,
                "baselines": {"text_full": f1_full, "prosody": f1_prosody, "combined": f1_combined},
                "per_pos": per_pos,
                "ablation_pos": abl_pos,
                "ablation": {
                    "text_no_pos": f1_text_abl,
                    "prosody": f1_prosody,
                    "combined": f1_combined_abl,
                    "delta_combined_minus_text": f1_combined_abl - f1_text_abl,
                },
            }, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
