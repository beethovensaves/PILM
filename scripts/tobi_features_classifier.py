"""
Use rule-based ToBI labels (from parametric_to_tobi.py) as utterance-level
features and compare to text + parametric prosody for emotion/sentiment.

Per utterance, build a fixed-size feature vector from the ToBI label
distribution:
    - 6 accent counts (NONE, H*, L*, L+H*, L*+H, H+!H*) — normalized to fractions
    - 3 break-index counts (1, 3, 4) — normalized
    - 3 boundary-tone counts (NONE, H%, L%) — normalized
    - 1 syllable count (raw)
    - 1 has_terminal_H% indicator (1 if final syllable has H%)
    - 1 has_terminal_L% indicator
    Total: 15 dims.

Compares 4 regimes train→test:
    - text-only (TF-IDF + LR)
    - parametric (54-dim pooled — same as compare_prosody_text.py)
    - tobi-categorical (15-dim from above)
    - text + tobi-categorical (concat)

Usage:
    .venv/bin/python scripts/tobi_features_classifier.py \\
        --train-tobi data/meld/parametric_prosody_train_tobi.jsonl \\
        --train-csv  data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-tobi  data/meld/parametric_prosody_test_tobi.jsonl \\
        --eval-csv   data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --label emotion
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.compare_prosody_text import clean_utterance_text

warnings.filterwarnings("ignore")

ACCENTS  = ("NONE", "H*", "L*", "L+H*", "L*+H", "H+!H*")
BREAKS   = (1, 3, 4)
BTONES   = ("NONE", "H%", "L%")
N_FEATS  = len(ACCENTS) + len(BREAKS) + len(BTONES) + 3  # = 15


def utterance_tobi_features(syllables: list[dict]) -> np.ndarray:
    """Build a 15-dim summary of ToBI labels in this utterance."""
    n = len(syllables)
    if n == 0:
        return np.zeros(N_FEATS)
    accent_counts = {a: 0 for a in ACCENTS}
    break_counts  = {b: 0 for b in BREAKS}
    btone_counts  = {t: 0 for t in BTONES}
    for s in syllables:
        accent_counts[s.get("tobi_accent", "NONE")] = accent_counts.get(s.get("tobi_accent", "NONE"), 0) + 1
        bi = s.get("tobi_break_index", 1)
        break_counts[bi] = break_counts.get(bi, 0) + 1
        btone_counts[s.get("tobi_boundary_tone", "NONE")] = btone_counts.get(s.get("tobi_boundary_tone", "NONE"), 0) + 1
    last = syllables[-1]
    feats = []
    for a in ACCENTS:
        feats.append(accent_counts.get(a, 0) / n)
    for b in BREAKS:
        feats.append(break_counts.get(b, 0) / n)
    for t in BTONES:
        feats.append(btone_counts.get(t, 0) / n)
    feats.append(n)  # raw syllable count
    feats.append(1.0 if last.get("tobi_boundary_tone") == "H%" else 0.0)
    feats.append(1.0 if last.get("tobi_boundary_tone") == "L%" else 0.0)
    return np.array(feats, dtype=np.float32)


def load_tobi(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["utterance_id"]] = utterance_tobi_features(d["syllables"])
    return out


def build_xy(tobi_path: Path, csv_path: Path, label_col: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    tobi = load_tobi(tobi_path)
    df = pd.read_csv(csv_path)
    df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    df = df[df["utterance_id"].isin(tobi)].copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    X = np.vstack([tobi[uid] for uid in df["utterance_id"]])
    texts = [clean_utterance_text(t) for t in df["Utterance"].astype(str)]
    y = df[label_col].values
    return X, texts, y


def make_text_probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, lowercase=True,
            token_pattern=r"(?u)\b\w+\b|[!?]+|\.{2,}",
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def evaluate_all(Xtr_t, texts_tr, ytr, Xte_t, texts_te, yte) -> dict:
    out: dict[str, float] = {}

    # Text-only
    txt = make_text_probe()
    txt.fit(texts_tr, ytr)
    out["text_only"] = float(f1_score(yte, txt.predict(texts_te), average="macro", zero_division=0))

    # ToBI-only
    tobi_clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    tobi_clf.fit(Xtr_t, ytr)
    out["tobi_only"] = float(f1_score(yte, tobi_clf.predict(Xte_t), average="macro", zero_division=0))

    # Combined: TF-IDF + ToBI (both standardized)
    tfidf = txt.named_steps["tfidf"]
    txt_tr = tfidf.transform(texts_tr).toarray()
    txt_te = tfidf.transform(texts_te).toarray()
    sc = StandardScaler()
    Xt_tr_s = sc.fit_transform(Xtr_t)
    Xt_te_s = sc.transform(Xte_t)
    Xc_tr = np.hstack([txt_tr, Xt_tr_s])
    Xc_te = np.hstack([txt_te, Xt_te_s])
    comb = LogisticRegression(max_iter=2000)
    comb.fit(Xc_tr, ytr)
    out["text_plus_tobi"] = float(f1_score(yte, comb.predict(Xc_te), average="macro", zero_division=0))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tobi", type=Path, required=True)
    parser.add_argument("--train-csv",  type=Path, required=True)
    parser.add_argument("--eval-tobi",  type=Path, required=True)
    parser.add_argument("--eval-csv",   type=Path, required=True)
    parser.add_argument("--label", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    label_col = "Emotion" if args.label == "emotion" else "Sentiment"
    Xtr, texts_tr, ytr = build_xy(args.train_tobi, args.train_csv, label_col)
    Xte, texts_te, yte = build_xy(args.eval_tobi,  args.eval_csv,  label_col)
    print(f"[load] train={len(ytr)} eval={len(yte)} ToBI feature dim={Xtr.shape[1]}")
    print(f"[load] eval class dist: {dict(pd.Series(yte).value_counts())}")

    res = evaluate_all(Xtr, texts_tr, ytr, Xte, texts_te, yte)

    print(f"\n[{args.label} train→test, macro-F1]")
    print(f"  text_only      : {res['text_only']:.4f}")
    print(f"  tobi_only (15D): {res['tobi_only']:.4f}")
    print(f"  text + tobi    : {res['text_plus_tobi']:.4f}  "
          f"(Δ vs text: {res['text_plus_tobi'] - res['text_only']:+.4f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({"label": args.label, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
