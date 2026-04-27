"""
Supervised classification comparison: prosody-only vs text-only vs combined.

Trains and evaluates probes on MELD utterances under three feature regimes
and asks: does the parametric prosody channel (D19 18-dim per syllable,
pooled to utterance via mean+max+std) add signal beyond what text alone
provides? This is the small-scale precursor to the Phase 5 killer experiment.

Three feature setups:
    1. TEXT     — TF-IDF over the utterance string + LogReg
    2. PROSODY  — pooled parametric vectors + LogReg (same as validate_parametric_prosody.py)
    3. COMBINED — concat(text TF-IDF features, pooled parametric vector) + LogReg

Two label tasks: `emotion` (7-way) and `sentiment` (3-way).

Modes:
    --mode cv         5-fold stratified CV on a single split (default)
    --mode train_eval Train on --train-* args, evaluate on --eval-* args

Usage:
    # Cross-validation on dev only (prototype before train/test land):
    .venv/bin/python scripts/compare_prosody_text.py \\
        --parametric data/meld/parametric_prosody_dev_mfa.jsonl \\
        --csv        data/meld/MELD.Raw/dev_sent_emo.csv \\
        --label      emotion

    # Train on train, evaluate on dev (or test):
    .venv/bin/python scripts/compare_prosody_text.py \\
        --mode train_eval \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv        data/meld/MELD.Raw/train_sent_emo.csv \\
        --eval-parametric  data/meld/parametric_prosody_dev_mfa.jsonl \\
        --eval-csv         data/meld/MELD.Raw/dev_sent_emo.csv \\
        --label emotion
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text cleaning — removes overpunctuation that leaks emotion into text features
# ---------------------------------------------------------------------------
# MELD transcripts are full of typographic emphasis ("Where?!", "Yeah!!!",
# "Really?!"), which is exactly the prosodic signal we *don't* want text to
# get for free. We collapse:
#   - "?!" / "!?" / "?!?!" → "?"   (mixed = question dominates)
#   - "!!" / "!!!"          → "!"
#   - "??" / "???"          → "?"
#   - "...." (4+ dots)      → "..."  (canonical ellipsis)
# Plus: normalize Windows-1252 / Unicode smart quotes that appear in MELD.

_SMART_QUOTES = str.maketrans({
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "\x91": "'", "\x92": "'",
    "\x93": '"', "\x94": '"',
})

_PUNCT_RUN = re.compile(r"([!?]+|\.{2,})")


def _collapse_run(m: re.Match) -> str:
    run = m.group(0)
    if run.startswith("."):
        return "..."
    return "?" if "?" in run else "!"


def clean_utterance_text(s: str) -> str:
    """Strip emotion-leaking overpunctuation. Use BEFORE TF-IDF."""
    s = str(s).translate(_SMART_QUOTES)
    s = _PUNCT_RUN.sub(_collapse_run, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_parametric(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            vecs = []
            for s in d["syllables"]:
                vecs.append([x if x is not None else float("nan") for x in s["vec"]])
            if vecs:
                out[d["utterance_id"]] = np.array(vecs, dtype=np.float64)
    return out


def pool_to_utterance(vecs: np.ndarray, mode: str = "mean+max+std") -> np.ndarray:
    pieces = []
    for op in mode.split("+"):
        if op == "mean":
            pieces.append(np.nanmean(vecs, axis=0))
        elif op == "max":
            pieces.append(np.nanmax(vecs, axis=0))
        elif op == "std":
            pieces.append(np.nanstd(vecs, axis=0))
        else:
            raise ValueError(f"unknown pool op: {op}")
    return np.concatenate(pieces)


def build_dataset(
    parametric_path: Path,
    csv_path: Path,
    label_key: str,
    pool_mode: str = "mean+max+std",
    clean_text: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Returns (X_prosody, texts, y) aligned on utterance_id.

    By default applies overpunctuation cleaning to the text (set clean_text=False
    to disable for an unfair-advantage comparison).
    """
    parametric = load_parametric(parametric_path)
    df = pd.read_csv(csv_path)
    df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    label_col = "Emotion" if label_key == "emotion" else "Sentiment"
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    X: list[np.ndarray] = []
    texts: list[str] = []
    y: list[str] = []
    for row in df.itertuples():
        if row.utterance_id not in parametric:
            continue
        vec = pool_to_utterance(parametric[row.utterance_id], pool_mode)
        if np.all(np.isnan(vec)):
            continue
        X.append(vec)
        utt = str(row.Utterance)
        texts.append(clean_utterance_text(utt) if clean_text else utt)
        y.append(getattr(row, label_col.title()))

    X_arr = np.vstack(X)
    col_means = np.nanmean(X_arr, axis=0)
    inds = np.where(np.isnan(X_arr))
    X_arr[inds] = np.take(col_means, inds[1])
    return X_arr, texts, np.array(y)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def make_text_probe() -> Pipeline:
    # token_pattern includes punctuation runs (?, !, ...) so the cleaning step
    # actually matters. Default sklearn pattern would strip them entirely.
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


def make_prosody_probe() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=2000, C=1.0)),
    ])


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

def cv_eval(model, X, y, cv) -> dict[str, float]:
    macro_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", error_score="raise").mean()
    accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy", error_score="raise").mean()
    return {"macro_f1": float(macro_f1), "accuracy": float(accuracy)}


def run_cv(
    X_prosody: np.ndarray,
    texts: list[str],
    y: np.ndarray,
    cv: StratifiedKFold,
) -> dict[str, dict[str, float]]:
    """Five-fold CV on a single split, three feature regimes."""
    # Combined: we have to do it manually because CV fold-aware TF-IDF + numeric concat is awkward
    # in pure sklearn; instead, fit TF-IDF on each train fold inside a FunctionTransformer-style flow.
    # The simplest correct path: precompute TF-IDF inside a per-fold loop.

    text_results: list[dict] = []
    prosody_results: list[dict] = []
    combined_results: list[dict] = []
    majority_results: list[dict] = []

    text_arr = np.array(texts)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_prosody, y)):
        X_pros_tr, X_pros_te = X_prosody[train_idx], X_prosody[test_idx]
        text_tr, text_te = text_arr[train_idx].tolist(), text_arr[test_idx].tolist()
        y_tr, y_te = y[train_idx], y[test_idx]

        # Text-only
        text_clf = make_text_probe()
        text_clf.fit(text_tr, y_tr)
        text_pred = text_clf.predict(text_te)
        text_results.append({
            "macro_f1": f1_score(y_te, text_pred, average="macro", zero_division=0),
        })

        # Prosody-only
        pros_clf = make_prosody_probe()
        pros_clf.fit(X_pros_tr, y_tr)
        pros_pred = pros_clf.predict(X_pros_te)
        prosody_results.append({
            "macro_f1": f1_score(y_te, pros_pred, average="macro", zero_division=0),
        })

        # Combined: stack TF-IDF (sparse → dense) with prosody
        tfidf = text_clf.named_steps["tfidf"]
        text_tr_feat = tfidf.transform(text_tr).toarray()
        text_te_feat = tfidf.transform(text_te).toarray()
        scaler = StandardScaler()
        X_pros_tr_s = scaler.fit_transform(X_pros_tr)
        X_pros_te_s = scaler.transform(X_pros_te)
        X_comb_tr = np.hstack([text_tr_feat, X_pros_tr_s])
        X_comb_te = np.hstack([text_te_feat, X_pros_te_s])
        comb_clf = LogisticRegression(max_iter=2000, C=1.0)
        comb_clf.fit(X_comb_tr, y_tr)
        comb_pred = comb_clf.predict(X_comb_te)
        combined_results.append({
            "macro_f1": f1_score(y_te, comb_pred, average="macro", zero_division=0),
        })

        # Majority baseline
        mb = DummyClassifier(strategy="most_frequent")
        mb.fit(X_pros_tr, y_tr)
        mb_pred = mb.predict(X_pros_te)
        majority_results.append({
            "macro_f1": f1_score(y_te, mb_pred, average="macro", zero_division=0),
        })

    def avg(rs):
        return {"macro_f1": float(np.mean([r["macro_f1"] for r in rs])),
                "fold_f1s": [round(r["macro_f1"], 4) for r in rs]}

    return {
        "majority_baseline": avg(majority_results),
        "text_only":         avg(text_results),
        "prosody_only":      avg(prosody_results),
        "combined":          avg(combined_results),
    }


def run_train_eval(
    X_pros_tr: np.ndarray, texts_tr: list[str], y_tr: np.ndarray,
    X_pros_te: np.ndarray, texts_te: list[str], y_te: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Train on one split, evaluate on another. Returns macro-F1 per regime."""
    out = {}

    mb = DummyClassifier(strategy="most_frequent")
    mb.fit(X_pros_tr, y_tr)
    out["majority_baseline"] = {"macro_f1": float(f1_score(y_te, mb.predict(X_pros_te), average="macro", zero_division=0))}

    text_clf = make_text_probe()
    text_clf.fit(texts_tr, y_tr)
    out["text_only"] = {"macro_f1": float(f1_score(y_te, text_clf.predict(texts_te), average="macro", zero_division=0))}

    pros_clf = make_prosody_probe()
    pros_clf.fit(X_pros_tr, y_tr)
    out["prosody_only"] = {"macro_f1": float(f1_score(y_te, pros_clf.predict(X_pros_te), average="macro", zero_division=0))}

    tfidf = text_clf.named_steps["tfidf"]
    txt_tr = tfidf.transform(texts_tr).toarray()
    txt_te = tfidf.transform(texts_te).toarray()
    scaler = StandardScaler()
    pros_tr_s = scaler.fit_transform(X_pros_tr)
    pros_te_s = scaler.transform(X_pros_te)
    X_comb_tr = np.hstack([txt_tr, pros_tr_s])
    X_comb_te = np.hstack([txt_te, pros_te_s])
    comb_clf = LogisticRegression(max_iter=2000, C=1.0)
    comb_clf.fit(X_comb_tr, y_tr)
    out["combined"] = {"macro_f1": float(f1_score(y_te, comb_clf.predict(X_comb_te), average="macro", zero_division=0))}

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "train_eval"], default="cv")
    parser.add_argument("--label", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--pool-mode", default="mean+max+std")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--parametric", type=Path,
                        help="(cv mode) Single parametric JSONL")
    parser.add_argument("--csv", type=Path,
                        help="(cv mode) Single MELD CSV")

    parser.add_argument("--train-parametric", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--eval-parametric", type=Path)
    parser.add_argument("--eval-csv", type=Path)

    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-clean-text", action="store_true",
                        help="Disable overpunctuation cleaning (gives text the leaked-emotion advantage).")
    args = parser.parse_args()
    clean_text = not args.no_clean_text

    if args.mode == "cv":
        if not (args.parametric and args.csv):
            raise SystemExit("cv mode requires --parametric and --csv")
        X, texts, y = build_dataset(args.parametric, args.csv, args.label, args.pool_mode, clean_text=clean_text)
        print(f"[load] {len(y)} utterances, {len(set(y))} classes")
        print(f"       class counts: {dict(pd.Series(y).value_counts())}")
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        results = run_cv(X, texts, y, cv)
    else:
        if not (args.train_parametric and args.train_csv and args.eval_parametric and args.eval_csv):
            raise SystemExit("train_eval mode requires all --train-* and --eval-* args")
        Xtr, texts_tr, ytr = build_dataset(args.train_parametric, args.train_csv, args.label, args.pool_mode, clean_text=clean_text)
        Xte, texts_te, yte = build_dataset(args.eval_parametric, args.eval_csv, args.label, args.pool_mode, clean_text=clean_text)
        print(f"[load] train: {len(ytr)} utterances")
        print(f"[load] eval:  {len(yte)} utterances")
        results = run_train_eval(Xtr, texts_tr, ytr, Xte, texts_te, yte)

    print(f"\n[task] {args.label}  ({args.mode})")
    print(f"{'regime':<22s}  {'macro_F1':>10s}  Δ vs majority")
    print("-" * 60)
    bl = results["majority_baseline"]["macro_f1"]
    for name in ("majority_baseline", "text_only", "prosody_only", "combined"):
        f1 = results[name]["macro_f1"]
        delta = f1 - bl
        marker = ""
        if name == "prosody_only" and delta >= 0.05:
            marker = "  ← prosody beats baseline"
        if name == "combined" and f1 > results["text_only"]["macro_f1"] + 0.005:
            marker = "  ← prosody adds to text"
        print(f"{name:<22s}  {f1:10.4f}  {delta:+.4f}{marker}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({"label": args.label, "mode": args.mode, "results": results}, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
