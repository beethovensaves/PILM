"""
Sanity check: can the 18-dim parametric vector alone predict whether an
utterance is a question (i.e., the original transcript ends with "?")?

If prosody encodes speech-act information at all, this should be one of
the easiest tasks for it. Yes/no questions in English have a robust
final-rise signature (high f0_offset, late peak, positive slope at the
last syllable). Wh-questions are subtler.

We label an utterance as a "question" if its raw Utterance text contains
a "?" anywhere (matches both wh- and yes/no). We use the *raw* utterance
text from the cleaned CSVs (which still preserves "?" — overpunctuation
collapse turns "?!" into "?", but "?" itself is kept).

Pipeline:
    train: MELD train split parametric vectors + has_question labels
    eval:  test split parametric vectors + has_question labels
    probes:
        - majority baseline (DummyClassifier)
        - chance baseline (stratified random)
        - LR over pooled 18-dim parametric vector
        - LR over LAST-syllable parametric vector only (the rise lives there)

Reports accuracy, macro-F1, ROC-AUC, and a per-feature look at which dims
are most predictive of a question.

Usage:
    .venv/bin/python scripts/predict_question_from_prosody.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.compare_prosody_text import load_parametric, pool_to_utterance

warnings.filterwarnings("ignore")

DIM_LABELS = (
    "f0_onset_st", "f0_nucleus_st", "f0_offset_st",
    "f0_max_st", "f0_min_st", "f0_range_st", "f0_slope_st_per_ms",
    "f0_peak_pos", "f0_rise_amp", "f0_fall_amp", "tilt",
    "rms_max_z", "rms_mean_z",
    "syl_dur_z", "nuc_dur_z",
    "pause_after_ms", "final_lengthen", "f0_reset_st",
)


WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}


def is_yesno_question(text: str) -> bool:
    """A utterance is a yes/no question if it contains '?' AND its first
    alphabetic word is NOT a wh-word."""
    if "?" not in text:
        return False
    # find first alphanumeric word
    import re as _re
    m = _re.search(r"[A-Za-z]+", text)
    if not m:
        return False
    return m.group(0).lower() not in WH_WORDS


def build(
    parametric_path: Path, csv_path: Path, yn_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (X_pooled, X_last_syl, y, texts).

    If yn_only=True: positives are yes/no questions only. Wh-questions
    (utterances with '?' that start with a wh-word) are DROPPED entirely
    (neither positive nor negative) to avoid contamination.
    """
    parametric = load_parametric(parametric_path)
    df = pd.read_csv(csv_path)
    df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    df = df[df["utterance_id"].isin(parametric)].copy()

    import re as _re
    X_pool: list[np.ndarray] = []
    X_last: list[np.ndarray] = []
    y: list[int] = []
    texts: list[str] = []
    n_skipped_wh = 0
    for row in df.itertuples():
        utt = str(row.Utterance)
        has_q = "?" in utt
        m = _re.search(r"[A-Za-z]+", utt)
        first_word = m.group(0).lower() if m else ""
        is_wh_q = has_q and first_word in WH_WORDS

        if yn_only and is_wh_q:
            n_skipped_wh += 1
            continue  # drop wh-questions entirely

        vecs = parametric[row.utterance_id]
        pooled = pool_to_utterance(vecs, "mean+max+std")
        if np.all(np.isnan(pooled)):
            continue
        last = vecs[-1]
        X_pool.append(pooled)
        X_last.append(last)
        if yn_only:
            y.append(1 if (has_q and not is_wh_q) else 0)
        else:
            y.append(1 if has_q else 0)
        texts.append(utt)
    if yn_only and n_skipped_wh:
        print(f"  (dropped {n_skipped_wh} wh-questions)")

    X_pool_arr = np.vstack(X_pool)
    X_last_arr = np.vstack(X_last)
    # NaN imputation — column means
    for arr in (X_pool_arr, X_last_arr):
        cm = np.nanmean(arr, axis=0)
        ind = np.where(np.isnan(arr))
        arr[ind] = np.take(cm, ind[1])
    return X_pool_arr, X_last_arr, np.array(y), texts


def report(name: str, y_true, y_pred, y_proba=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1p = f1_score(y_true, y_pred, pos_label=1)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    print(f"  {name:<28s}  acc={acc:.4f}  macro_F1={f1m:.4f}  question_F1={f1p:.4f}  AUC={auc:.4f}")
    return {"accuracy": acc, "macro_f1": f1m, "question_f1": f1p, "auc": auc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path,
                        default=Path("data/meld/parametric_prosody_train_mfa.jsonl"))
    parser.add_argument("--train-csv", type=Path,
                        default=Path("data/meld/MELD.Raw/train_sent_emo_cleaned.csv"))
    parser.add_argument("--eval-parametric", type=Path,
                        default=Path("data/meld/parametric_prosody_test_mfa.jsonl"))
    parser.add_argument("--eval-csv", type=Path,
                        default=Path("data/meld/MELD.Raw/test_sent_emo_cleaned.csv"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--yn-only", action="store_true",
                        help="Restrict positive class to yes/no questions only "
                             "(drop wh-questions entirely)")
    args = parser.parse_args()

    print(f"[load] train={args.train_parametric.name} eval={args.eval_parametric.name}"
          + ("  (yes/no-only)" if args.yn_only else ""))
    Xtr_p, Xtr_l, ytr, texts_tr = build(args.train_parametric, args.train_csv, yn_only=args.yn_only)
    Xte_p, Xte_l, yte, texts_te = build(args.eval_parametric, args.eval_csv, yn_only=args.yn_only)
    n_tr_q = int(ytr.sum())
    n_te_q = int(yte.sum())
    print(f"  train: {len(ytr)} utts, {n_tr_q} questions ({100*n_tr_q/len(ytr):.1f}%)")
    print(f"  eval:  {len(yte)} utts, {n_te_q} questions ({100*n_te_q/len(yte):.1f}%)")
    print()

    print("[probes — train→test, predict whether utterance contains '?']")

    mb = DummyClassifier(strategy="most_frequent")
    mb.fit(Xtr_p, ytr)
    report("majority baseline (always 0)", yte, mb.predict(Xte_p),
           mb.predict_proba(Xte_p)[:, 1])

    sb = DummyClassifier(strategy="stratified", random_state=42)
    sb.fit(Xtr_p, ytr)
    report("stratified random", yte, sb.predict(Xte_p),
           sb.predict_proba(Xte_p)[:, 1])

    pip_pooled = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1.0))])
    pip_pooled.fit(Xtr_p, ytr)
    pred_pool = pip_pooled.predict(Xte_p)
    proba_pool = pip_pooled.predict_proba(Xte_p)[:, 1]
    res_pool = report("LR on pooled (54 dims)", yte, pred_pool, proba_pool)

    pip_last = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1.0))])
    pip_last.fit(Xtr_l, ytr)
    pred_last = pip_last.predict(Xte_l)
    proba_last = pip_last.predict_proba(Xte_l)[:, 1]
    res_last = report("LR on last-syllable (18)", yte, pred_last, proba_last)

    # Top features in last-syllable model — these tell us which dims matter for "question"
    coef = pip_last.named_steps["c"].coef_[0]
    print()
    print("[last-syllable LR coefficients — most predictive of '?']")
    print("  positive coef → dim higher when utterance is a question")
    ranked = sorted(enumerate(coef), key=lambda kv: abs(kv[1]), reverse=True)
    for i, c in ranked[:8]:
        sign = "+" if c > 0 else "−"
        print(f"    {sign}{abs(c):.3f}  {DIM_LABELS[i]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "train_questions": n_tr_q, "eval_questions": n_te_q,
                "lr_pooled": res_pool, "lr_last_syllable": res_last,
                "last_syllable_top_coef": [
                    {"dim": DIM_LABELS[i], "coef": float(coef[i])}
                    for i, _ in ranked[:18]
                ],
            }, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
