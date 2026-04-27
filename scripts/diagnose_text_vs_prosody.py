"""
Per-class breakdown + text/prosody disagreement analysis on MELD.

Uses 5-fold cross_val_predict to get one prediction per utterance from each
probe (text-only, prosody-only, combined), then reports:

    1. Per-class precision/recall/F1 for each probe — answers "does text or
       prosody do better on the neutral class?".
    2. Pairwise disagreement: where text predicts X and prosody predicts Y,
       what was the true label and what does that tell us?
    3. Sample utterances at the most informative disagreement cells.

Usage:
    .venv/bin/python scripts/diagnose_text_vs_prosody.py \\
        --parametric data/meld/parametric_prosody_dev_mfa.jsonl \\
        --csv data/meld/MELD.Raw/dev_sent_emo.csv \\
        --label emotion
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse loaders from the comparison script
from scripts.compare_prosody_text import build_dataset, make_text_probe, make_prosody_probe

warnings.filterwarnings("ignore")


def cv_predict_text_prosody_combined(
    X_pros: np.ndarray, texts: list[str], y: np.ndarray, n_splits: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-fold CV: return per-utterance (text_pred, prosody_pred, combined_pred) arrays."""
    text_pred = np.empty_like(y)
    pros_pred = np.empty_like(y)
    comb_pred = np.empty_like(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(X_pros, y):
        X_tr, X_te = X_pros[train_idx], X_pros[test_idx]
        t_tr = [texts[i] for i in train_idx]
        t_te = [texts[i] for i in test_idx]
        y_tr = y[train_idx]

        text_clf = make_text_probe()
        text_clf.fit(t_tr, y_tr)
        text_pred[test_idx] = text_clf.predict(t_te)

        pros_clf = make_prosody_probe()
        pros_clf.fit(X_tr, y_tr)
        pros_pred[test_idx] = pros_clf.predict(X_te)

        # Combined: TF-IDF features + scaled prosody, then LR
        tfidf = text_clf.named_steps["tfidf"]
        txt_tr_feat = tfidf.transform(t_tr).toarray()
        txt_te_feat = tfidf.transform(t_te).toarray()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        Xc_tr = np.hstack([txt_tr_feat, X_tr_s])
        Xc_te = np.hstack([txt_te_feat, X_te_s])
        comb_clf = LogisticRegression(max_iter=2000, C=1.0)
        comb_clf.fit(Xc_tr, y_tr)
        comb_pred[test_idx] = comb_clf.predict(Xc_te)
    return text_pred, pros_pred, comb_pred


def per_class_table(y_true, y_pred, labels) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for lbl in labels:
        d = rep[lbl]
        rows.append({"class": lbl, "precision": d["precision"], "recall": d["recall"],
                     "f1": d["f1-score"], "support": int(d["support"])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parametric", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--label", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-disagreement-examples", type=int, default=5)
    args = parser.parse_args()

    X, texts, y = build_dataset(args.parametric, args.csv, args.label, "mean+max+std")
    print(f"[load] {len(y)} utterances, {len(set(y))} classes\n")

    text_pred, pros_pred, comb_pred = cv_predict_text_prosody_combined(
        X, texts, y, args.cv, args.seed,
    )

    classes = sorted(set(y))

    # Per-class breakdown
    print("=" * 78)
    print("Per-class precision / recall / F1")
    print("=" * 78)
    text_tbl = per_class_table(y, text_pred, classes)
    pros_tbl = per_class_table(y, pros_pred, classes)
    comb_tbl = per_class_table(y, comb_pred, classes)
    merged = text_tbl[["class", "support"]].copy()
    for name, tbl in [("text", text_tbl), ("prosody", pros_tbl), ("combined", comb_tbl)]:
        merged[f"{name}_P"] = tbl["precision"].round(3)
        merged[f"{name}_R"] = tbl["recall"].round(3)
        merged[f"{name}_F1"] = tbl["f1"].round(3)
    # column order
    cols = ["class", "support",
            "text_P", "text_R", "text_F1",
            "prosody_P", "prosody_R", "prosody_F1",
            "combined_P", "combined_R", "combined_F1"]
    print(merged[cols].to_string(index=False))

    # Class-by-class headlines
    print()
    print("Headlines:")
    for c in classes:
        t_f1 = float(text_tbl.loc[text_tbl["class"] == c, "f1"].iloc[0])
        p_f1 = float(pros_tbl.loc[pros_tbl["class"] == c, "f1"].iloc[0])
        delta = p_f1 - t_f1
        winner = "PROSODY" if delta > 0.02 else ("TEXT" if delta < -0.02 else "tie")
        print(f"  {c:<12s}  text_F1={t_f1:.3f}  prosody_F1={p_f1:.3f}  Δ={delta:+.3f}  → {winner}")

    # Disagreement analysis
    print()
    print("=" * 78)
    print("Where text and prosody disagree — confusion of (text_pred, prosody_pred)")
    print("=" * 78)
    disagree = text_pred != pros_pred
    n_dis = int(disagree.sum())
    print(f"  {n_dis} of {len(y)} utterances disagree ({100*n_dis/len(y):.1f}%)")
    print()
    # crosstab: rows = text predicted, cols = prosody predicted, value = count
    ct = pd.crosstab(
        pd.Series(text_pred, name="text→"),
        pd.Series(pros_pred, name="↓prosody"),
    )
    ct = ct.reindex(index=classes, columns=classes, fill_value=0)
    print(ct.to_string())
    print()
    print(f"  Diagonal = agreement.  Off-diagonal = disagreement.")
    print(f"  Highest disagreement cells (text→ ≠ ↓prosody):")
    triu_mask = ~np.eye(len(classes), dtype=bool)
    flat = []
    for i, ti in enumerate(classes):
        for j, pj in enumerate(classes):
            if i != j:
                flat.append((ti, pj, int(ct.iloc[i, j])))
    flat.sort(key=lambda r: r[2], reverse=True)
    for ti, pj, cnt in flat[:8]:
        if cnt == 0:
            break
        print(f"    text→{ti:<10s}  prosody→{pj:<10s}  count={cnt}")

    # When text and prosody disagree, who tends to be right?
    print()
    print("=" * 78)
    print("When text + prosody disagree, who's right?")
    print("=" * 78)
    text_right = (text_pred == y) & (pros_pred != y)
    pros_right = (pros_pred == y) & (text_pred != y)
    both_wrong = (text_pred != y) & (pros_pred != y) & disagree
    print(f"  text right, prosody wrong:  {int(text_right.sum())}")
    print(f"  prosody right, text wrong:  {int(pros_right.sum())}")
    print(f"  both wrong (disagree):      {int(both_wrong.sum())}")

    # Sample utterances where each is uniquely right
    print()
    print(f"Examples where PROSODY is uniquely right (text wrong):")
    idxs = np.flatnonzero(pros_right)[: args.n_disagreement_examples]
    for i in idxs:
        print(f"    true={y[i]:<10s}  text→{text_pred[i]:<10s}  prosody→{pros_pred[i]:<10s}  | {texts[i][:100]!r}")

    print()
    print(f"Examples where TEXT is uniquely right (prosody wrong):")
    idxs = np.flatnonzero(text_right)[: args.n_disagreement_examples]
    for i in idxs:
        print(f"    true={y[i]:<10s}  text→{text_pred[i]:<10s}  prosody→{pros_pred[i]:<10s}  | {texts[i][:100]!r}")


if __name__ == "__main__":
    main()
