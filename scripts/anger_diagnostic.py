"""
EXP-009 — Anger per-dim diagnostic.

The Phase 1.5 per-class result (`diagnose_text_vs_prosody.py`) showed prosody
beats text on anger by +0.140 F1. This script asks: which of the 18 parametric
dims carry that signal, and how concentrated is it?

Three views, all on the prosody-only feature side, treating anger as a binary
target (anger = 1, all other emotions = 0):

    1. Per-dim ANOVA F-statistic. Univariate ranking of how each dim alone
       separates anger from non-anger at the *syllable* level. High F = the
       dim's distribution shifts when the utterance is angry.

    2. Per-dim univariate AUC at the *utterance* level (pooled mean+max+std).
       Each dim independently fed to LR; AUC tells us how much of the angry-
       vs-not signal that dim alone carries.

    3. Single-dim-removed LR ablation. Drop each dim from the pooled feature
       set, retrain prosody-only LR, measure F1 drop on anger. Identifies
       load-bearing dims (large drop) vs redundant dims (no drop).

Reads MELD train + test parametric JSONLs and CSVs. Outputs a printed table +
optional JSON.

Usage:
    .venv/bin/python scripts/anger_diagnostic.py \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv        data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-parametric  data/meld/parametric_prosody_test_mfa.jsonl \\
        --eval-csv         data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --out              data/meld/anger_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
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


def build_anger(parametric_path: Path, csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_pooled[N,54], X_syl[M,18], y_utt[N], y_syl[M]).

    Utterance-level: one row per utterance, label 1 if Emotion=='anger'.
    Syllable-level: one row per syllable from those same utterances; the
    syllable inherits its utterance's anger label.
    """
    parametric = load_parametric(parametric_path)
    df = pd.read_csv(csv_path)
    df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    df = df[df["utterance_id"].isin(parametric)].copy()
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()

    X_pool = []
    y_utt = []
    X_syl_rows = []
    y_syl_rows = []
    for row in df.itertuples():
        is_anger = 1 if row.Emotion == "anger" else 0
        vecs = parametric[row.utterance_id]
        pooled = pool_to_utterance(vecs, "mean+max+std")
        if np.all(np.isnan(pooled)):
            continue
        X_pool.append(pooled)
        y_utt.append(is_anger)
        for v in vecs:
            X_syl_rows.append(v)
            y_syl_rows.append(is_anger)
    X_pool = np.vstack(X_pool)
    X_syl = np.vstack(X_syl_rows)
    y_utt = np.array(y_utt)
    y_syl = np.array(y_syl_rows)

    # NaN imputation — column means
    for arr in (X_pool, X_syl):
        cm = np.nanmean(arr, axis=0)
        ind = np.where(np.isnan(arr))
        arr[ind] = np.take(cm, ind[1])
    return X_pool, X_syl, y_utt, y_syl


def per_dim_anova(X_syl: np.ndarray, y_syl: np.ndarray) -> list[tuple[str, float, float]]:
    """ANOVA F-stat for each of the 18 dims at syllable level."""
    f_stat, p_val = f_classif(X_syl, y_syl)
    rows = [(DIM_LABELS[i], float(f_stat[i]), float(p_val[i])) for i in range(18)]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def per_dim_univariate_auc(X_pool_tr: np.ndarray, y_tr: np.ndarray,
                            X_pool_te: np.ndarray, y_te: np.ndarray) -> list[tuple[str, float]]:
    """Per dim (averaged over its mean/max/std slots), single-feature LR AUC for anger."""
    rows = []
    for i in range(18):
        # mean+max+std means dim i appears at offsets i, 18+i, 36+i
        cols = [i, 18 + i, 36 + i]
        Xtr = X_pool_tr[:, cols]
        Xte = X_pool_te[:, cols]
        pip = Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=2000, C=1.0))])
        pip.fit(Xtr, y_tr)
        proba = pip.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_te, proba)
        rows.append((DIM_LABELS[i], float(auc)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def single_dim_removed_ablation(X_pool_tr: np.ndarray, y_tr: np.ndarray,
                                  X_pool_te: np.ndarray, y_te: np.ndarray) -> tuple[float, list[tuple[str, float, float]]]:
    """Drop each dim's mean/max/std slots → retrain LR → measure anger-F1 drop."""
    # Baseline: full feature set
    pip = Pipeline([("s", StandardScaler()),
                    ("c", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))])
    pip.fit(X_pool_tr, y_tr)
    pred = pip.predict(X_pool_te)
    base_f1 = f1_score(y_te, pred, pos_label=1, zero_division=0)

    rows = []
    for i in range(18):
        keep = [j for j in range(54) if j not in (i, 18 + i, 36 + i)]
        Xtr = X_pool_tr[:, keep]
        Xte = X_pool_te[:, keep]
        pip = Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))])
        pip.fit(Xtr, y_tr)
        pred = pip.predict(Xte)
        f1 = f1_score(y_te, pred, pos_label=1, zero_division=0)
        rows.append((DIM_LABELS[i], float(f1), float(base_f1 - f1)))
    rows.sort(key=lambda r: r[2], reverse=True)
    return base_f1, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--eval-parametric", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] train {args.train_parametric.name}")
    Xp_tr, Xs_tr, y_tr, ys_tr = build_anger(args.train_parametric, args.train_csv)
    print(f"[load] eval  {args.eval_parametric.name}")
    Xp_te, Xs_te, y_te, ys_te = build_anger(args.eval_parametric, args.eval_csv)
    n_anger_tr, n_anger_te = int(y_tr.sum()), int(y_te.sum())
    print(f"  train: {len(y_tr)} utts ({n_anger_tr} anger, {100*n_anger_tr/len(y_tr):.1f}%)")
    print(f"  eval:  {len(y_te)} utts ({n_anger_te} anger, {100*n_anger_te/len(y_te):.1f}%)")
    print(f"  syllables: train {len(ys_tr)}, eval {len(ys_te)}")
    print()

    # 1. Syllable-level ANOVA F-statistic
    print("=" * 78)
    print("View 1 — Per-dim ANOVA F-statistic (syllable-level, anger vs not)")
    print("=" * 78)
    print("  (computed on TRAIN only; higher F = bigger distribution shift on anger)")
    print()
    print(f"  {'dim':<22s}  {'F':>10s}  {'p':>10s}")
    rows1 = per_dim_anova(Xs_tr, ys_tr)
    for name, F, p in rows1:
        print(f"  {name:<22s}  {F:10.2f}  {p:10.2e}")

    # 2. Per-dim univariate AUC (utterance pooled)
    print()
    print("=" * 78)
    print("View 2 — Per-dim univariate AUC for anger (pooled mean+max+std)")
    print("=" * 78)
    print("  (LR on each dim alone, train→test, predicting anger=1 vs other=0)")
    print()
    print(f"  {'dim':<22s}  {'AUC':>10s}")
    rows2 = per_dim_univariate_auc(Xp_tr, y_tr, Xp_te, y_te)
    for name, auc in rows2:
        print(f"  {name:<22s}  {auc:10.4f}")

    # 3. Drop-one-dim ablation
    print()
    print("=" * 78)
    print("View 3 — Drop-one-dim ablation (LR with class_weight='balanced')")
    print("=" * 78)
    base_f1, rows3 = single_dim_removed_ablation(Xp_tr, y_tr, Xp_te, y_te)
    print(f"  baseline F1 (anger, full 54-dim): {base_f1:.4f}")
    print()
    print(f"  {'dim removed':<22s}  {'anger_F1':>10s}  {'Δ vs base':>12s}")
    for name, f1, delta in rows3:
        marker = ""
        if delta >= 0.01:
            marker = "  ← load-bearing"
        elif delta <= -0.01:
            marker = "  ← removing helps"
        print(f"  {name:<22s}  {f1:10.4f}  {delta:+12.4f}{marker}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "n_train": int(len(y_tr)),
                "n_eval": int(len(y_te)),
                "n_anger_train": n_anger_tr,
                "n_anger_eval": n_anger_te,
                "baseline_f1": float(base_f1),
                "anova_F": [{"dim": n, "F": F, "p": p} for n, F, p in rows1],
                "univariate_auc": [{"dim": n, "auc": a} for n, a in rows2],
                "drop_one_ablation": [
                    {"dim_removed": n, "f1": f1, "delta_vs_base": d} for n, f1, d in rows3
                ],
            }, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
