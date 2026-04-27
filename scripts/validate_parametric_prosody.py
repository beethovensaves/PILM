"""
Phase 1.5 gate test — train probes over the parametric prosody vectors and
ask whether they recover MELD's emotion / sentiment labels better than a
majority-class baseline. (Per pivot 2026-04-25, MELD emotion is the primary
gate target since AuToBI's pre-trained classifiers are not readily available.)

Pipeline:
    1. Load per-utterance parametric vectors from
       `scripts/extract_parametric_prosody.py`'s JSONL output.
    2. Pool per-syllable 18-dim vectors → utterance-level feature vector
       (default: concat of mean+max+std across syllables = 54 dims).
    3. NaN → column-mean imputation.
    4. Standardize, fit several probes with stratified k-fold CV.
    5. Compare against majority-class and stratified-random baselines.
    6. PASS gate if linear-probe macro-F1 exceeds majority baseline by ≥ 0.10.

Usage:
    .venv/bin/python scripts/validate_parametric_prosody.py \\
        --parametric data/meld/parametric_prosody_dev.jsonl \\
        --targets    data/meld/emotion_probe_targets_dev.jsonl \\
        --label-key  emotion \\
        --out        data/meld/phase1_5_gate_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_parametric(path: Path) -> dict[str, np.ndarray]:
    """utterance_id -> array of shape (n_syllables, 18) with NaN for null."""
    out: dict[str, np.ndarray] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            vecs = []
            for syl in d["syllables"]:
                vecs.append([x if x is not None else float("nan") for x in syl["vec"]])
            if vecs:
                out[d["utterance_id"]] = np.array(vecs, dtype=np.float64)
    return out


def load_targets(path: Path, label_key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["utterance_id"]] = d[label_key]
    return out


def pool_to_utterance(vecs_per_syl: np.ndarray, mode: str) -> np.ndarray:
    """Pool per-syllable (n_syl, 18) → flat utterance feature.

    Modes:
        'mean'         — mean over syllables (18 dims)
        'mean+max'     — concat (36 dims)
        'mean+max+std' — concat (54 dims)
        'mean+max+std+min' — concat (72 dims)
    """
    pieces = []
    for op in mode.split("+"):
        if op == "mean":
            pieces.append(np.nanmean(vecs_per_syl, axis=0))
        elif op == "max":
            pieces.append(np.nanmax(vecs_per_syl, axis=0))
        elif op == "min":
            pieces.append(np.nanmin(vecs_per_syl, axis=0))
        elif op == "std":
            pieces.append(np.nanstd(vecs_per_syl, axis=0))
        else:
            raise ValueError(f"unknown pool op: {op}")
    return np.concatenate(pieces)


def build_dataset(
    parametric_path: Path,
    targets_path: Path,
    label_key: str,
    pool_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    parametric = load_parametric(parametric_path)
    targets = load_targets(targets_path, label_key)
    common_ids = sorted(set(parametric) & set(targets))
    X: list[np.ndarray] = []
    y: list[str] = []
    skipped_no_label = 0
    skipped_no_features = 0
    for uid in common_ids:
        if not targets[uid] or targets[uid] == "nan":
            skipped_no_label += 1
            continue
        feat = pool_to_utterance(parametric[uid], pool_mode)
        if np.all(np.isnan(feat)):
            skipped_no_features += 1
            continue
        X.append(feat)
        y.append(targets[uid])
    X_arr = np.vstack(X)
    # Mean-impute NaN with per-column mean across utterances
    col_means = np.nanmean(X_arr, axis=0)
    inds = np.where(np.isnan(X_arr))
    X_arr[inds] = np.take(col_means, inds[1])
    if skipped_no_label or skipped_no_features:
        print(f"[load] skipped {skipped_no_label} no-label, {skipped_no_features} all-NaN")
    return X_arr, np.array(y), common_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parametric", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--label-key", default="emotion", choices=["emotion", "sentiment"])
    parser.add_argument("--pool-mode", default="mean+max+std")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-delta", type=float, default=0.10,
                        help="Macro-F1 margin over majority baseline required to PASS")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] parametric={args.parametric}")
    print(f"[load] targets   ={args.targets}  label_key={args.label_key}")
    X, y, _ = build_dataset(args.parametric, args.targets, args.label_key, args.pool_mode)
    print(f"[load] dataset  : X={X.shape}  classes={len(set(y))}")
    counts = pd.Series(y).value_counts()
    print(f"[load] class counts: {dict(counts)}")
    majority_pct = counts.iloc[0] / len(y) * 100
    print(f"[load] majority class = {counts.index[0]!r} ({majority_pct:.1f}%)")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    probes = {
        "majority_baseline":   DummyClassifier(strategy="most_frequent"),
        "stratified_baseline": DummyClassifier(strategy="stratified", random_state=args.seed),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(max_iter=2000, random_state=args.seed)),
        ]),
        "mlp_64": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=args.seed)),
        ]),
    }

    print(f"\n[probe] {args.cv}-fold stratified CV, pool_mode={args.pool_mode}")
    print(f"{'probe':<24s}  {'macro_F1':>10s}  {'accuracy':>10s}")
    print("-" * 50)
    results: dict[str, dict] = {}
    for name, model in probes.items():
        macro_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_macro").mean()
        accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        results[name] = {"macro_f1": float(macro_f1), "accuracy": float(accuracy)}
        print(f"{name:<24s}  {macro_f1:10.4f}  {accuracy:10.4f}")

    bl = results["majority_baseline"]["macro_f1"]
    lr = results["logistic_regression"]["macro_f1"]
    mlp = results["mlp_64"]["macro_f1"]
    delta_lr = lr - bl
    delta_mlp = mlp - bl
    print(f"\n[gate] LR macro-F1 − majority macro-F1 = {lr:.4f} − {bl:.4f} = {delta_lr:+.4f}")
    print(f"[gate] MLP macro-F1 − majority macro-F1 = {mlp:.4f} − {bl:.4f} = {delta_mlp:+.4f}")
    passed = delta_lr >= args.gate_delta or delta_mlp >= args.gate_delta
    if passed:
        print(f"[gate] PASS (Δ ≥ {args.gate_delta:.2f})")
    else:
        print(f"[gate] FAIL (Δ < {args.gate_delta:.2f})")
        print("       parametric vector → emotion is weak; revisit dim spec or pool strategy")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "dataset_shape": list(X.shape),
                "classes": sorted(set(y)),
                "class_counts": {k: int(v) for k, v in counts.items()},
                "pool_mode": args.pool_mode,
                "cv_folds": args.cv,
                "label_key": args.label_key,
                "probes": results,
                "delta_lr_vs_majority": float(delta_lr),
                "delta_mlp_vs_majority": float(delta_mlp),
                "gate_delta_required": args.gate_delta,
                "gate_passed": passed,
            }, f, indent=2)
        print(f"\n[out] results written to {args.out}")


if __name__ == "__main__":
    main()
