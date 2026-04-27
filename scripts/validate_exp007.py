"""
EXP-007 validation suite — four perturbations of the question-prediction probe
to test whether the AUC 0.65–0.69 result is real or driven by confounds.

Variants (all on yes/no questions only — wh-Qs dropped from positives):

    (a) Speaker-held-out via GroupKFold(5) on speaker. Tests whether the
        probe is partly speaker-fingerprinting + matching their question
        habits, given that 96% of MELD test utterances are from speakers
        also in train.

    (b) Neutral-only subset. Restrict to utterances with Emotion='neutral'.
        Tests whether "question prosody" is actually a confounded
        emotion-prosody (yes/no Qs cluster in surprise/anxiety).

    (c) Position ablation: first-syl LR, middle-syl LR, last-syl LR.
        Theory predicts the boundary tone is on the last syllable; if
        random positions work as well, the theory's wrong.

    (d) Wh-only subset. Positives are wh-Qs (text contains '?' AND first
        word is wh). Wh-Qs canonically *fall*. Our LR coefficients were
        tuned to rises on yn-only. AUC should drop substantially.

Plus (e) bootstrap CIs on the top 4 last-syllable LR coefficients from the
yn-only fit.

Usage:
    .venv/bin/python scripts/validate_exp007.py \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv        data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-parametric  data/meld/parametric_prosody_test_mfa.jsonl \\
        --eval-csv         data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --dev-parametric   data/meld/parametric_prosody_dev_mfa.jsonl \\
        --dev-csv          data/meld/MELD.Raw/dev_sent_emo_cleaned.csv \\
        --out              data/meld/validate_exp007.json
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
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


# ---------------------------------------------------------------------------
# Loader — returns rich rows that variants slice from
# ---------------------------------------------------------------------------

def build_rich(parametric_path: Path, csv_path: Path) -> pd.DataFrame:
    """One row per utterance with everything we need for any variant.

    Columns:
        utterance_id, speaker, emotion, text, has_q, is_wh_q, is_yn_q
        x_pool: np.ndarray[54]  (mean+max+std)
        x_first, x_middle, x_last: np.ndarray[18]
    """
    parametric = load_parametric(parametric_path)
    df = pd.read_csv(csv_path)
    df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
    df = df[df["utterance_id"].isin(parametric)].copy()
    df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()

    rows = []
    for row in df.itertuples():
        utt = str(row.Utterance)
        has_q = "?" in utt
        m = re.search(r"[A-Za-z]+", utt)
        first_word = m.group(0).lower() if m else ""
        is_wh_q = has_q and first_word in WH_WORDS
        is_yn_q = has_q and not is_wh_q

        vecs = parametric[row.utterance_id]
        if len(vecs) == 0:
            continue
        pooled = pool_to_utterance(vecs, "mean+max+std")
        if np.all(np.isnan(pooled)):
            continue
        first = vecs[0]
        last = vecs[-1]
        middle = vecs[len(vecs) // 2]
        rows.append({
            "utterance_id": row.utterance_id,
            "speaker": row.Speaker,
            "emotion": row.Emotion,
            "text": utt,
            "has_q": has_q,
            "is_wh_q": is_wh_q,
            "is_yn_q": is_yn_q,
            "x_pool": pooled,
            "x_first": first,
            "x_middle": middle,
            "x_last": last,
        })
    out = pd.DataFrame(rows)
    # NaN-impute each x_* matrix using its column means
    for col in ("x_pool", "x_first", "x_middle", "x_last"):
        mat = np.vstack(out[col].values)
        cm = np.nanmean(mat, axis=0)
        ind = np.where(np.isnan(mat))
        mat[ind] = np.take(cm, ind[1])
        out[col] = list(mat)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack(series) -> np.ndarray:
    return np.vstack(series.values)


def lr_auc(Xtr, ytr, Xte, yte) -> tuple[float, float, np.ndarray]:
    """Standard-scaled LR; return (AUC, pos_F1, coefficients)."""
    pip = Pipeline([("s", StandardScaler()),
                    ("c", LogisticRegression(max_iter=2000, C=1.0))])
    pip.fit(Xtr, ytr)
    proba = pip.predict_proba(Xte)[:, 1]
    pred = pip.predict(Xte)
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    f1 = f1_score(yte, pred, pos_label=1, zero_division=0) if len(set(yte)) > 1 else float("nan")
    coef = pip.named_steps["c"].coef_[0]
    return float(auc), float(f1), coef


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def variant_baseline(train: pd.DataFrame, eval_: pd.DataFrame) -> dict:
    """Reproduce EXP-007 numbers as the reference: yn-only, last-syl LR."""
    tr = train[~train["is_wh_q"] | (train["is_wh_q"] & False)].copy()  # drop wh-q from positives
    tr = tr[~tr["is_wh_q"]]
    te = eval_[~eval_["is_wh_q"]].copy()
    Xtr, ytr = _stack(tr["x_last"]), tr["is_yn_q"].astype(int).values
    Xte, yte = _stack(te["x_last"]), te["is_yn_q"].astype(int).values
    auc, f1, coef = lr_auc(Xtr, ytr, Xte, yte)
    return {"variant": "baseline (yn-only, last-syl LR, train→test)",
            "n_train": len(tr), "n_eval": len(te),
            "n_pos_train": int(ytr.sum()), "n_pos_eval": int(yte.sum()),
            "auc": auc, "pos_f1": f1,
            "coef": [{"dim": DIM_LABELS[i], "coef": float(coef[i])} for i in range(18)]}


def variant_a_speaker_groupkfold(all_data: pd.DataFrame, n_splits: int = 5) -> dict:
    """Speaker-disjoint GroupKFold(n_splits) on yn-only, last-syl LR."""
    pool = all_data[~all_data["is_wh_q"]].copy()
    X = _stack(pool["x_last"])
    y = pool["is_yn_q"].astype(int).values
    groups = pool["speaker"].values
    aucs, f1s = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(X, y, groups):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        auc, f1, _ = lr_auc(Xtr, ytr, Xte, yte)
        aucs.append(auc)
        f1s.append(f1)
    return {"variant": f"(a) speaker-held-out GroupKFold({n_splits}), yn-only, last-syl LR",
            "n_total": len(pool),
            "n_pos": int(y.sum()),
            "n_speakers": int(pool["speaker"].nunique()),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "auc_per_fold": [float(a) for a in aucs],
            "pos_f1_mean": float(np.mean(f1s))}


def variant_b_neutral_only(train: pd.DataFrame, eval_: pd.DataFrame) -> dict:
    """Drop emotional utterances; check yn-only AUC on neutral-only subset."""
    tr = train[(~train["is_wh_q"]) & (train["emotion"] == "neutral")]
    te = eval_[(~eval_["is_wh_q"]) & (eval_["emotion"] == "neutral")]
    if len(tr) == 0 or len(te) == 0:
        return {"variant": "(b) neutral-only", "error": "empty after filter"}
    Xtr, ytr = _stack(tr["x_last"]), tr["is_yn_q"].astype(int).values
    Xte, yte = _stack(te["x_last"]), te["is_yn_q"].astype(int).values
    auc, f1, _ = lr_auc(Xtr, ytr, Xte, yte)
    return {"variant": "(b) neutral-only utterances, yn-only, last-syl LR",
            "n_train": len(tr), "n_eval": len(te),
            "n_pos_train": int(ytr.sum()), "n_pos_eval": int(yte.sum()),
            "auc": auc, "pos_f1": f1}


def variant_c_position(train: pd.DataFrame, eval_: pd.DataFrame) -> dict:
    """First / middle / last syllable LRs, all yn-only."""
    tr = train[~train["is_wh_q"]]
    te = eval_[~eval_["is_wh_q"]]
    ytr = tr["is_yn_q"].astype(int).values
    yte = te["is_yn_q"].astype(int).values
    out = {"variant": "(c) position ablation: first vs middle vs last syllable, yn-only",
           "n_train": len(tr), "n_eval": len(te),
           "n_pos_train": int(ytr.sum()), "n_pos_eval": int(yte.sum())}
    for name, col in [("first_syl", "x_first"), ("middle_syl", "x_middle"), ("last_syl", "x_last")]:
        Xtr, Xte = _stack(tr[col]), _stack(te[col])
        auc, f1, _ = lr_auc(Xtr, ytr, Xte, yte)
        out[name] = {"auc": auc, "pos_f1": f1}
    return out


def variant_d_wh_only(train: pd.DataFrame, eval_: pd.DataFrame) -> dict:
    """Positives = wh-Qs; negatives = utterances without '?'. Drop yn-Qs."""
    tr = train[~train["is_yn_q"]]
    te = eval_[~eval_["is_yn_q"]]
    Xtr, ytr = _stack(tr["x_last"]), tr["is_wh_q"].astype(int).values
    Xte, yte = _stack(te["x_last"]), te["is_wh_q"].astype(int).values
    if int(yte.sum()) == 0:
        return {"variant": "(d) wh-only", "error": "no positives in eval"}
    auc, f1, _ = lr_auc(Xtr, ytr, Xte, yte)
    return {"variant": "(d) wh-only positives (yn-Qs dropped), last-syl LR",
            "n_train": len(tr), "n_eval": len(te),
            "n_pos_train": int(ytr.sum()), "n_pos_eval": int(yte.sum()),
            "auc": auc, "pos_f1": f1}


def variant_e_bootstrap_coef(train: pd.DataFrame, eval_: pd.DataFrame,
                               n_boot: int = 200, seed: int = 42) -> dict:
    """Bootstrap CI on each last-syllable coefficient from yn-only LR."""
    tr = train[~train["is_wh_q"]]
    te = eval_[~eval_["is_wh_q"]]
    Xtr, ytr = _stack(tr["x_last"]), tr["is_yn_q"].astype(int).values
    Xte, yte = _stack(te["x_last"]), te["is_yn_q"].astype(int).values

    rng = np.random.default_rng(seed)
    n = len(Xtr)
    coefs = []
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(ytr[idx])) < 2:
            continue
        auc, _, coef = lr_auc(Xtr[idx], ytr[idx], Xte, yte)
        coefs.append(coef)
        aucs.append(auc)
    coefs = np.vstack(coefs)
    medians = np.median(coefs, axis=0)
    lo = np.percentile(coefs, 2.5, axis=0)
    hi = np.percentile(coefs, 97.5, axis=0)

    rows = []
    for i in range(18):
        sign_stable = (lo[i] > 0 and hi[i] > 0) or (lo[i] < 0 and hi[i] < 0)
        rows.append({"dim": DIM_LABELS[i],
                     "median": float(medians[i]),
                     "ci_low": float(lo[i]),
                     "ci_high": float(hi[i]),
                     "sign_stable": bool(sign_stable)})
    rows.sort(key=lambda r: abs(r["median"]), reverse=True)
    return {"variant": f"(e) bootstrap CI on last-syl coefficients (n_boot={n_boot}, yn-only)",
            "auc_median": float(np.median(aucs)),
            "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "coefficients": rows}


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_baseline(r: dict) -> None:
    print_section("Baseline — reproduce EXP-007 (last-syl LR, yn-only, train→test)")
    print(f"  n_train={r['n_train']} ({r['n_pos_train']} yn-Qs)")
    print(f"  n_eval ={r['n_eval']} ({r['n_pos_eval']} yn-Qs)")
    print(f"  AUC = {r['auc']:.4f}   pos_F1 = {r['pos_f1']:.4f}")
    print()
    print("  Top 6 coefficients:")
    sorted_coef = sorted(r["coef"], key=lambda c: abs(c["coef"]), reverse=True)[:6]
    for c in sorted_coef:
        sign = "+" if c["coef"] > 0 else "−"
        print(f"    {sign}{abs(c['coef']):.3f}  {c['dim']}")


def print_a(r: dict) -> None:
    print_section("(a) Speaker-held-out (GroupKFold by speaker)")
    print(f"  pool: {r['n_total']} utts, {r['n_pos']} yn-Qs, {r['n_speakers']} speakers")
    print(f"  AUC mean ± std: {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    print(f"  per-fold AUC:   {[round(a, 3) for a in r['auc_per_fold']]}")
    print(f"  pos_F1 mean:    {r['pos_f1_mean']:.4f}")


def print_b(r: dict) -> None:
    print_section("(b) Neutral-only utterances")
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
    print(f"  n_train={r['n_train']} ({r['n_pos_train']} yn-Qs)")
    print(f"  n_eval ={r['n_eval']} ({r['n_pos_eval']} yn-Qs)")
    print(f"  AUC = {r['auc']:.4f}   pos_F1 = {r['pos_f1']:.4f}")


def print_c(r: dict) -> None:
    print_section("(c) Position ablation: first vs middle vs last syllable")
    print(f"  n_train={r['n_train']} ({r['n_pos_train']} yn-Qs)")
    print(f"  n_eval ={r['n_eval']} ({r['n_pos_eval']} yn-Qs)")
    print(f"  {'position':<12s}  {'AUC':>8s}  {'pos_F1':>8s}")
    for name in ("first_syl", "middle_syl", "last_syl"):
        d = r[name]
        print(f"  {name:<12s}  {d['auc']:>8.4f}  {d['pos_f1']:>8.4f}")


def print_d(r: dict) -> None:
    print_section("(d) Wh-only positives (negatives = no-'?')")
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
    print(f"  n_train={r['n_train']} ({r['n_pos_train']} wh-Qs)")
    print(f"  n_eval ={r['n_eval']} ({r['n_pos_eval']} wh-Qs)")
    print(f"  AUC = {r['auc']:.4f}   pos_F1 = {r['pos_f1']:.4f}")


def print_e(r: dict) -> None:
    print_section("(e) Bootstrap CIs on last-syl coefficients (yn-only)")
    print(f"  AUC median: {r['auc_median']:.4f}   95% CI: [{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}]")
    print()
    print(f"  {'dim':<22s}  {'median':>8s}  {'CI 2.5%':>8s}  {'CI 97.5%':>8s}  stable?")
    for c in r["coefficients"][:8]:
        marker = "✓" if c["sign_stable"] else "×"
        print(f"  {c['dim']:<22s}  {c['median']:>+8.3f}  {c['ci_low']:>+8.3f}  {c['ci_high']:>+8.3f}    {marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--eval-parametric", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--dev-parametric", type=Path, default=None,
                        help="Optional: dev split, used only for the GroupKFold pool in (a)")
    parser.add_argument("--dev-csv", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] train {args.train_parametric.name}")
    train = build_rich(args.train_parametric, args.train_csv)
    print(f"[load] eval  {args.eval_parametric.name}")
    eval_ = build_rich(args.eval_parametric, args.eval_csv)

    print(f"  train: {len(train)} utts ({train['is_yn_q'].sum()} yn-Q, "
          f"{train['is_wh_q'].sum()} wh-Q, {train['speaker'].nunique()} speakers)")
    print(f"  eval:  {len(eval_)} utts ({eval_['is_yn_q'].sum()} yn-Q, "
          f"{eval_['is_wh_q'].sum()} wh-Q, {eval_['speaker'].nunique()} speakers)")

    # baseline
    rb = variant_baseline(train, eval_)
    print_baseline(rb)

    # (a) — pool train+eval(+dev) for GroupKFold
    pool = train.copy()
    pool = pd.concat([pool, eval_], ignore_index=True)
    if args.dev_parametric and args.dev_csv:
        print(f"[load] dev  {args.dev_parametric.name}")
        dev = build_rich(args.dev_parametric, args.dev_csv)
        pool = pd.concat([pool, dev], ignore_index=True)
        print(f"  +dev: pool now {len(pool)} utts, {pool['speaker'].nunique()} speakers")
    ra = variant_a_speaker_groupkfold(pool)
    print_a(ra)

    # (b)
    rb_neu = variant_b_neutral_only(train, eval_)
    print_b(rb_neu)

    # (c)
    rc = variant_c_position(train, eval_)
    print_c(rc)

    # (d)
    rd = variant_d_wh_only(train, eval_)
    print_d(rd)

    # (e)
    re_ = variant_e_bootstrap_coef(train, eval_, n_boot=args.n_boot)
    print_e(re_)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "baseline": rb, "a_speaker": ra, "b_neutral": rb_neu,
                "c_position": rc, "d_wh_only": rd, "e_bootstrap": re_,
            }, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
