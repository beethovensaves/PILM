"""
EXP-010 — Cross-corpus probes on AMI.

Reads parametric_prosody_ami_v1.jsonl (one line per AMI segment, with
das_in_seg labels) and runs the analogues of EXP-005 and EXP-007 on AMI.

Probes:
  (A) Prosody-only AUC for predicting el.inf (Elicit-Inform) vs clear-statement
      DAs. Three flavours: LR pooled (mean+max+std), LR last-syllable (18 dim),
      bi-LSTM sequence (mean pool).
  (B) Text vs prosody vs combined macro-F1 for the same task, 5-fold CV
      stratified by meeting.
  (C) Speaker-held-out (GroupKFold over global_name) AUC on prosody-only
      last-syllable LR.
  (D) Position ablation: first / middle / last syllable LR AUC.
  (E) Bootstrap CIs (n=200) on top last-syllable LR coefficients.

Then runs the SAME prosody-only last-syl LR on MELD v1 question-prediction
data for apples-to-apples cross-corpus comparison.

Output: data/ami/exp010_results.json with all metrics.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

DIM_LABELS = (
    "f0_onset_st", "f0_nucleus_st", "f0_offset_st",
    "f0_max_st", "f0_min_st", "f0_range_st", "f0_slope_st_per_ms",
    "f0_peak_pos", "f0_rise_amp", "f0_fall_amp", "tilt",
    "rms_max_z", "rms_mean_z",
    "syl_dur_z", "nuc_dur_z",
    "pause_after_ms", "final_lengthen", "f0_reset_st",
)

POS_DA = {"el.inf"}
NEG_DA = {"inf", "ass", "sug", "off", "und", "be.pos", "be.neg"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def pool_to_utt(vecs: np.ndarray) -> np.ndarray:
    """mean+max+std pool over syllable axis, NaN-tolerant."""
    return np.concatenate([
        np.nanmean(vecs, axis=0), np.nanmax(vecs, axis=0), np.nanstd(vecs, axis=0)
    ])


def label_for(das: list[str]) -> int:
    if any(d in POS_DA for d in das):
        return 1
    if any(d in NEG_DA for d in das) and not any(d in POS_DA for d in das):
        return 0
    return -1  # drop


def load_ami(path: Path) -> dict:
    """Returns dict with: x_pool (N,54), x_first (N,18), x_middle, x_last,
    x_seq (N variable-length sequences), y (N,), groups (N,), texts (N,),
    meeting (N,)."""
    rows = {"x_pool": [], "x_first": [], "x_middle": [], "x_last": [],
            "x_seq": [], "y": [], "groups": [], "texts": [], "meeting": []}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            das = d.get("das_in_seg", [])
            y = label_for(das)
            if y < 0:
                continue
            syls = d.get("syllables", [])
            if not syls:
                continue
            vecs = np.array([
                [float("nan") if v is None else float(v) for v in s["vec"]]
                for s in syls
            ])
            if np.all(np.isnan(vecs)):
                continue
            rows["x_pool"].append(pool_to_utt(vecs))
            rows["x_first"].append(vecs[0])
            rows["x_middle"].append(vecs[len(vecs) // 2])
            rows["x_last"].append(vecs[-1])
            rows["x_seq"].append(vecs)
            rows["y"].append(y)
            rows["groups"].append(d.get("speaker_id", "?"))  # global_name
            rows["texts"].append(d.get("text", ""))
            rows["meeting"].append(d.get("meeting", "?"))
    out = {}
    for col in ("x_pool", "x_first", "x_middle", "x_last"):
        arr = np.vstack(rows[col])
        cm = np.nanmean(arr, axis=0)
        ind = np.where(np.isnan(arr))
        arr[ind] = np.take(cm, ind[1])
        out[col] = arr
    out["x_seq"] = rows["x_seq"]
    out["y"] = np.array(rows["y"])
    out["groups"] = np.array(rows["groups"])
    out["texts"] = rows["texts"]
    out["meeting"] = np.array(rows["meeting"])
    return out


def load_meld_question(parametric_path: Path, csv_path: Path) -> dict:
    """For cross-corpus comparison: MELD v1 yn-only question prediction."""
    import pandas as pd
    import re
    WH = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}

    df = pd.read_csv(csv_path)
    df["uid"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)

    parametric = {}
    with parametric_path.open() as f:
        for line in f:
            d = json.loads(line)
            parametric[d["utterance_id"]] = d["syllables"]

    rows = {"x_pool": [], "x_last": [], "y": [], "groups": []}
    for row in df.itertuples():
        if row.uid not in parametric:
            continue
        syls = parametric[row.uid]
        if not syls:
            continue
        text = str(row.Utterance)
        has_q = "?" in text
        m = re.search(r"[A-Za-z]+", text)
        first_w = m.group(0).lower() if m else ""
        is_wh = has_q and first_w in WH
        if has_q and is_wh:
            continue  # drop wh-Qs for yn-only
        y = 1 if has_q else 0

        vecs = np.array([[float("nan") if v is None else float(v) for v in s["vec"]] for s in syls])
        if np.all(np.isnan(vecs)):
            continue
        rows["x_pool"].append(pool_to_utt(vecs))
        rows["x_last"].append(vecs[-1])
        rows["y"].append(y)
        rows["groups"].append(row.Speaker)

    out = {}
    for col in ("x_pool", "x_last"):
        arr = np.vstack(rows[col])
        cm = np.nanmean(arr, axis=0)
        ind = np.where(np.isnan(arr))
        arr[ind] = np.take(cm, ind[1])
        out[col] = arr
    out["y"] = np.array(rows["y"])
    out["groups"] = np.array(rows["groups"])
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def lr_auc(Xtr, ytr, Xte, yte) -> tuple[float, float, np.ndarray]:
    pip = Pipeline([("s", StandardScaler()),
                    ("c", LogisticRegression(max_iter=2000, C=1.0))])
    pip.fit(Xtr, ytr)
    proba = pip.predict_proba(Xte)[:, 1]
    pred = pip.predict(Xte)
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    f1 = f1_score(yte, pred, pos_label=1, zero_division=0) if len(set(yte)) > 1 else 0.0
    return float(auc), float(f1), pip.named_steps["c"].coef_[0]


def kfold_lr(X, y, groups, n_splits=5, by_group=False) -> dict:
    """5-fold CV either stratified or grouped. Returns mean AUC and per-fold."""
    if by_group:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
    aucs, f1s = [], []
    for tr, te in splits:
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        auc, f1, _ = lr_auc(X[tr], y[tr], X[te], y[te])
        aucs.append(auc)
        f1s.append(f1)
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "auc_per_fold": [float(a) for a in aucs],
            "pos_f1_mean": float(np.mean(f1s))}


# ---------------------------------------------------------------------------
# bi-LSTM probe
# ---------------------------------------------------------------------------

class SeqDS(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        s = self.seqs[i]
        # NaN -> 0 imputation
        s = np.where(np.isnan(s), 0.0, s)
        return torch.tensor(s, dtype=torch.float32), int(self.labels[i])


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels_t


class BiLSTMProbe(nn.Module):
    def __init__(self, input_dim=18, hidden=64, layers=2, dropout=0.2, pool="mean"):
        super().__init__()
        assert pool in {"mean", "attention"}
        self.pool_kind = pool
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                             bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.attn_q = nn.Linear(2 * hidden, 1) if pool == "attention" else None
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, lengths):
        x = self.norm(x)
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        B, T, H = out.shape
        mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.to(out.device).unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        if self.pool_kind == "mean":
            pooled = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        else:
            scores = self.attn_q(out).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (out * weights).sum(1)
        return self.head(pooled).squeeze(-1)


def bilstm_eval(seqs, y, hidden=64, layers=2, epochs=10, pool="mean",
                 n_splits=5, seed=42) -> dict:
    """5-fold stratified bi-LSTM probe with configurable architecture."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        ytr, yte = y[tr], y[te]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        seqs_tr = [seqs[i] for i in tr]
        seqs_te = [seqs[i] for i in te]
        n_pos = max(1, int(ytr.sum()))
        pos_w = torch.tensor((len(ytr) - n_pos) / n_pos, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        model = BiLSTMProbe(hidden=hidden, layers=layers, pool=pool).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_loader = DataLoader(SeqDS(seqs_tr, ytr), batch_size=64, shuffle=True, collate_fn=collate)
        te_loader = DataLoader(SeqDS(seqs_te, yte), batch_size=64, shuffle=False, collate_fn=collate)
        for _ in range(epochs):
            model.train()
            for s, lens, lbl in tr_loader:
                s, lbl = s.to(device), lbl.to(device)
                opt.zero_grad()
                logits = model(s, lens)
                loss = loss_fn(logits, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        all_logits, all_lbl = [], []
        with torch.no_grad():
            for s, lens, lbl in te_loader:
                s = s.to(device)
                all_logits.extend(model(s, lens).cpu().numpy().tolist())
                all_lbl.extend(lbl.numpy().tolist())
        probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
        aucs.append(float(roc_auc_score(all_lbl, probs)))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "auc_per_fold": aucs,
            "config": {"hidden": hidden, "layers": layers, "epochs": epochs, "pool": pool}}


# ---------------------------------------------------------------------------
# Text vs prosody comparison (analogue of EXP-005 for AMI el.inf)
# ---------------------------------------------------------------------------

def text_probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                    sublinear_tf=True, lowercase=True,
                                    token_pattern=r"(?u)\b\w+\b|[!?]+|\.{2,}")),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def text_vs_prosody(d: dict, n_splits=5) -> dict:
    """5-fold CV. Text-only / prosody-only / combined macro-F1."""
    X_pros = d["x_pool"]
    texts = d["texts"]
    y = d["y"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = {"text": [], "prosody": [], "combined": []}
    for tr, te in cv.split(X_pros, y):
        ytr, yte = y[tr], y[te]
        # Text
        clf = text_probe()
        clf.fit([texts[i] for i in tr], ytr)
        pred = clf.predict([texts[i] for i in te])
        res["text"].append(f1_score(yte, pred, average="macro", zero_division=0))
        # Prosody
        pros = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000, C=1.0))])
        pros.fit(X_pros[tr], ytr)
        res["prosody"].append(f1_score(yte, pros.predict(X_pros[te]),
                                          average="macro", zero_division=0))
        # Combined: stack TF-IDF features + scaled prosody
        tfidf = clf.named_steps["tfidf"]
        ttr = tfidf.transform([texts[i] for i in tr]).toarray()
        tte = tfidf.transform([texts[i] for i in te]).toarray()
        scaler = StandardScaler()
        ptr = scaler.fit_transform(X_pros[tr])
        pte = scaler.transform(X_pros[te])
        Xc_tr = np.hstack([ttr, ptr]); Xc_te = np.hstack([tte, pte])
        comb = LogisticRegression(max_iter=2000, C=1.0)
        comb.fit(Xc_tr, ytr)
        res["combined"].append(f1_score(yte, comb.predict(Xc_te),
                                          average="macro", zero_division=0))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                  "per_fold": [float(x) for x in v]} for k, v in res.items()}


# ---------------------------------------------------------------------------
# Bootstrap CIs on last-syl LR coefficients
# ---------------------------------------------------------------------------

def bootstrap_coefs(X, y, n_boot=200, seed=42) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(X)
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        pip = Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=2000, C=1.0))])
        pip.fit(X[idx], y[idx])
        coefs.append(pip.named_steps["c"].coef_[0])
    coefs = np.vstack(coefs)
    out = []
    for i in range(coefs.shape[1]):
        med = float(np.median(coefs[:, i]))
        lo = float(np.percentile(coefs[:, i], 2.5))
        hi = float(np.percentile(coefs[:, i], 97.5))
        out.append({"dim": DIM_LABELS[i], "median": med, "ci_low": lo,
                     "ci_high": hi, "sign_stable": (lo > 0 and hi > 0) or (lo < 0 and hi < 0)})
    out.sort(key=lambda r: abs(r["median"]), reverse=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-parametric", type=Path,
                        default=Path("data/ami/parametric_prosody_ami_v1.jsonl"))
    parser.add_argument("--meld-train-parametric", type=Path,
                        default=Path("data/meld/parametric_prosody_train_v1.jsonl"))
    parser.add_argument("--meld-test-parametric", type=Path,
                        default=Path("data/meld/parametric_prosody_test_v1.jsonl"))
    parser.add_argument("--meld-train-csv", type=Path,
                        default=Path("data/meld/MELD.Raw/train_sent_emo_cleaned.csv"))
    parser.add_argument("--meld-test-csv", type=Path,
                        default=Path("data/meld/MELD.Raw/test_sent_emo_cleaned.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/ami/exp010_results.json"))
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()

    print(f"[load] AMI parametric: {args.ami_parametric}")
    ami = load_ami(args.ami_parametric)
    print(f"  n={len(ami['y'])}  pos (el.inf)={int(ami['y'].sum())}  neg={int((1-ami['y']).sum())}")
    print(f"  speakers (global_name)={len(set(ami['groups']))}")
    print(f"  meetings={len(set(ami['meeting']))}")

    results = {"counts": {
        "n_total": int(len(ami["y"])),
        "n_pos_el_inf": int(ami["y"].sum()),
        "n_neg_statement": int((1 - ami["y"]).sum()),
        "n_speakers": int(len(set(ami["groups"]))),
        "n_meetings": int(len(set(ami["meeting"]))),
    }}

    # (A) prosody-only LR pooled / last-syl, 5-fold CV
    print("\n[A] prosody-only LR (5-fold CV)")
    results["prosody_pooled_lr"] = kfold_lr(ami["x_pool"], ami["y"], ami["groups"])
    print(f"  pooled: AUC mean={results['prosody_pooled_lr']['auc_mean']:.4f}")
    results["prosody_last_syl_lr"] = kfold_lr(ami["x_last"], ami["y"], ami["groups"])
    print(f"  last-syl: AUC mean={results['prosody_last_syl_lr']['auc_mean']:.4f}")

    # bi-LSTM (small + rich)
    print("\n[A] prosody-only bi-LSTM (5-fold CV)")
    results["prosody_bilstm_small"] = bilstm_eval(
        ami["x_seq"], ami["y"], hidden=64, layers=2, epochs=10, pool="mean")
    print(f"  bi-LSTM small (2x64, mean, 10 ep): AUC={results['prosody_bilstm_small']['auc_mean']:.4f}")
    results["prosody_bilstm_rich"] = bilstm_eval(
        ami["x_seq"], ami["y"], hidden=128, layers=4, epochs=30, pool="attention")
    print(f"  bi-LSTM rich  (4x128, attn, 30 ep): AUC={results['prosody_bilstm_rich']['auc_mean']:.4f}")
    # Backwards-compat alias for downstream consumers
    results["prosody_bilstm"] = results["prosody_bilstm_small"]

    # (B) text vs prosody vs combined
    print("\n[B] text vs prosody vs combined (macro-F1, 5-fold CV)")
    results["text_vs_prosody"] = text_vs_prosody(ami)
    for k, v in results["text_vs_prosody"].items():
        print(f"  {k}: {v['mean']:.4f}")

    # (C) speaker-held-out (GroupKFold by global_name)
    print("\n[C] speaker-held-out (GroupKFold by global_name) last-syl LR")
    results["speaker_held_out_last_syl"] = kfold_lr(
        ami["x_last"], ami["y"], ami["groups"], n_splits=5, by_group=True)
    print(f"  AUC mean={results['speaker_held_out_last_syl']['auc_mean']:.4f}")

    # (D) position ablation
    print("\n[D] position ablation (5-fold CV)")
    pos = {}
    for name, X in [("first_syl", ami["x_first"]),
                     ("middle_syl", ami["x_middle"]),
                     ("last_syl", ami["x_last"])]:
        pos[name] = kfold_lr(X, ami["y"], ami["groups"])
        print(f"  {name}: AUC mean={pos[name]['auc_mean']:.4f}")
    results["position_ablation"] = pos

    # (E) bootstrap CIs on last-syl coefficients
    print("\n[E] bootstrap CIs on last-syl coefficients")
    results["bootstrap_coefs"] = bootstrap_coefs(ami["x_last"], ami["y"], n_boot=args.n_boot)
    for c in results["bootstrap_coefs"][:6]:
        marker = "✓" if c["sign_stable"] else "×"
        print(f"  {c['dim']:<22s}  med={c['median']:+.3f}  CI=[{c['ci_low']:+.3f},{c['ci_high']:+.3f}] {marker}")

    # (F) MELD v1 cross-corpus comparison: same prosody-only last-syl LR setup
    print("\n[F] MELD v1 yn-Q prediction (apples-to-apples cross-corpus)")
    meld_tr = load_meld_question(args.meld_train_parametric, args.meld_train_csv)
    meld_te = load_meld_question(args.meld_test_parametric, args.meld_test_csv)
    auc, _, _ = lr_auc(meld_tr["x_last"], meld_tr["y"], meld_te["x_last"], meld_te["y"])
    results["meld_v1_yn_q_last_syl_auc"] = float(auc)
    auc_p, _, _ = lr_auc(meld_tr["x_pool"], meld_tr["y"], meld_te["x_pool"], meld_te["y"])
    results["meld_v1_yn_q_pooled_auc"] = float(auc_p)
    print(f"  MELD v1 yn-Q last-syl AUC = {auc:.4f}  (was {0.65:.4f} on v2 — EXP-007 baseline)")
    print(f"  MELD v1 yn-Q pooled    AUC = {auc_p:.4f}")

    # Cross-corpus headline
    print("\n=== HEADLINE ===")
    ami_auc = results["prosody_last_syl_lr"]["auc_mean"]
    print(f"  AMI el.inf prosody-only AUC (5-fold, last-syl LR): {ami_auc:.4f}")
    print(f"  MELD yn-Q prosody-only AUC (last-syl LR, v1): {auc:.4f}")
    print(f"  Gap: {ami_auc - auc:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
