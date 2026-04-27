"""
bi-LSTM probe over per-syllable parametric vectors for emotion / sentiment
classification. Generalizes scripts/bilstm_question_probe.py with:
  - multi-class output (emotion 7-way / sentiment 3-way)
  - mean-pool OR attention-pool over LSTM outputs
  - optional frame-level F0 augmentation (concat K extra dims per syllable
    if the input JSONL was processed with scripts/add_frame_f0.py)

Architecture:
    input  : (batch, T, F)   F = 19 (parametric+voicing) or 19+K (with frame-F0)
    bi-LSTM: hidden, num_layers, bidirectional → output (batch, T, 2*hidden)
    pool   : mean (mask-aware) OR attention (learned weighted sum)
    head   : Linear(2*hidden, hidden) → ReLU → Dropout → Linear(hidden, n_classes)
    loss   : CrossEntropyLoss with optional class weighting

Usage:
    .venv/bin/python scripts/bilstm_emotion.py \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv        data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-parametric  data/meld/parametric_prosody_test_mfa.jsonl \\
        --eval-csv         data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --label emotion --pool attention --epochs 12
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
N_PARAMETRIC = 18  # +1 for voicing flag


class MELDProsodySeqDataset(Dataset):
    def __init__(self, parametric_path: Path, csv_path: Path, label_col: str,
                 label_to_id: dict[str, int]):
        super().__init__()
        parametric: dict[str, list] = {}
        self.use_frame_f0 = False
        with parametric_path.open() as f:
            for line in f:
                d = json.loads(line)
                parametric[d["utterance_id"]] = d["syllables"]
                if d["syllables"] and "frame_f0_st" in d["syllables"][0]:
                    self.use_frame_f0 = True
        self.k_frame_f0 = 0
        if self.use_frame_f0:
            for syls in parametric.values():
                if syls:
                    self.k_frame_f0 = len(syls[0]["frame_f0_st"])
                    break
        self.feature_dim = (N_PARAMETRIC + 1) + self.k_frame_f0

        df = pd.read_csv(csv_path)
        df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
        df = df[df["utterance_id"].isin(parametric)].copy()
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        df = df[df[label_col].isin(label_to_id)]

        self.examples: list[tuple[torch.Tensor, int]] = []
        for row in df.itertuples():
            syls = parametric[row.utterance_id]
            if not syls:
                continue
            rows = []
            for s in syls:
                v = [0.0 if x is None else float(x) for x in s["vec"]]
                v.append(float(s.get("voiced_fraction", 0.0)))
                if self.use_frame_f0:
                    f0_seq = s.get("frame_f0_st") or [None] * self.k_frame_f0
                    v.extend(0.0 if x is None else float(x) for x in f0_seq)
                rows.append(v)
            self.examples.append((
                torch.tensor(rows, dtype=torch.float32),
                label_to_id[getattr(row, label_col.title())],
            ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.examples[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels_t


class AttentionPool(nn.Module):
    """Standard additive attention pooling — learn a query, score each
    timestep, softmax over valid positions, weighted sum."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  mask: (B, T) bool, True = valid
        scores = self.attn(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (x * w).sum(dim=1)  # (B, D)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int, layers: int,
                 dropout: float, pool: str):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=(dropout if layers > 1 else 0.0),
        )
        out_dim = 2 * hidden
        self.pool_kind = pool
        self.attention = AttentionPool(out_dim) if pool == "attention" else None
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x, lengths):
        x = self.feature_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        B, T, _ = out.shape
        mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.to(out.device).unsqueeze(1)
        if self.pool_kind == "attention":
            pooled = self.attention(out, mask)
        else:
            mf = mask.unsqueeze(-1).float()
            pooled = (out * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0; n = 0
    for seqs, lengths, labels in loader:
        seqs = seqs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(labels); n += len(labels)
    return total / n


@torch.no_grad()
def eval_model(model, loader, device, classes):
    model.eval()
    all_pred = []; all_true = []
    for seqs, lengths, labels in loader:
        seqs = seqs.to(device)
        pred = model(seqs, lengths).argmax(dim=-1).cpu().numpy()
        all_pred.extend(pred.tolist()); all_true.extend(labels.numpy().tolist())
    acc = accuracy_score(all_true, all_pred)
    macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return acc, macro, all_true, all_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path, required=True)
    parser.add_argument("--train-csv",        type=Path, required=True)
    parser.add_argument("--eval-parametric",  type=Path, required=True)
    parser.add_argument("--eval-csv",         type=Path, required=True)
    parser.add_argument("--label", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--pool", choices=["mean", "attention"], default="attention")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", action="store_true",
                        help="Weight loss by inverse class frequency")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  pool={args.pool}")

    label_col = "Emotion" if args.label == "emotion" else "Sentiment"
    if args.label == "emotion":
        classes = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    else:
        classes = ["negative", "neutral", "positive"]
    label_to_id = {c: i for i, c in enumerate(classes)}

    train_ds = MELDProsodySeqDataset(args.train_parametric, args.train_csv, label_col, label_to_id)
    eval_ds  = MELDProsodySeqDataset(args.eval_parametric,  args.eval_csv,  label_col, label_to_id)
    print(f"[load] train={len(train_ds)} eval={len(eval_ds)} feature_dim={train_ds.feature_dim}  "
          f"frame_f0={'yes' if train_ds.use_frame_f0 else 'no'}")

    train_counts = np.bincount([l for _, l in train_ds.examples], minlength=len(classes))
    print(f"[load] train class counts: {dict(zip(classes, train_counts.tolist()))}")
    cw = None
    if args.class_weight:
        weights = (train_counts.sum() / (len(classes) * train_counts.clip(min=1))).astype(np.float32)
        cw = torch.tensor(weights, device=device)
        print(f"[load] class weights: {dict(zip(classes, [round(float(w), 2) for w in weights]))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = BiLSTMClassifier(
        input_dim=train_ds.feature_dim, n_classes=len(classes),
        hidden=args.hidden, layers=args.layers, dropout=args.dropout, pool=args.pool,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    history = []
    best_macro = 0.0
    best_pred = None
    print(f"\n{'epoch':<6s}  {'train_loss':<11s}  {'eval_acc':<8s}  {'macro_F1':<8s}")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, macro, y_true, y_pred = eval_model(model, eval_loader, device, classes)
        history.append({"epoch": epoch, "train_loss": loss, "acc": acc, "macro_f1": macro})
        marker = ""
        if macro > best_macro:
            best_macro = macro
            best_pred = (y_true, y_pred)
            marker = "  ← best"
        print(f"{epoch:<6d}  {loss:<11.4f}  {acc:<8.4f}  {macro:<8.4f}{marker}")

    print(f"\n[best] macro_F1={best_macro:.4f}")
    if best_pred is not None:
        y_true, y_pred = best_pred
        rep = classification_report([classes[i] for i in y_true], [classes[i] for i in y_pred],
                                    labels=classes, zero_division=0, digits=3)
        print(rep)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "label": args.label, "pool": args.pool,
                "class_weight": args.class_weight,
                "feature_dim": train_ds.feature_dim,
                "frame_f0": train_ds.use_frame_f0,
                "best_macro_f1": best_macro,
                "history": history,
            }, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
