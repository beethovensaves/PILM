"""
bi-LSTM probe over per-syllable parametric vectors → predict whether the
utterance is a question. Direct comparison to scripts/predict_question_from_prosody.py
which used pooled-LR on the same input. Tests the claim that pooling is
the dominant performance bottleneck for prosody-only probing.

Architecture:
    input  : (batch, T, 19)                 # 18 parametric dims + voicing flag
    bi-LSTM: hidden=64, layers=2 → output (batch, T, 128)
    pool   : masked mean over valid positions → (batch, 128)
    dense  : Linear(128, 1) → logit
    loss   : BCEWithLogitsLoss
    opt    : AdamW(lr=1e-3, weight_decay=1e-4)

Usage:
    .venv/bin/python scripts/bilstm_question_probe.py \\
        --train-parametric data/meld/parametric_prosody_train_mfa.jsonl \\
        --train-csv        data/meld/MELD.Raw/train_sent_emo_cleaned.csv \\
        --eval-parametric  data/meld/parametric_prosody_test_mfa.jsonl \\
        --eval-csv         data/meld/MELD.Raw/test_sent_emo_cleaned.csv \\
        --epochs 10
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

WH_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}
N_DIMS = 18


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MELDProsodySequenceDataset(Dataset):
    """One example = one utterance, returned as
    (vec_seq: Tensor[T, 19], length: int, label: 0/1)."""

    def __init__(self, parametric_path: Path, csv_path: Path, yn_only: bool):
        super().__init__()
        # Load parametric
        parametric: dict[str, list] = {}
        with parametric_path.open() as f:
            for line in f:
                d = json.loads(line)
                parametric[d["utterance_id"]] = d["syllables"]
        # Load CSV with question labels
        df = pd.read_csv(csv_path)
        df["utterance_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
        df = df[df["utterance_id"].isin(parametric)].copy()

        self.examples: list[tuple[torch.Tensor, int]] = []
        n_dropped = 0
        for row in df.itertuples():
            utt = str(row.Utterance)
            has_q = "?" in utt
            m = re.search(r"[A-Za-z]+", utt)
            first_word = m.group(0).lower() if m else ""
            is_wh_q = has_q and first_word in WH_WORDS
            if yn_only and is_wh_q:
                continue  # drop wh-questions
            label = 1 if (has_q and not (yn_only and is_wh_q)) else 0

            syls = parametric[row.utterance_id]
            if not syls:
                n_dropped += 1
                continue
            # Build (T, 19) tensor: 18 parametric dims + voiced_fraction
            vec_rows = []
            for s in syls:
                v = [0.0 if x is None else float(x) for x in s["vec"]]
                v.append(float(s.get("voiced_fraction", 0.0)))
                vec_rows.append(v)
            seq = torch.tensor(vec_rows, dtype=torch.float32)  # (T, 19)
            self.examples.append((seq, label))

        if n_dropped:
            print(f"  dropped {n_dropped} utterances with no syllables")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.examples[idx]


def collate(batch: list[tuple[torch.Tensor, int]]):
    """Pad variable-length sequences. Returns (padded_seqs, lengths, labels)."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # (B, T_max, 19)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiLSTMProbe(nn.Module):
    """bi-LSTM → masked mean pool → linear → logit."""

    def __init__(self, input_dim: int = 19, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        # bidirectional → 2 * hidden
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)  lengths: (B,)
        x = self.feature_norm(x)
        # Pack so the LSTM only computes over valid positions
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        # Unpack to (B, T, 2*hidden) with zeros at padded positions
        from torch.nn.utils.rnn import pad_packed_sequence
        out, _ = pad_packed_sequence(out, batch_first=True)

        # Masked mean pool over valid positions
        B, T, H = out.shape
        mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.to(out.device).unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom  # (B, 2*hidden)

        return self.head(pooled).squeeze(-1)  # (B,) — logits


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for seqs, lengths, labels in loader:
        seqs = seqs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def eval_model(model, loader, device) -> tuple[float, float, float, float]:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    for seqs, lengths, labels in loader:
        seqs = seqs.to(device)
        logits = model(seqs, lengths).cpu().numpy()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.numpy().tolist())
    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, probs)
    acc = accuracy_score(all_labels, preds)
    macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    qf1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
    return acc, macro, qf1, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parametric", type=Path, required=True)
    parser.add_argument("--train-csv",        type=Path, required=True)
    parser.add_argument("--eval-parametric",  type=Path, required=True)
    parser.add_argument("--eval-csv",         type=Path, required=True)
    parser.add_argument("--yn-only", action="store_true",
                        help="Drop wh-questions; positives are yes/no questions only.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[device] {device}")

    print(f"[load] train {args.train_parametric.name}")
    train_ds = MELDProsodySequenceDataset(args.train_parametric, args.train_csv, args.yn_only)
    print(f"[load] eval  {args.eval_parametric.name}")
    eval_ds  = MELDProsodySequenceDataset(args.eval_parametric,  args.eval_csv,  args.yn_only)
    n_pos_tr = sum(1 for _, l in train_ds.examples if l == 1)
    n_pos_te = sum(1 for _, l in eval_ds.examples  if l == 1)
    print(f"  train: {len(train_ds)} utts, {n_pos_tr} questions ({100*n_pos_tr/len(train_ds):.1f}%)")
    print(f"  eval:  {len(eval_ds)}  utts, {n_pos_te} questions ({100*n_pos_te/len(eval_ds):.1f}%)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0)

    model = BiLSTMProbe(input_dim=N_DIMS + 1, hidden=args.hidden,
                        layers=args.layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Class weighting helps because positives are minority
    pos_weight = torch.tensor((len(train_ds) - n_pos_tr) / max(1, n_pos_tr),
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_auc = 0.0
    print()
    print(f"{'epoch':<6s}  {'train_loss':<11s}  {'eval_acc':<8s}  {'macro_F1':<8s}  "
          f"{'q_F1':<8s}  {'AUC':<6s}")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, macro, qf1, auc = eval_model(model, eval_loader, device)
        history.append({"epoch": epoch, "train_loss": loss,
                        "eval_acc": acc, "macro_f1": macro, "q_f1": qf1, "auc": auc})
        marker = ""
        if auc > best_auc:
            best_auc = auc
            marker = "  ← best AUC"
        print(f"{epoch:<6d}  {loss:<11.4f}  {acc:<8.4f}  {macro:<8.4f}  {qf1:<8.4f}  {auc:<6.4f}{marker}")

    print(f"\n[best] AUC={best_auc:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "yn_only": args.yn_only,
                "best_auc": best_auc,
                "final_epoch": history[-1],
                "history": history,
                "config": {
                    "hidden": args.hidden, "layers": args.layers,
                    "dropout": args.dropout, "lr": args.lr,
                    "batch_size": args.batch_size, "epochs": args.epochs,
                },
            }, f, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
