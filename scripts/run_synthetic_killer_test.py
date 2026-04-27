"""
Synthetic killer-experiment harness for PILM Phase 1.

PHASE 1 ARCHIVE — kept for reproducibility and as a smoke test.
The Switchboard NXT pipeline (Phase 2+) supersedes this for the core
scientific work. See docs/experiments.md (EXP-001/002/004) and
docs/writeups/exp001_modality_collapse.md for what this harness produced.

Sweeps prosody-dropout p over a list of values. For each p:
    Train one PILMToyEncoder, where each training example has its prosody
    slice (accent_ids, boundary_ids, continuous) zeroed with probability p.

Then evaluate on test in two conditions:
    A. with_prosody=True at inference (upper bound; the model can use prosody
       if it learned to).
    B. with_prosody=False at inference (text-only inference).

Special cases:
    p = 0.0  → vanilla multimodal training (the EXP-001 setup; expected to
                fail in Condition B due to lazy-feature collapse onto prosody).
    p = 1.0  → text-only training (the EXP-001 baseline; never sees prosody).
    0 < p < 1 → modality-dropout regime (D9 default p=0.2).

The headline test is whether some 0 < p < 1 yields B-accuracy that exceeds
the p=1.0 floor — i.e., whether prosody pretraining gives text-only inference
anything beyond what a text-only baseline would have learned alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.pilm_toy import PILMToyEncoder, ToyConfig
from models.synthetic_dataset import SyntheticProsodyDataset, collate_fn, load_vocab


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def apply_prosody_dropout(
    batch: Dict[str, torch.Tensor],
    p_drop: float,
    rng: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """For each example in the batch, zero its prosody slice with probability p_drop.

    Acts as the modality-dropout regularizer described in design_decisions.md D9.
    Returns a *new* batch dict; the original is left untouched.
    """
    if p_drop <= 0.0:
        return batch

    B = batch["phone_ids"].size(0)
    drop_mask = torch.rand(B, generator=rng, device=batch["phone_ids"].device) < p_drop
    if not drop_mask.any():
        return batch

    accent_ids = batch["accent_ids"].clone()
    boundary_ids = batch["boundary_ids"].clone()
    continuous = batch["continuous"].clone()
    accent_ids[drop_mask] = 0           # NONE
    boundary_ids[drop_mask] = 0         # NONE
    continuous[drop_mask] = 0.0

    return {
        **batch,
        "accent_ids": accent_ids,
        "boundary_ids": boundary_ids,
        "continuous": continuous,
    }


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    p_drop: float,
    name: str,
    dropout_seed: int,
) -> dict:
    """Train one model with modality-dropout p_drop applied at the input level.

    The model itself is always called with with_prosody=True; the dropout is
    realized by zeroing the prosody slice of selected examples *before* the
    forward pass. This keeps the model architecture invariant to the dropout
    rate, so a single trained checkpoint can be evaluated in both Condition A
    (with_prosody=True at inference) and Condition B (with_prosody=False at
    inference) afterward.
    """
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    rng = torch.Generator(device=device)
    rng.manual_seed(dropout_seed)

    history = {"name": name, "p_drop": p_drop, "epochs": []}
    print(f"\n=== Training {name} (p_drop={p_drop:.2f}) ===")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = apply_prosody_dropout(batch, p_drop, rng)

            logits = model(
                batch["phone_ids"],
                batch["accent_ids"],
                batch["boundary_ids"],
                batch["continuous"],
                batch["attention_mask"],
                with_prosody=True,
            )
            loss = criterion(logits, batch["labels"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running_loss += loss.item() * batch["labels"].size(0)
            running_correct += (logits.argmax(-1) == batch["labels"]).sum().item()
            running_total += batch["labels"].size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        dev_acc_A = evaluate_accuracy(model, dev_loader, device, with_prosody=True)
        dev_acc_B = evaluate_accuracy(model, dev_loader, device, with_prosody=False)
        dt = time.time() - t0
        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "dev_acc_A": dev_acc_A,
            "dev_acc_B": dev_acc_B,
            "seconds": dt,
        })
        print(
            f"  ep {epoch+1:2d}/{epochs}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"devA={dev_acc_A:.4f}  devB={dev_acc_B:.4f}  ({dt:.1f}s)"
        )
    return history


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, with_prosody: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            batch["phone_ids"], batch["accent_ids"], batch["boundary_ids"],
            batch["continuous"], batch["attention_mask"],
            with_prosody=with_prosody,
        )
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / max(total, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, with_prosody: bool
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            batch["phone_ids"], batch["accent_ids"], batch["boundary_ids"],
            batch["continuous"], batch["attention_mask"],
            with_prosody=with_prosody,
        )
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def per_label_metrics(preds: np.ndarray, labels: np.ndarray, n_labels: int) -> dict:
    out = {"per_label_accuracy": {}, "support": {}}
    for c in range(n_labels):
        mask = labels == c
        n = int(mask.sum())
        out["support"][c] = n
        out["per_label_accuracy"][c] = float((preds[mask] == c).mean()) if n > 0 else float("nan")
    out["overall_accuracy"] = float((preds == labels).mean())
    return out


def bootstrap_ci(
    preds: np.ndarray, labels: np.ndarray,
    n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(preds)
    accs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        accs[i] = (preds[idx] == labels[idx]).mean()
    return float(np.percentile(accs, 100 * alpha / 2)), float(np.percentile(accs, 100 * (1 - alpha / 2)))


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cpu / cuda / mps. Auto-detect if omitted.")
    p.add_argument(
        "--prosody-dropout", type=float, nargs="+", default=[0.0, 0.2, 0.5, 1.0],
        help="Sweep values for prosody dropout probability. p=0 is vanilla multimodal; p=1 is text-only baseline.",
    )
    p.add_argument("--results-out", type=Path, default=Path("data/synthetic/killer_test_results.json"))
    return p.parse_args()


def pick_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}")

    vocab = load_vocab(args.data_dir / "vocab.json")
    n_labels = len(vocab["labels"])
    label_names = sorted(vocab["labels"], key=lambda k: vocab["labels"][k])

    train_ds = SyntheticProsodyDataset(args.data_dir / "train.jsonl", vocab)
    dev_ds = SyntheticProsodyDataset(args.data_dir / "dev.jsonl", vocab)
    test_ds = SyntheticProsodyDataset(args.data_dir / "test.jsonl", vocab)

    common_loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    dev_loader = DataLoader(dev_ds, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_kwargs)

    print(f"train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}  labels={label_names}")

    cfg = ToyConfig(
        n_phones=len(vocab["phones"]),
        n_accents=len(vocab["accents"]),
        n_boundaries=len(vocab["boundaries"]),
        n_labels=n_labels,
    )

    sweep: Dict[str, dict] = {}
    for p_drop in args.prosody_dropout:
        # Each model gets its own init seed, derived from the global seed and p_drop.
        torch.manual_seed(args.seed + int(round(p_drop * 1000)))

        model = PILMToyEncoder(cfg)
        if not sweep:  # print parameter count once
            print(f"PILMToyEncoder params: {model.num_parameters():,}")

        history = train_one(
            model, train_loader, dev_loader,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            device=device, p_drop=p_drop, name=f"PILM(p_drop={p_drop:.2f})",
            dropout_seed=args.seed + int(round(p_drop * 1000)) + 7,
        )

        # Evaluate on test in both inference conditions.
        results: dict = {"history": history, "p_drop": p_drop, "conditions": {}}
        for cond, with_prosody in [("A_with_prosody", True), ("B_text_only", False)]:
            preds, labels = collect_predictions(model, test_loader, device, with_prosody=with_prosody)
            metrics = per_label_metrics(preds, labels, n_labels)
            ci_lo, ci_hi = bootstrap_ci(preds, labels)
            cm = confusion_matrix(preds, labels, n_labels)
            results["conditions"][cond] = {
                "with_prosody_at_inference": with_prosody,
                "overall_accuracy": metrics["overall_accuracy"],
                "ci_95": [ci_lo, ci_hi],
                "per_label_accuracy": {label_names[c]: metrics["per_label_accuracy"][c] for c in range(n_labels)},
                "support":             {label_names[c]: metrics["support"][c]            for c in range(n_labels)},
                "confusion_matrix":    cm.tolist(),
            }

        sweep[f"p_drop={p_drop:.2f}"] = results

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'p_drop':>8s}  {'A acc':>8s}  {'A 95%CI':>16s}  {'B acc':>8s}  {'B 95%CI':>16s}")
    floor_B = sweep[f"p_drop={max(args.prosody_dropout):.2f}"]["conditions"]["B_text_only"]["overall_accuracy"]
    for p_drop in args.prosody_dropout:
        row = sweep[f"p_drop={p_drop:.2f}"]
        A = row["conditions"]["A_with_prosody"]
        B = row["conditions"]["B_text_only"]
        print(
            f"{p_drop:>8.2f}  "
            f"{A['overall_accuracy']:>8.4f}  ({A['ci_95'][0]:.3f}–{A['ci_95'][1]:.3f})  "
            f"{B['overall_accuracy']:>8.4f}  ({B['ci_95'][0]:.3f}–{B['ci_95'][1]:.3f})"
        )

    print("\nKiller-experiment metric: B accuracy at p_drop ∈ (0,1) vs floor (p_drop = max).")
    print(f"Floor (p_drop={max(args.prosody_dropout):.2f}, B): {floor_B:.4f}")
    for p_drop in args.prosody_dropout:
        if p_drop in (0.0, max(args.prosody_dropout)):
            continue
        B = sweep[f"p_drop={p_drop:.2f}"]["conditions"]["B_text_only"]["overall_accuracy"]
        delta = B - floor_B
        print(f"  p_drop={p_drop:.2f}:  B = {B:.4f}  ({delta:+.4f} vs floor)")

    # ---- Per-label printout ----
    print("\n" + "-" * 70)
    print("Per-label, Condition B (text-only inference)")
    print("-" * 70)
    print(f"{'p_drop':>8s}  " + "  ".join(f"{n[:10]:>10s}" for n in label_names))
    for p_drop in args.prosody_dropout:
        row = sweep[f"p_drop={p_drop:.2f}"]["conditions"]["B_text_only"]["per_label_accuracy"]
        cells = "  ".join(f"{row[n]:>10.4f}" for n in label_names)
        print(f"{p_drop:>8.2f}  {cells}")

    # ---- Save ----
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    with args.results_out.open("w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "weight_decay": args.weight_decay, "seed": args.seed, "device": str(device),
                "prosody_dropout_sweep": args.prosody_dropout,
            },
            "sweep": sweep,
        }, f, indent=2)
    print(f"\nResults saved to {args.results_out}")


if __name__ == "__main__":
    main()
