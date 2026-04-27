"""
JSONL dataset and collator for the Phase 1 synthetic prosody data.

PHASE 1 ARTIFACT — paired with `models/pilm_toy.py` as the unit-test
dataset. The Phase 2+ Switchboard pipeline will introduce a sister
module `models/switchboard_dataset.py` that emits the same per-phone
JSONL schema, so the encoder works on real data with no architectural
change.

Reads the output of `scripts/gen_synthetic_prosody.py`. Each example becomes:

    {
        "phone_ids":    LongTensor (T,)
        "accent_ids":   LongTensor (T,)
        "boundary_ids": LongTensor (T,)
        "continuous":   FloatTensor (T, 4)   — [log_f0_z, voiceless_flag, energy_z, dur_rel]
        "label":        LongTensor scalar
    }

Voiceless handling: the generator emits `log_f0_z = null` on consonants. We
substitute 0.0 and set `voiceless_flag = 1.0`, so the model has an explicit
indicator that this phone has no F0 measurement.

`collate_fn` pads to the longest sequence in the batch and emits an
`attention_mask` (1 for real positions, 0 for padding) compatible with
`PILMToyEncoder.forward`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


def load_vocab(vocab_path: Path) -> dict:
    with vocab_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SyntheticProsodyDataset(Dataset):
    def __init__(self, jsonl_path: Path, vocab: dict) -> None:
        self.path = Path(jsonl_path)
        self.vocab = vocab
        self.examples: List[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        phones = ex["phones"]

        phone_ids = [self.vocab["phones"][p["phone"]] for p in phones]
        accent_ids = [self.vocab["accents"][p["accent"]] for p in phones]
        boundary_ids = [self.vocab["boundaries"][p["boundary"]] for p in phones]

        # Continuous: [log_f0_z (0 if voiceless), voiceless_flag, energy_z, dur_rel]
        continuous: List[List[float]] = []
        for p in phones:
            log_f0 = p["log_f0_z"] if p["log_f0_z"] is not None else 0.0
            voiceless = 0.0 if p["is_vowel"] else 1.0
            continuous.append([log_f0, voiceless, p["energy_z"], p["dur_rel"]])

        label_id = self.vocab["labels"][ex["label"]]

        return {
            "phone_ids": torch.tensor(phone_ids, dtype=torch.long),
            "accent_ids": torch.tensor(accent_ids, dtype=torch.long),
            "boundary_ids": torch.tensor(boundary_ids, dtype=torch.long),
            "continuous": torch.tensor(continuous, dtype=torch.float32),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad to the longest sequence in the batch."""
    B = len(batch)
    max_len = max(b["phone_ids"].size(0) for b in batch)

    phone_ids = torch.zeros(B, max_len, dtype=torch.long)
    accent_ids = torch.zeros(B, max_len, dtype=torch.long)
    boundary_ids = torch.zeros(B, max_len, dtype=torch.long)
    continuous = torch.zeros(B, max_len, batch[0]["continuous"].size(-1), dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["phone_ids"].size(0)
        phone_ids[i, :L] = b["phone_ids"]
        accent_ids[i, :L] = b["accent_ids"]
        boundary_ids[i, :L] = b["boundary_ids"]
        continuous[i, :L] = b["continuous"]
        attention_mask[i, :L] = 1
        labels[i] = b["label"]

    return {
        "phone_ids": phone_ids,
        "accent_ids": accent_ids,
        "boundary_ids": boundary_ids,
        "continuous": continuous,
        "attention_mask": attention_mask,
        "labels": labels,
    }
