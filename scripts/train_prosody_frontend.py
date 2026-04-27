"""
Train a tiny wav2vec2-based prosody front-end for:

- Binary stress: σ_S vs σ_U.
- Binary boundary: B0 vs B1.

This script assumes you have:
- A manifest file listing utterances and paths to audio + aligned segment labels.
- For v1, you can use pseudo-labels derived from text and pauses.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.prosody_frontend.wav2vec_frontend import (
    ProsodyFrontendConfig,
    load_processor_and_model,
)


@dataclass
class SegmentLabel:
    """
    Metadata for a single segment (e.g., syllable) within an utterance.
    """

    start_frame: int
    end_frame: int
    stress_label: int  # 0 or 1
    boundary_label: int  # 0 or 1


class ProsodyDataset(Dataset):
    """
    Minimal dataset that reads a manifest of utterances and associated
    segment labels. The manifest format is intentionally simple:

    One JSON line per utterance:
    {
        "audio_path": "/path/to/audio.wav",
        "segment_spans": [[start_frame, end_frame], ...],
        "stress_labels": [0, 1, ...],
        "boundary_labels": [0, 1, ...]
    }

    For v1 you can generate this manifest with a separate preprocessing script.
    """

    def __init__(self, manifest_path: Path, processor) -> None:
        import json

        self.processor = processor
        self.examples: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import soundfile as sf

        ex = self.examples[idx]
        audio_path = Path(ex["audio_path"])
        audio, sr = sf.read(audio_path)

        processed = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )

        spans = [(int(s), int(e)) for s, e in ex["segment_spans"]]
        stress_labels = torch.tensor(ex["stress_labels"], dtype=torch.long)
        boundary_labels = torch.tensor(ex["boundary_labels"], dtype=torch.long)

        return {
            "input_values": processed.input_values[0],
            "attention_mask": processed.attention_mask[0],
            "segment_spans": spans,
            "stress_labels": stress_labels,
            "boundary_labels": boundary_labels,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collator for variable-length audio.
    """
    input_values = [item["input_values"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    segment_spans = [item["segment_spans"] for item in batch]
    stress_labels = [item["stress_labels"] for item in batch]
    boundary_labels = [item["boundary_labels"] for item in batch]

    input_values = nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)

    return {
        "input_values": input_values,
        "attention_mask": attention_masks,
        "segment_spans": segment_spans,
        "stress_labels": stress_labels,
        "boundary_labels": boundary_labels,
    }


def train(
    train_manifest: Path,
    output_dir: Path,
    config: ProsodyFrontendConfig,
    batch_size: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    processor, model = load_processor_and_model(config=config, device=device)
    dataset = ProsodyDataset(train_manifest, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_spans = batch["segment_spans"]

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                segment_spans=segment_spans,
            )

            stress_logits = outputs["stress_logits"]
            boundary_logits = outputs["boundary_logits"]

            stress_labels = torch.cat(batch["stress_labels"]).to(device)
            boundary_labels = torch.cat(batch["boundary_labels"]).to(device)

            loss_stress = criterion(stress_logits, stress_labels)
            loss_boundary = criterion(boundary_logits, boundary_labels)
            loss = loss_stress + loss_boundary

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "prosody_frontend.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wav2vec2-based prosody front-end.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Path to JSONL manifest.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save model.")
    parser.add_argument("--pretrained-name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-freeze-encoder", action="store_true", help="Do not freeze wav2vec encoder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ProsodyFrontendConfig(
        pretrained_name=args.pretrained_name,
        freeze_encoder=not args.no_freeze_encoder,
    )
    train(
        train_manifest=args.train_manifest,
        output_dir=args.output_dir,
        config=config,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    main()

