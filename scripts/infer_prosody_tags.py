"""
Run inference with a trained wav2vec2-based prosody front-end to obtain
binary stress (σ_S/σ_U) and boundary (B0/B1) tags for an utterance.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import soundfile as sf

from models.prosody_frontend.wav2vec_frontend import (
    ProsodyFrontendConfig,
    load_processor_and_model,
)


def naive_segmenter(num_frames: int, num_segments: int) -> List[Tuple[int, int]]:
    """
    Very simple segmenter that splits frames into equal-length segments.
    This is just for quick smoke tests; for real experiments, replace with
    syllable or word-aligned spans.
    """
    if num_segments <= 0:
        raise ValueError("num_segments must be positive.")

    segment_length = max(num_frames // num_segments, 1)
    spans: List[Tuple[int, int]] = []
    start = 0
    for _ in range(num_segments - 1):
        end = min(start + segment_length, num_frames)
        spans.append((start, end))
        start = end
    spans.append((start, num_frames))
    return spans


def run_inference(
    audio_path: Path,
    model_path: Path,
    num_segments: int,
) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    config_dict = checkpoint.get("config", {})
    config = ProsodyFrontendConfig(**config_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_processor_and_model(config=config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio, sr = sf.read(audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(input_values=input_values, attention_mask=attention_mask)
        frames = encoder_outputs.last_hidden_state

    num_frames = frames.size(1)
    spans = [naive_segmenter(num_frames, num_segments)]

    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            segment_spans=spans,
        )

    stress_logits = outputs["stress_logits"]
    boundary_logits = outputs["boundary_logits"]

    stress_pred = stress_logits.argmax(dim=-1).cpu().tolist()
    boundary_pred = boundary_logits.argmax(dim=-1).cpu().tolist()

    for i, (s, b) in enumerate(zip(stress_pred, boundary_pred)):
        stress_symbol = "σ_S" if s == 1 else "σ_U"
        boundary_symbol = "B1" if b == 1 else "B0"
        print(f"Segment {i}: stress={stress_symbol}, boundary={boundary_symbol}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer prosody tags from audio.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to audio file.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model checkpoint.")
    parser.add_argument(
        "--num-segments",
        type=int,
        default=8,
        help="Number of segments to split the utterance into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(audio_path=args.audio, model_path=args.model, num_segments=args.num_segments)


if __name__ == "__main__":
    main()

