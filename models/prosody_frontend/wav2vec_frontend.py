from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
except ImportError as exc:
    raise ImportError(
        "transformers is required for the wav2vec2 prosody front-end. "
        "Install it with `pip install transformers`."
    ) from exc


@dataclass
class ProsodyFrontendConfig:
    """
    Configuration for the wav2vec-based prosody front-end.
    """

    pretrained_name: str = "facebook/wav2vec2-base"
    num_stress_labels: int = 2  # σ_S, σ_U
    num_boundary_labels: int = 2  # B0, B1
    freeze_encoder: bool = True


class Wav2VecProsodyFrontend(nn.Module):
    """
    Wav2Vec2-based prosody frontend with small classification heads for:

    - Binary stress: σ_S vs σ_U.
    - Binary boundary: B0 (no boundary) vs B1 (boundary).

    The model expects already-aligned segments and uses simple pooling over
    wav2vec2 frame-level features to obtain one vector per segment.
    """

    def __init__(self, config: Optional[ProsodyFrontendConfig] = None) -> None:
        super().__init__()
        self.config = config or ProsodyFrontendConfig()

        self.processor = Wav2Vec2Processor.from_pretrained(self.config.pretrained_name)
        self.encoder = Wav2Vec2Model.from_pretrained(self.config.pretrained_name)

        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.stress_head = nn.Linear(hidden_size, self.config.num_stress_labels)
        self.boundary_head = nn.Linear(hidden_size, self.config.num_boundary_labels)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        segment_spans: List[List[Tuple[int, int]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values:
                Tensor of shape (batch, time) with audio samples in the
                wav2vec2 expected range.
            attention_mask:
                Optional mask of shape (batch, time).
            segment_spans:
                For each batch element, a list of (start_frame, end_frame)
                indices in wav2vec2 frame space corresponding to segments
                (e.g., syllables or boundary positions).

        Returns:
            A dict with:
                - "stress_logits": (total_segments, num_stress_labels)
                - "boundary_logits": (total_segments, num_boundary_labels)
        """
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        frame_features = outputs.last_hidden_state  # (batch, frames, hidden)

        pooled_segments: List[torch.Tensor] = []
        batch_size = frame_features.size(0)

        if len(segment_spans) != batch_size:
            raise ValueError("segment_spans length must match batch size.")

        for batch_index, spans in enumerate(segment_spans):
            features = frame_features[batch_index]
            for start, end in spans:
                if end <= start:
                    raise ValueError(f"Invalid span ({start}, {end}).")
                segment_feat = features[start:end].mean(dim=0)
                pooled_segments.append(segment_feat)

        if not pooled_segments:
            raise ValueError("No segments provided in segment_spans.")

        segment_tensor = torch.stack(pooled_segments, dim=0)

        stress_logits = self.stress_head(segment_tensor)
        boundary_logits = self.boundary_head(segment_tensor)

        return {
            "stress_logits": stress_logits,
            "boundary_logits": boundary_logits,
        }


def load_processor_and_model(
    config: Optional[ProsodyFrontendConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Wav2Vec2Processor, Wav2VecProsodyFrontend]:
    """
    Convenience loader for the processor and model.
    """
    cfg = config or ProsodyFrontendConfig()
    model = Wav2VecProsodyFrontend(cfg)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = model.processor
    return processor, model


