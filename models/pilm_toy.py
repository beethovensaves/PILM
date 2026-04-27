"""
Toy PILM encoder for the Phase 1 synthetic killer-experiment harness.

PHASE 1 ARTIFACT — architecture validated; the same per-position
concatenation design carries forward to the Phase 4 ~30M-param model
trained on Switchboard NXT. The toy version stays in-repo as a fast
smoke test of the architecture (run via `python -m models.pilm_toy`).

Per-position input embedding (the architectural commitment from
`docs/design_decisions.md` D7):

    [phone_embed ⊕ accent_embed ⊕ boundary_embed ⊕ continuous_proj]
              ↓
        input_proj → d_model
              ↓
     transformer encoder (pre-norm, GELU, dropout)
              ↓
       CLS-pooled → label head

The prosody slice (accent_ids, boundary_ids, continuous features) can be
zeroed at inference via `with_prosody=False`. This is the clean ablation
that powers the killer experiment: same parameters, same forward pass,
prosody masked at the input-embedding level.

A single binary `voiceless_flag` is part of the continuous channel so the
model has an explicit signal that a phone has no F0 measurement (rather
than treating "voiceless" as identical to "F0 = 0").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ToyConfig:
    # Vocabularies. Defaults match scripts/gen_synthetic_prosody.py.
    n_phones: int = 22         # 7 vowels + 15 consonants
    n_accents: int = 4         # NONE, H*, L*, L+H*
    n_boundaries: int = 5      # NONE, B1, B4_L, B4_H, B4_HH
    n_labels: int = 4          # STATEMENT, QUESTION, SURPRISED_QUESTION, FOCUS

    # Embedding dims for each input slice. Sum + d_cont feeds input_proj.
    d_phone: int = 64
    d_accent: int = 16
    d_boundary: int = 16
    d_cont: int = 16

    # Transformer.
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 256          # phones per utterance ceiling (+1 for CLS)

    # Continuous channel: [log_f0_z (0 if voiceless), voiceless_flag, energy_z, dur_rel]
    n_continuous: int = 4

    @property
    def d_input(self) -> int:
        return self.d_phone + self.d_accent + self.d_boundary + self.d_cont


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PILMToyEncoder(nn.Module):
    """Toy PILM encoder with classification head.

    Inputs (all batched):
        phone_ids:     LongTensor (B, T)
        accent_ids:    LongTensor (B, T)
        boundary_ids:  LongTensor (B, T)
        continuous:    FloatTensor (B, T, n_continuous)
        attention_mask: BoolTensor or LongTensor (B, T), 1 for real tokens.

    Output:
        logits: FloatTensor (B, n_labels)
    """

    def __init__(self, cfg: Optional[ToyConfig] = None) -> None:
        super().__init__()
        cfg = cfg or ToyConfig()
        self.cfg = cfg

        self.phone_embed = nn.Embedding(cfg.n_phones, cfg.d_phone)
        self.accent_embed = nn.Embedding(cfg.n_accents, cfg.d_accent)
        self.boundary_embed = nn.Embedding(cfg.n_boundaries, cfg.d_boundary)
        self.cont_proj = nn.Linear(cfg.n_continuous, cfg.d_cont)

        self.input_proj = nn.Linear(cfg.d_input, cfg.d_model)

        # Learned [CLS] for utterance-level pooling.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learned absolute positional embeddings.
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_labels)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        phone_ids: torch.Tensor,
        accent_ids: torch.Tensor,
        boundary_ids: torch.Tensor,
        continuous: torch.Tensor,
        attention_mask: torch.Tensor,
        with_prosody: bool = True,
    ) -> torch.Tensor:
        if not with_prosody:
            # Killer-experiment ablation: zero the prosody slice. accent/boundary
            # id 0 is "NONE" by convention (matches the synthetic vocab).
            accent_ids = torch.zeros_like(accent_ids)
            boundary_ids = torch.zeros_like(boundary_ids)
            continuous = torch.zeros_like(continuous)

        e_phone = self.phone_embed(phone_ids)            # (B, T, d_phone)
        e_accent = self.accent_embed(accent_ids)         # (B, T, d_accent)
        e_bdry = self.boundary_embed(boundary_ids)       # (B, T, d_boundary)
        e_cont = self.cont_proj(continuous)              # (B, T, d_cont)

        x = torch.cat([e_phone, e_accent, e_bdry, e_cont], dim=-1)  # (B, T, d_input)
        x = self.input_proj(x)                                       # (B, T, d_model)

        # Prepend a [CLS] token.
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)                       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                                # (B, T+1, d_model)

        # Add learned absolute positional embeddings.
        T_full = x.size(1)
        if T_full > self.cfg.max_len:
            raise ValueError(
                f"Sequence length {T_full} exceeds max_len {self.cfg.max_len}. "
                "Increase ToyConfig.max_len or shorten inputs."
            )
        positions = torch.arange(T_full, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)

        # Attention key-padding mask. PyTorch convention: True = padded (ignored).
        # Prepend a non-padded position for the CLS token.
        attention_mask = attention_mask.bool()
        cls_kept = torch.ones(B, 1, dtype=torch.bool, device=attention_mask.device)
        kept_full = torch.cat([cls_kept, attention_mask], dim=1)
        key_padding_mask = ~kept_full

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.head(x[:, 0])                                   # (B, n_labels)
        return logits


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Synthesize random tensors of the expected shape and run a forward pass.

    Verifies parameter count, output shape, and that the with_prosody=False
    ablation produces a different output (which it must, since the prosody
    slices are zeroed).
    """
    torch.manual_seed(0)
    cfg = ToyConfig()
    model = PILMToyEncoder(cfg).eval()

    B, T = 3, 24
    phone_ids = torch.randint(0, cfg.n_phones, (B, T))
    accent_ids = torch.randint(0, cfg.n_accents, (B, T))
    boundary_ids = torch.randint(0, cfg.n_boundaries, (B, T))
    continuous = torch.randn(B, T, cfg.n_continuous)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    # Mark last 4 positions of the third example as padding to exercise the mask.
    attention_mask[2, -4:] = 0

    with torch.no_grad():
        logits_full = model(phone_ids, accent_ids, boundary_ids, continuous, attention_mask, with_prosody=True)
        logits_text = model(phone_ids, accent_ids, boundary_ids, continuous, attention_mask, with_prosody=False)

    assert logits_full.shape == (B, cfg.n_labels), logits_full.shape
    assert logits_text.shape == (B, cfg.n_labels), logits_text.shape

    # The two outputs must differ — if they don't, the prosody mask isn't
    # actually changing the input embedding.
    diff = (logits_full - logits_text).abs().max().item()
    assert diff > 0, "with_prosody flag had no effect; ablation is broken"

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Output shape:     {tuple(logits_full.shape)}")
    print(f"|Δ logits| max:   {diff:.4f}  (must be > 0)")
    print(f"d_input = {cfg.d_input}, d_model = {cfg.d_model}, layers = {cfg.n_layers}")
    print("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
