# PILM — High-level Roadmap

_Last updated: 2026-04-25_  
_Owner: Felipe_

This is a short overview. For the operational plan see `docs/phases.md`. For locked architectural choices see `docs/design_decisions.md`. For the conceptual statement see `docs/theory_notes.md`.

---

## Thesis

Humans internalize prosody during first-language acquisition (Vygotsky / Fernyhough), and silent readers project prosody onto text in a way that affects parsing and disambiguation (Fodor's Implicit Prosody Hypothesis). Text-only language models never had access to prosody and are therefore missing this structural layer.

PILM tests whether installing the layer at training time leaves a prosodic prior that persists at inference, including on text-only inputs.

## Architecture in one paragraph

For each phone position, the input embedding is the concatenation of phone identity, AM/ToBI accent and boundary categories, and continuous prosody features (log-F0 z-scored to speaker, log-energy z-scored to speaker, duration relative to speaker rate). A single transformer encoder processes the unified representation. v1 trains the encoder + classification heads only; the decoder exists in the checkpoint for Phase 7.

## Differentiation

- vs **ProsodyLM**: parallel sub-word tier, not sequential word-level token interleaving.
- vs **pGSLM**: text + prosody, not textless; pragmatic-inference target, not just generation.
- vs **SpiritLM / Moshi**: explicit AM/ToBI symbolic structure + per-speaker continuous features; not raw codec tokens.
- vs **CHiVE**: pretrained for understanding at scale, not a TTS-time prosody encoder.

## Phases (summary)

| Phase | Goal | Status | Compute | Gate |
|---|---|---|---|---|
| 0 | Foundations: lit review, decisions, scaffolding | ✅ done | Laptop | — |
| 1 | Toy synthetic world; validate harness; lock D9 dropout | ✅ done | Laptop | Architecture validated; modality collapse diagnosed |
| 2 | Switchboard NXT pipeline (parser, MFA + Parselmouth, sanity) | 🅿 pinned on LDC | CPU | 5–10 hr ToBI-aligned dialogue produced end-to-end |
| 3 | Annotation tool + auto-ToBI labeler | next | 1 GPU-day | Accent F1 ≥ 0.75, break F1 ≥ 0.85 |
| 4 | Unsupervised pretraining on Switchboard (+ MELD) | future | 1–3 GPU-days | ProsAudit ≥ best SSL baseline |
| 5 | **Killer experiment** (dialog acts, kontrast, emotion, S-vs-Q) | future | Single GPU runs | Text-only PILM beats text-only baseline on ≥1 probe |
| 6 | Scaling (~300M–1B params, real-world data tier) | future | $5k–$25k or department | Beat ProsodyLM on shared evals |
| 7 | Generation: activate decoder | future | 1–2 GPU-weeks | Demos compelling |
| 8 | Multilingual | future | varies | Cross-lingual transfer holds |
| 9 | Speculative big bets (~10 candidates) | exploratory | varies | Per-bet |

Phases 0–5 are the constrained-first proof. Phase 6+ trigger on Phase 5 success.

**Dataset stack** (per D15): Switchboard NXT (Phase 2) → MELD auto-labeled by our Phase 3 labeler (folded into Phase 4) → real-world hand-annotated set (Phase 6+).

## Falsification

If the Phase 5 killer experiment fails (no statistically significant Condition B > text-only baseline on any of the four pragmatic tasks), we publish the negative result and pivot. See `docs/theory_notes.md` §8 for the full failure-mode list.

## Reading priority

If only one document is read, read `docs/theory_notes.md`. After that, `docs/phases.md`.
