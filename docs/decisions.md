# PILM — Decisions

_Last updated: 2026-04-26 (D21–D23 added; D5/D9/D10/D19/D20 reflect the parametric-prosody pivot of 2026-04-25)._

One line per decision. Rationale lives in `writeups/` or in `archive/design_decisions.md` (the long-form predecessor).

## Locked

| ID | Decision | Where the rationale lives |
|---|---|---|
| **D1** | Project name: PILM (Prosody-Internalized Language Model). | `archive/design_decisions.md` §D1 |
| **D2** | No Pynini / OpenFST in v1; WFST tokenizer dropped. | `archive/design_decisions.md` §D2 |
| **D3** | Encoder-decoder; v1 trains encoder + heads only. Decoder activates Phase 7. | `archive/design_decisions.md` §D3 |
| **D4** | Sub-word units: **phones** for segmental, **syllables** for prosodic. | `archive/design_decisions.md` §D4 |
| **D5** | ToBI labels are **probe targets only**, never training signal. | `writeups/parametric_prosody_pivot.md` |
| **D6** | All prosody dims are computed against a **per-speaker baseline** (median F0, energy z-score, duration ratios). | `archive/design_decisions.md` §D6 |
| **D7** | Per-position concatenation: `[phone_embed ⊕ syllable_param_proj]`; the 18-dim parametric vector is replicated across all phones in a syllable. | `archive/design_decisions.md` §D7 |
| **D8** | Killer-experiment Condition B is implemented by **zeroing the parametric slice** at every position at inference. | `archive/design_decisions.md` §D8 |
| **D9** | Pretraining loss = **masked phone CE + masked parametric vector MSE + modality dropout p=0.2**. | `archive/design_decisions.md` §D9, EXP-001/002 |
| **D10** | Prosody extraction is **deterministic** (Parselmouth-driven). AuToBI is a probe wrapper, not a labeler. | `writeups/parametric_prosody_pivot.md` |
| **D11** | _superseded by D15._ | — |
| **D12** | Evaluation is **linear probing over frozen representations**, never fine-tuning. | `archive/design_decisions.md` §D12 |
| **D13** | Compute is **constrained-first**: laptop / Colab / single-GPU until Phase 5 succeeds. | `archive/design_decisions.md` §D13 |
| **D14** | **Honesty**: raw numbers, bootstrap CIs, negative results published. | `archive/design_decisions.md` §D14 |
| **D15** | Dataset stack: **Switchboard NXT** (primary) → **MELD** (auto-labeled by Phase 3 labeler) → **real-world** (NPR/podcasts, hand-annotated). **AMI added 2026-04-26** as a free natural-conversation hedge while LDC access is pending. | `archive/design_decisions.md` §D15, `writeups/ami_scoping.md` |
| **D16** | Label space: 2-axis `speech_act × affect`, plus optional `contrastive_focus_word_idx` and `confidence`. | `archive/design_decisions.md` §D16 |
| **D17** | Annotation tool: **Streamlit + TextGrid + JSON**, with active-learning hook. | `archive/design_decisions.md` §D17 |
| **D18** | Pretraining is **fully unsupervised**; pragmatic + ToBI labels are probing-only. | `archive/design_decisions.md` §D18 |
| **D19** | **18-dim per-syllable parametric vector**: 7 pitch geometry + 4 tilt event geometry + 2 energy + 2 duration + 3 boundary. Plus a companion `voiced_fraction`. All speaker-normalised per D6. | `writeups/parametric_prosody_extractor.md` |
| **D20** | Phase 5 includes a **frame-level F0 + energy ablation** (pGSLM-style) as a same-compute control against the 18-dim parametric vector. EXP-008 result: frame-F0 adds < 0.01 — D19 captures the load-bearing F0 information at MELD's resolution. | `archive/design_decisions.md` §D20, EXP-008 |
| **D21** | **Prosody-modulated attention bias.** Each transformer layer adds `+α_h · MLP(prosody_j)` to attention scores, where the MLP is a 1-layer 18→1 net producing a per-position prosodic-salience scalar. Theoretically grounded ("prosody directs attention in humans"), cheap (~600 params/layer), falsifiable (compare to no-bias baseline), probeable (read MLP weights against named dims). | `findings.md` (rationale from EXP-007b position ablation) |
| **D22** | **No prosody tokenisation at input.** Continuous 18-dim vectors throughout. EXP-008 ToBI result is dispositive: bucketing throws away gradient information that's doing the work. A discrete prosody auxiliary output head (k-means / VQ-VAE → ~256 classes) is **deferred to Phase 7** as a possible decoder-side feature for autoregressive generation. | EXP-008 |
| **D23** | **Syllable-position features.** Each phone position carries 6 extra dims: 4 sinusoidal-relative (`sin/cos(2πk/N)`, `sin/cos(4πk/N)`) + 2 boundary-distance (`1/(1+k)`, `1/(1+(N−1−k))`). Direct architectural response to EXP-007b's position-ablation finding (last-syl AUC +0.10 over middle-syl). The boundary-distance pair is the load-bearing piece; sinusoidal additions are ablation-eligible. | EXP-007b |

## Deferred

These have been raised and explicitly postponed:

- **Cross-attention vs concatenation revisit** if Phase 4 probing shows D7 replication leaks across syllables (Phase 4+).
- **Disentanglement losses** (speaker vs prosody) if Phase 4 shows speaker info bleeding into prosody channel.
- **Discrete prosody auxiliary output head** for autoregressive prosody generation (Phase 7).
- **Decoder activation** (Phase 7).
- **Multilingual extension** (Phase 8).
- **Curated pragmatic minimal-pair corpus** as community resource (Phase 9.8).
- **Reading-time / silent-reading psycholinguistics test** (Phase 9.10).

## Rejected

- **Word-level prosody tokens (ProsodyLM-style).** Granularity argument; see `archive/theory_notes.md` §2.
- **Textless model (pGSLM-style) as the architecture.** Removes the killer experiment. (Frame-level F0 returns as a Phase 5 baseline ablation per D20.)
- **Fine-tuning-based evaluation as primary** (D12).
- **Per-utterance F0 normalisation as default** (D6).
- **Hand-written WFST tokenizer** (D2).
- **LibriTTS as v1 primary corpus** (D15 supersedes; read-speech monologue doesn't exercise pragmatic inference).
- **Conflated single-axis pragmatic labels (e.g., SURPRISED_QUESTION)** (D16).
- **Label-supervised pretraining** (D18 supersedes).
- **AM/ToBI categorical labels as training target** (D5; ToBI inter-annotator agreement caps at ~80% on accent presence / ~60% on accent type).
- **Custom auto-ToBI labeler as deliverable** (D10; rationale evaporated once D5 demoted ToBI to probe-only).
- **Prosody tokenisation at input** (D22; EXP-008 dispositive).
