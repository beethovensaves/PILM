# PILM — Phased Roadmap

_Last updated: 2026-04-25 (parametric prosody pivot — Phase 1.5 inserted; Phase 2/3/4/5 retargeted; auto-ToBI labeler dropped as standalone deliverable per revised D10)_

This document is the working roadmap for the **Prosody-Internalized Language Model (PILM)** project. It supersedes the original PAT roadmap.

The plan is organized as 10 phases, from current state through speculative long-term bets. **Phases 0–5 are the constrained-first proof.** Phases 6+ trigger only if the killer experiment in Phase 5 succeeds.

Conventions:

- **Go/no-go gates** are explicit. We do not slide silently from one phase to the next.
- **Honest cost estimates** in days/weeks of focused work and approximate compute. Expect ~1.5–2× slippage.
- ✅ = done. 🅿 = pinned (waiting on external dependency). ▶ = in progress.

---

## Phase 0 — Foundations ✅

**Goal:** finish framing the project so that every later decision has a written referent.

**Status: complete (2026-04-25).**

Delivered:
- `docs/literature_review.md`, `docs/theory_notes.md`, `docs/design_decisions.md`, `docs/phases.md`, `docs/uw_pitch.md`, `docs/TODO.md`.
- `pyproject.toml`, `scripts/setup_env.sh`, `.gitignore`.
- 21/30 priority papers in `litt/`; 9 still need browser download (Cloudflare-blocked).
- Project renamed PAT → PILM; WFST formally dropped.

---

## Phase 1 — Synthetic toy world ✅

**Goal:** validate the architecture and the killer-experiment harness on synthetic data with no audio.

**Status: complete (2026-04-25). Architecture validated, modality-collapse failure mode diagnosed and fixed.**

What was built:
- `scripts/gen_synthetic_prosody.py` — toy AM/ToBI generative model with weak/strong lexical-signal regimes.
- `models/pilm_toy.py` — 842K-param transformer encoder with per-position concatenation and an input-level prosody-mask ablation.
- `models/synthetic_dataset.py`, `scripts/run_synthetic_killer_test.py` — dataset + dropout-sweep harness with bootstrap CIs.

Key findings (full writeup in `docs/writeups/exp001_modality_collapse.md`):
- EXP-001 (vanilla, no dropout): A=100%, B=28% — **modality collapse / shortcut learning**. PILM with prosody zeroed at inference is *worse* than a text-only baseline.
- EXP-002 (dropout sweep): p=0.2 fully restores text-only competence (B=42% within CI of floor) at no upper-bound cost.
- EXP-004 (stronger lexical signal): floor rises to ~56%; PILM-B still doesn't exceed floor — synthetic data is too sparse for Fernyhough effect to show.

**Decision changes that fell out of Phase 1:**
- D9 — modality dropout p=0.2 locked into v1 pretraining.
- D16 — refactored label space (synthetic's conflated SURPRISED_QUESTION dropped; replaced with 2-axis speech_act × affect).
- D18 — pretraining is now strictly unsupervised; pragmatic labels are probing-only.

The synthetic harness is preserved in-repo as a Phase 1 reference / smoke test. It is not deleted; it is superseded for the core scientific work.

---

## Phase 1.5 — Parametric prosody pipeline validation ▶ (active, on MELD)

**Goal:** validate the deterministic parametric prosody extractor (D10, D19) on real conversational audio before NXT data lands.

**Status: in progress (2026-04-25). MELD raw tarball downloading at `data/meld/MELD.Raw.tar.gz` as of session close.**

**Why this exists:** the parametric prosody pivot (D5/D6/D9/D10/D19) replaced the supervised auto-ToBI labeler with a deterministic feature extractor. That extractor needs to be (a) implemented, (b) validated on real audio with per-speaker baselines, (c) shown to encode prosodic events recognizable to AuToBI as a probe, before we anchor Phase 4 pretraining on its outputs. MELD is the right validation corpus — unblocked from LDC, conversational dialogue, has speaker IDs and emotion labels for downstream probes.

### 1.5.1 Pipeline components

- **`scripts/extract_parametric_prosody.py`** — Parselmouth-driven F0 / voicing / energy / duration → 18-dim per-syllable parametric vector per D19. Per-speaker baselines computed from each speaker's own audio per D6.
- **`scripts/run_autobi_on_meld.py`** — runs AuToBI (Rosenberg 2010) on the same syllables. Outputs are the **probe target**, not training labels.
- **`scripts/validate_parametric_prosody.py`** — reports: (i) does a small linear probe trained on parametric vectors recover AuToBI accent labels with F1 ≥ ~0.65 on MELD? (ii) do speaker-normalized values cluster sensibly across speakers? (iii) sanity-check 10 hand-listened clips.

### 1.5.2 Data

- **MELD dev split** (~1.1 GB, ~1,109 utterances), extracted from `MELD.Raw.tar.gz` after download. Train/test splits remain unfetched until needed.

### 1.5.3 Gates

- Linear probe over parametric vectors recovers AuToBI accent presence/absence at F1 ≥ 0.65 (i.e., the parametric vector demonstrably encodes accent-related signal — we expect the AuToBI ceiling on conversational speech to be ~0.73 itself, so 0.65 means the parametric vector captures most of what AuToBI captures).
- Per-speaker normalization passes a sanity check: same parametric values for speaker A and speaker B yield perceptually similar prosodic events.
- 10-clip hand spot-check confirms no glaring extractor bugs.

### 1.5.4 Compute

- All CPU. ~1–2 days wall-clock once MELD finishes downloading.

### What does NOT happen in 1.5

- No pretraining. The encoder is not yet trained on parametric features; that's Phase 4.
- No NXT-quality ToBI labels (those land in Phase 2 with LDC access).
- No Phase 5 ablation against frame-level F0 (D20) — that comes when we have a pretrained checkpoint.

---

## Phase 2 — Switchboard NXT pipeline 🅿 (pinned on LDC access)

**Goal:** stand up the real-data pipeline on Switchboard NXT — the unique English corpus combining gold ToBI with rich dialog-act + focus/contrast annotations on conversational dialogue. Reuse the parametric prosody extractor validated in Phase 1.5; ToBI gold becomes a high-quality probe target (per D5), not a training label.

**Status: pinned on UW LDC access confirmation (2026-04-25). Parser scaffolding can begin in parallel.**

Replaces the original Phase 2 plan (build supervised auto-ToBI labeler on LibriTTS-monologue + BURNC). The supervised labeler is dropped per D10; deterministic parametric extraction replaces it.

### 2.1 Data acquisition (external dependency)

- **LDC97S62** — Switchboard-1 audio (~4 GB stereo `.sph`, one channel per speaker).
- **LDC2009T26** — NXT annotations (~50 MB XML).
- ~63 conversations have ToBI (45 Ostendorf-style + 18 Calhoun-style); ~5–10 hours of usable ToBI-aligned dialogue.
- UW co-developed NXT-Switchboard so institutional access is expected. Status: confirming.

### 2.2 Pipeline components (parallel work, scaffolds while LDC pin clears)

- **NXT XML reader** — parses the multi-layer XML (terminals, accents, breaks, kontrast, dialog acts, syllables, phones).
- **`.sph` → `.wav` converter** — `sph2pipe` or `sox`.
- **MFA forced alignment** — phone-level alignment as backup / extension to NXT phone layer.
- **Parametric prosody extractor (D10, D19)** — reuse from Phase 1.5; produce the 18-dim per-syllable parametric vector with per-speaker baseline from each side of the stereo channel.
- **PILM dataset adapter** — emits per-phone JSONL with phone-tier segmental info + replicated syllable parametric vector (per D7), matching the schema used by `models/synthetic_dataset.py` so the encoder works with no architectural change.
- **NXT ToBI labels** — extracted into a parallel JSONL stream as **probe targets only** (D5). Linear probe over PILM hidden states must predict these; the pretraining loss does not see them.
- **Sanity loop** — process one conversation end-to-end, spot-check F0 contour against the parametric vector and against annotated ToBI labels.

### 2.3 Gates

- Pipeline produces 5–10 hours of phone-aligned, parametric-prosody-extracted, ToBI-probe-target-tagged, dialog-act-tagged dialogue with reasonable per-speaker baselines.
- Linear probe over parametric vectors predicts NXT-gold ToBI accent presence/absence at F1 ≥ 0.75 (higher than the MELD/AuToBI gate in Phase 1.5 because NXT labels are gold, not auto-derived).
- Spot-check: 5 random conversations are manually verified end-to-end.

### Compute

- All CPU. ~1–2 days wall-clock once data is in hand.

---

## Phase 3 — Annotation tool + parametric extractor packaging

**Goal:** (a) build the human-in-the-loop annotation app, (b) package the parametric prosody extractor as a reusable open-source release.

### 3.1 Streamlit annotation app (per D17)

- Plays audio + waveform + spectrogram + F0 contour, all synchronized.
- Displays existing alignments and any pre-existing ToBI / dialog-act / kontrast labels.
- Allows the user to:
  - Place AM/ToBI tones on syllable nuclei (used as probe targets / spot-check, never training labels).
  - Mark break indices on word boundaries.
  - Tag utterance-level (speech_act, affect, optional focus_word_idx).
- Saves to TextGrid (Praat-compatible) and JSON (PILM-native).
- Active-learning mode: load AuToBI predictions as initial labels, user corrects. AuToBI is the pre-annotator because it's free and adequate; the user's role is to be faster than annotating from scratch, not to produce ML targets.

### 3.2 Parametric prosody extractor — pip / Hugging Face release

- Package `pilm-prosody-frontend`: Parselmouth-driven extractor producing the 18-dim per-syllable parametric vector (D19) from any wav + word-aligned text.
- Per-speaker baseline computation included; ECAPA-TDNN clustering fallback for unknown speakers (D6).
- Targets: drop-in usability — `from pilm_prosody import extract; vec = extract(wav, alignment)`.
- Replaces the previously-planned Phase 3.2 supervised auto-ToBI labeler (dropped per D10). The community-resource role is preserved; only the architecture and training-cost change.

### Gate

- Annotation tool produces a labeled file the parser can read end-to-end.
- `pilm-prosody-frontend` produces parametric vectors that match Phase 1.5/Phase 2 outputs to floating-point precision on a held-out test set.

### Compute

- All CPU. Both deliverables are laptop work.

---

## Phase 4 — Unsupervised pretraining on Switchboard

**Goal:** pretrain PILM-small on Switchboard with the modality-dropout-corrected unsupervised regime (per D9, D18).

### Architecture

- Encoder-decoder; v1 trains encoder + masked-prediction heads only.
- Encoder: ~30M params (~10× the toy). Per-position input embedding = `[phone_embed ⊕ syllable_param_proj]` (D7), where `syllable_param_proj` is the projection of the 18-dim parametric vector for the syllable containing this phone, replicated across all phones in that syllable (D4).
- Pretraining objectives (D9, revised):
  1. Masked phone prediction (cross-entropy).
  2. Masked parametric prosody regression — predict the 18-dim D19 vector at masked syllables (MSE) + voicing flag (BCE).
  3. Modality dropout p=0.2.
- Categorical accent / break-index prediction objectives are removed (D5/D9 revisions).

### Data

- Switchboard NXT (5–10 hr — entirety used as parametric-extraction input; ToBI gold reserved as probe target only).
- MELD (entire corpus once Phase 1.5 pipeline is stable; parametric features extracted with the same `pilm-prosody-frontend`).

### Probes (during training, not part of loss)

- Linear probe: predict NXT-gold ToBI accent class — confirms parametric → categorical bridge holds.
- Linear probe: predict speaker — confirms representations don't collapse identity into prosody (or do, in which case D6 disentanglement deferred-decision triggers).
- ProsAudit (de Seyssel 2023) — confirms representations are competitive with WavLM-class SSL baselines.

### Gate

- ProsAudit ≥ WavLM baseline (otherwise the architecture isn't earning its keep).
- Pretraining converges; probes recover prosody from hidden states.

### Compute

- ~1–3 GPU-days on A100 / equivalent.

---

## Phase 5 — The killer experiment (2 weeks)

**Goal:** the headline test. Does PILM's text-only inference beat a same-compute text-only baseline on pragmatic tasks?

### Setup

- Three models, matched architecture and compute:
  - **PILM-parametric** — pretrained with modality dropout p=0.2, parametric prosody (D19) channel.
  - **PILM-frame-level (D20 ablation)** — pretrained identically but with the parametric channel replaced by a 100-Hz F0 + energy contour processed by a 1D CNN front-end pooled to phone level. Tells us whether D19's hand-engineered dimensions capture the linguistic signal or whether microprosodic detail matters.
  - **Text-only baseline** — trained with prosody zeroed throughout (equivalent to dropout p=1.0).
- Three test conditions for each prosody-trained model:
  - **A**: prosody in / all channels at inference (upper bound).
  - **B**: prosody zeroed at inference (the killer condition).
  - **C**: prosody re-imagined by the decoder (deferred to Phase 7).

### Probes (linear, not fine-tuning)

1. **Switchboard NXT dialog acts** (statement / question / backchannel / repair / ...).
2. **Switchboard NXT focus/contrast (kontrast)** — predict which word(s) carry contrastive focus.
3. **MELD emotion** (categorical, 7-way) — auto-labeled subset for held-out testing only.
4. **Statement vs question on identical surface form** — minimal-pair subset constructed from NXT or curated.

### Headline comparison

- **PILM-parametric Condition B vs text-only baseline** — the primary Fernyhough test.
- If B beats baseline on at least one probe with bootstrap CI separation: Fernyhough prediction supported. Phase 6 unlocked.
- If B matches but does not exceed baseline: PILM is "merely" a good multimodal model. Negative result is publishable.

### Secondary comparison (D20 ablation)

- **PILM-parametric vs PILM-frame-level on all probes.**
- If parametric ≈ frame-level: D19's 18 dims capture the linguistic prosody available in F0; we ship parametric for its interpretability and lower compute.
- If frame-level beats parametric meaningfully: revisit D19 — either extend the dimension list or move to a hybrid input (parametric + frame-level both as streams). This blocks Phase 6 scaling commitments.

### Gate

- ≥1 statistically significant B > baseline → Phase 6.
- Otherwise → write up as benchmark contribution and pivot.

---

## Phase 6 — Scaling (6 weeks, contingent on Phase 5)

**Goal:** scale the proven architecture to a model size and data scale where it could compete with ProsodyLM and Moshi.

Triggers:
- Funded compute: UW department A100 access, or paid GPUs (Lambda Labs, RunPod).

Tasks:
- Scale to ~300M–1B params.
- Add LibriSpeech 960 + GigaSpeech subset → ~5,000 hr.
- Re-run all Phase 5 evaluations.
- Add ProsodyLM as a head-to-head comparison (replicate their reported tasks).
- Write the paper. Working title: *"Prosody Internalization: a Sub-word Parallel-Tier Speech Language Model that Retains Prosodic Inductive Bias on Text-only Inputs."*

Compute: order-of-magnitude $5k–$25k or department compute.

Gate:
- Beat ProsodyLM on its own evaluation by ≥1 task.
- Or, beat all text-only baselines on all four killer-experiment probes at scale.

---

## Phase 7 — Generation (medium-term, ~3 months)

**Goal:** activate the decoder. Now PILM doesn't only *understand* prosody, it *generates* with prosody-internalized representations.

Tasks:
- Train decoder for: (i) text generation, (ii) prosody generation given text, (iii) joint speech generation via TTS head.
- Demos:
  - Same input prompt → different prosodically-conditioned outputs.
  - Real-time dialog with appropriate prosody (compete with Moshi on prosodic appropriateness).
- Evaluation: human ratings of prosodic naturalness and pragmatic appropriateness.

Compute: 1–2 GPU-weeks on A100s.

---

## Phase 8 — Multilingual (long-term, ~6 months)

**Goal:** generalize PILM beyond English. Strongest test of the framework: does the parallel-tier abstraction transfer cleanly to a tone language?

Languages to target:
- **Mandarin**: tone language. AM-style annotation needs adaptation; tone is a primary lexical feature, not a suprasegmental overlay.
- **Yoruba** or **Igbo**: another tonal family.
- **Spanish** or **French**: stress-accent neighbor of English; easier transfer test.
- **Japanese**: pitch-accent language; intermediate case.

Tasks:
- Adapt the symbolic vocabulary per language family.
- Test cross-lingual transfer: does an English-pretrained PILM provide useful initialization for Spanish PILM?
- Find or build pragmatic-inference probes for each language.

---

## Phase 9 — Speculative big bets (year+, exploratory)

These are open research directions that PILM could support, ranked by my subjective ratio of (impact if true) / (effort to test).

### 9.1 The inner-speech computational model

Build a generative model whose internal monologue has *prosodic* structure (in the sense of Moshi's "Inner Monologue" stream, but with explicit AM/ToBI rather than raw text). Test: do the model's internal representations during silent reading-like tasks show prosodic phrasing patterns (e.g., implicit boundaries at clause junctures)? This is the strong form of the Fernyhough hypothesis as a computational thesis. **Could be a major paper or a long winter, depending.**

### 9.2 Prosody-as-cognition tokens

Introduce explicit "pragmatic state" tokens — IRONY, EMPHASIS, SURPRISE, RHETORICAL — that the model emits as part of its output stream alongside text. Treat these as first-class cognitive variables, not as labels. The hypothesis: such tokens become a controllable layer for AI behavior, useful for alignment (e.g., "be sincere," "do not signal sarcasm").

### 9.3 Embodied prosody

Pair PILM with co-occurring gesture and gaze data (from datasets like Trinity Speech-Gesture, BEAT). Hypothesis: prosodic peaks align tightly with gestural beats; a model that learns this alignment will produce more naturalistic embodied agents.

### 9.4 Therapeutic / clinical applications

Atypical prosody is a clinical marker for autism spectrum, depression, Parkinson's, schizophrenia. PILM's per-speaker baselined representations could be a sensitive detector of prosodic atypicality. This intersects with active clinical-research budgets and can fund the rest of the project.

### 9.5 Cross-species

Birdsong has prosody (in a structural sense). Cetacean and primate vocalizations have suprasegmental structure. A PILM-style architecture trained on animal communication corpora could test theories of prosodic precursors to language. High-risk, high-reward; would need a co-author with a comparative animal communication background.

### 9.6 Prosody for AI alignment and safety

Prosody is the channel through which humans signal sincerity, irony, certainty, and discomfort. A model that *generates* prosody-aware speech is also a model that can manipulate listeners via prosodic cues (sounding more confident than warranted; sounding sincere while being misleading). PILM should anticipate this: include "honest prosody" as a control axis (calibrate prosodic confidence to model uncertainty), and write up the alignment implications. **This is also a defensive necessity** — a paper on prosody-aware speech without an alignment chapter will face appropriate scrutiny.

### 9.7 Real-time prosody-aware dialog agents

The end-application demo. A voice assistant that hears your prosody, reads your emotional state, and responds with appropriately-calibrated prosody in real time. Moshi is the obvious comparison; PILM's wedge is the explicit prosodic structure (interpretable, controllable) vs. Moshi's codec-level approach.

### 9.8 Prosody minimal-pair corpus

Build and release the first large hand-curated minimal-pair pragmatic-inference corpus for English. ~5,000 sentences × 4–6 prosodic variants × pragmatic labels. This is a community resource that would outlive any single model. Potentially fundable independently (NSF, NIH, foundations).

### 9.9 Disentangled prosody control for TTS personalization

Use PILM's representations to build a TTS engine where users can control prosodic dimensions (warmth, certainty, irony, formality) independently of voice identity. Direct commercial relevance; could fund the academic work.

### 9.10 Reading-time / silent-reading psycholinguistics

Use PILM as a *cognitive model* of human silent reading. Compare PILM's surprisal at each word with human reading-time data (Dundee, GECO, MECO). Hypothesis: PILM's surprisal predicts human reading times *better* than text-only LLMs of matched size, because it captures implicit prosodic effects. This connects PILM directly to the cognitive-science community and would be a strong cross-disciplinary result.

---

## Summary diagram

```
Phase 0 ✅ ── Phase 1 ✅ ── Phase 1.5 ▶ ── Phase 2 🅿 ── Phase 3
foundations  synthetic    parametric on    NXT pipeline   annotation tool
                          MELD + AuToBI   (LDC pending)   + frontend release
                                                                 │
                                                                 ▼
                                  Phase 4 (pretrain) ── Phase 5 (KILLER EXPERIMENT)
                                                              │      includes D20
                                              gate: text-only retains prosodic prior?
                                                              │
                                          yes ◄───────────────┴───────────────► no
                                           │                                    │
                                           ▼                                    ▼
                Phase 6 (scale) → Phase 7 (gen) → Phase 8 (multi)    negative-result paper
                                           │
                                           ▼
                          Phase 9 (speculative big bets, choose 2–3)
```

---

## Publications & milestones

Each phase maps to one or more concrete external deliverables. This is what we are aiming to ship from the project, not just internal artifacts.

| Phase | Deliverable | Form | Target venue / channel | Trigger |
|---|---|---|---|---|
| 1 | Synthetic prosody-LM toolkit | Open-source repo + technical report | GitHub release + arXiv tech report (optional) | Phase 1 complete |
| 1.5 | Parametric prosody pipeline validation note | Internal writeup + repo notes | `docs/writeups/parametric_prosody_pivot.md` + `docs/experiments.md` entry | Phase 1.5 gate met on MELD |
| 2 | NXT-trained parametric extractor + ToBI-probe baseline | Open-source extractor weights + technical note | arXiv tech note + checkpoint release | Phase 2 gate hit (F1 ≥ 0.75 ToBI accent recovery via linear probe) |
| 3 | `pilm-prosody-frontend` package | pip / Hugging Face release | PyPI + HF Hub | Phase 3 complete + spot-check passes |
| 4 | Pretrained PILM-small (~30M params) checkpoint | Hugging Face release | HF Hub | Phase 4 ProsAudit gate passed |
| 5 | **Killer-experiment paper** (positive or negative result) | Full paper | NeurIPS / ACL / Interspeech | Phase 5 complete; statistical analysis done |
| 6 | Scaled PILM (~300M–1B params) + main paper | Full paper + checkpoint | Top-tier ML venue | Phase 6 complete |
| 7 | Prosody-aware generation paper + demo | Paper + interactive demo | Interspeech / ICASSP / AAAI | Phase 7 complete |
| 8 | Multilingual PILM | Paper + checkpoints (per language) | ACL / EMNLP / Interspeech | Phase 8 complete |
| 9 | Selected speculative bets | Per-bet | Per-bet | Per-bet |

**Independent paper opportunities** (could be done in parallel with the main track if compute and time allow):

- **9.1 Inner-speech computational model** — workshop-track paper at NeurIPS Cognitive Science track or CogSci main conference.
- **9.6 Prosody and AI alignment** — position paper for AIES or a NeurIPS Safety Workshop.
- **9.8 Pragmatic minimal-pair benchmark** — community resource paper at LREC-COLING or ACL Datasets and Benchmarks Track.
- **9.10 Reading-time psycholinguistics** — CogSci or *Cognition* journal paper. Strong cross-disciplinary impact.

**Rules we hold ourselves to:**

- Negative Phase 5 result is published (as a benchmark / failure-mode contribution). We do not abandon a falsified hypothesis quietly.
- The parametric prosody extractor is released as `pilm-prosody-frontend` regardless of Phase 5 outcome — it is community infrastructure either way.
- Every checkpoint we use in a paper is released publicly (subject to data-license constraints from the underlying corpora).
- One-figure summary in every paper that lay-explains the killer experiment, so a non-specialist reviewer can grasp the contribution.

---

## Decision log (what is locked vs open)

**Locked (through 2026-04-25, parametric pivot edition):**
- Architecture: encoder-decoder, per-position concatenation with replicated syllable parametric vector, single transformer stack. Encoder + heads only in v1; decoder activates Phase 7.
- Prosody representation: 18-dim per-syllable parametric vector (D19), all per-speaker normalized (D6). ToBI categorical labels are probe-only targets (D5).
- Sub-word units: phone for segmental tier, syllable for prosodic tier (D4). Replicated to phones at input.
- Pretraining objective: masked phone prediction + masked parametric vector regression + modality dropout p=0.2 (D9).
- Pretraining is fully unsupervised — no pragmatic labels, no ToBI labels seen during pretraining (D18).
- Prosody extractor: deterministic Parselmouth-driven (D10). AuToBI is a probe wrapper, not a labeler.
- pGSLM frame-level baseline ablation in Phase 5 (D20).
- WFST: dropped.
- Project name: PILM. Directory: `PILM/` (renamed 2026-04-25).
- UW pitch drafted and held in `docs/uw_pitch.md`; not sent until Phase 5 lands.

**Open:**
- Compute partner (UW department vs Lambda Labs vs other) — final choice deferred until Phase 5 result.
- Which Phase 9 bet(s) to pursue first.
- Whether i-LABS / Patricia Kuhl group / specific UW faculty are the right pitch targets — needs investigation.
