# PILM — UW Compute Pitch

_Status: draft. Holding until Phase 5 lands. Do not send before then._

_Last updated: 2026-04-25_

---

## One-paragraph pitch

The Prosody-Internalized Language Model (PILM) project tests a precise, falsifiable hypothesis at the intersection of psycholinguistics and machine learning: that pretraining a language model with parallel sub-word prosodic representations produces inductive biases that persist on text-only inputs, mirroring the implicit prosody phenomenon documented in human silent reading (Fodor 1998, 2002). If supported, this would be the first computational analogue of the Fernyhough / Fodor inner-speech-as-prosody claim, and a meaningful advance over current speech language models (ProsodyLM, pGSLM, SpiritLM, Moshi) which evaluate only with prosody available at inference. We have completed an initial proof-of-concept on a constrained pretraining setup; we are seeking compute support to scale the demonstrated result to publication-quality at ~300M–1B parameters on multi-thousand-hour speech corpora.

## What we have shown (FILL IN AFTER PHASE 5)

- _\<insert killer experiment results\>_
- _\<insert ProsAudit and downstream probing numbers\>_
- _\<insert head-to-head vs text-only baseline at small scale\>_

## What we are asking for

- ~$X k of A100 / H100 hours, or equivalent department-allocated compute, to run Phase 6 of the project plan.
- Optional: a faculty co-author with relevant background (computational psycholinguistics, speech processing, or language acquisition modeling).

## Why now

- Speech-text foundation models (Moshi, SpiritLM) are crystallizing in 2024–2025; the architectural choices being made now will define the field for several years. PILM offers a structurally different approach (parallel sub-word prosodic tier with explicit symbolic + continuous channels) that has not been tested at scale.
- Recent progress on auto-ToBI labeling (Roll et al. 2025, Phoneme-BERT + WavLM) makes the necessary symbolic supervision affordable for the first time.
- Inner speech and implicit prosody are well-replicated psycholinguistic phenomena that have not been used as training-objective design principles in language modeling.

## Why UW

- Strong cognitive-science and computational-linguistics communities; PILM intersects both.
- _\<faculty fit: TBD — to be filled with relevant UW faculty in CS, Linguistics, Psychology, or i-LABS\>_
- UW has the speech-research infrastructure (alignments, voice corpora, GPU clusters) the project needs.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Auto-ToBI labels too noisy at scale | Phase 2 fallback to AuToBI / commercial labelers; published failure mode either way. |
| Killer experiment fails at scale even if it succeeds at small scale | Negative result is publishable as a benchmark contribution to the field. |
| Compute requirements grow beyond initial estimate | Constrained-first plan (Phases 1–5) demonstrates the approach is viable before scaling spend. |
| Reviewer skepticism about "prosody internalization" framing | Killer experiment is structurally a Fodor-IPH probe; psycholinguistic precedent is robust. |

## Timeline (post-Phase-5)

- Months 1–2: scale to ~300M params, ~5,000 hr corpus.
- Month 3: full evaluation suite; head-to-head vs ProsodyLM on shared tasks.
- Month 4: paper writing.
- Month 5: submission to top venue (NeurIPS / ACL / Interspeech, depending on cycle).

## Outputs

- One main paper (peer-reviewed).
- One open-source model release (~300M params; checkpoint + tokenizer + auto-labeler).
- One auto-ToBI labeler release (potentially as a short separate paper at Interspeech / ICASSP).
- One pragmatic-inference benchmark contribution.
- Followup directions documented in `phases.md` Phase 9 (multilingual, generation, alignment, clinical).

---

## Notes for finalization

- Confirm UW faculty fit before sending. Candidates to investigate: i-LABS (Patricia Kuhl group, language acquisition); CS speech / NLP faculty (TBD); Linguistics phonetics (TBD); Psychology language and cognition.
- Customize "What we have shown" with concrete numbers from Phase 5.
- Add a one-figure summary if pitching in person.
- Consider parallel pitches to: Lambda Labs research grants, Modal AI credits, Anthropic / OpenAI academic credits, NSF SCH / SBE programs.
