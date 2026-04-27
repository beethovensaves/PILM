# PILM — Prosody-Internalized Language Model

_Last updated: 2026-04-26._
_Read this file first. Everything else in `docs/` is appendix._

## What we're building, in one paragraph

A computational test of the Fernyhough/Fodor inner-speech-as-prosody claim.
We pretrain a transformer encoder on parallel sub-word streams: a discrete
**phone tier** (forced-aligned) and a continuous **prosody tier** (an 18-dim
parametric vector per syllable, replicated across the syllable's phones).
Modality dropout p=0.2 forces the encoder to develop genuine text-only
competence. At evaluation, we probe whether prosody-trained representations
beat same-compute text-only representations on text-only downstream tasks.
A positive result says prosody pretraining installs an inductive bias that
survives text-only inference. A negative result is publishable too.

## Architecture in a picture

```
Per phone position p:
  phone_id_p          → PhoneEmbed lookup           → 256d
  prosody_vec_p (18d) → Linear projection           → 64d
  position p          → SyllablePosFeatures(D23)    → 6d
  ↓
  concat → 326d input embedding
  ↓
  Transformer encoder (6 layers)
    each layer: self-attention with prosody-attention bias (D21):
      score(i,j) += α_h · MLP(prosody_j)
  ↓
  Two output heads:
    phone_logits_p ∈ ℝ^40  →  CrossEntropy(masked phone targets)
    prosody_pred_p ∈ ℝ^18 →  MSE(masked vector targets)
  + Modality dropout p=0.2 on the prosody slice during training
```

## Current phase

| Phase | Status | What it does |
|---|---|---|
| **0** Foundations | ✅ done | Project framing, scaffolding, lit review |
| **1** Synthetic toy | ✅ done | Architecture validated; modality collapse fixed via dropout (D9) |
| **1.5** MELD parametric | ✅ done | 18-dim vector validated; EXP-005..008 documented |
| **1.5+** MELD follow-up | ✅ done (today) | EXP-007b validations all passed; EXP-009 anger diagnostic |
| **AMI scoping** | ✅ done (today) | Annotations + audio mirror in flight; cross-corpus probe queued |
| **2** Switchboard NXT | 🅿 pinned | Waiting on UW LDC institutional access (LDC97S62 + LDC2009T26) |
| **3** Annotation tool | not started | Streamlit; deferred |
| **4** v1 pretraining | not started | The actual model — D21+D23 are new architecture as of today |
| **5** Killer experiment | not started | Phase 4 representations probed text-only on Switchboard NXT / MELD / NXT-kontrast |
| **6+** Scaling, decoder, multilingual | deferred | Conditional on Phase 5 outcome |

## What we know empirically (the bones of the architecture)

1. **Continuous vectors > discrete tokens for input.** EXP-008: ToBI categorical (15-dim) is the weakest probe (0.094 emotion). Bucketing throws away gradient information. → D22 locks continuous-only at input.
2. **18 dims is enough.** EXP-008: frame-level F0 added +0.005 emotion / regressed sentiment. Microprosody isn't load-bearing at utterance level.
3. **F0 contour features carry the signal.** EXP-007b top dims: tilt, f0_peak_pos, f0_min_st, f0_range_st, f0_nucleus_st (5 of 6). Bootstrap CIs sign-stable.
4. **Prosody is positionally localised.** EXP-007b position ablation: last-syl AUC 0.65 vs middle-syl 0.55 (+0.10). Boundary tones live where AM theory predicts. → D23 proposes explicit syllable-position features.
5. **Anger is high pitch register, not "loud + fast."** EXP-009: f0_nucleus_st alone gets AUC 0.67. Folk theory loses; data wins.
6. **MELD's `?` is a perfect text oracle.** Cross-corpus baseline: MELD text-with-punct AUC = 1.00 on yn-Q; AMI = 0.93. Words-only ~0.86–0.89 on both. The PILM thesis lands on **emotion / affect / pragmatic prosody**, not question detection — questions are too lexically anchored in any English corpus.

## The locked architectural commitments (links to `decisions.md`)

- **Input:** phone tokens + continuous 18-d parametric vector + 6-d syllable-position features. (D4, D7, D19, D22, D23.)
- **Per-speaker normalisation** of all prosody dims (D6).
- **Self-attention** with **prosody-modulated attention bias** in every layer (D21, NEW).
- **Loss:** masked phone CE + masked prosody MSE + modality dropout p=0.2 (D9).
- **No ToBI labels in training**, only as probe targets (D5).
- **Pretraining is fully unsupervised**; pragmatic labels are probing-only (D18).
- **Encoder-decoder architecture, encoder-only training in v1**; decoder activates Phase 7 (D3).
- **Probing-based evaluation, no fine-tuning** (D12).

## What's next

See `plan.md` for the full ordered list. Top three:

1. **Finish AMI v1 cross-corpus probe** — pipelines running now; tonight's deliverable. Tells us whether the +0.017 emotion uplift on MELD generalises to natural conversation.
2. **Build the Phase 4 model.** D21 attention bias + D22 continuous input + D23 position features. Encoder-only, ~50M params, train on MELD+AMI parametric outputs. ~2-week build.
3. **Phase 5 killer experiment** once Phase 4 lands and Switchboard NXT data clears LDC.

## Where to find things

- **Decisions:** `decisions.md` — D1..D23, one line each, with links to writeups.
- **Empirical findings:** `findings.md` — EXP-001..009, one paragraph each.
- **Next-up plan:** `plan.md` — the only TODO list. Replaces TODO.md, phases.md, session_handoff.md.
- **Long-form rationale:** `writeups/` — appendices, only read when the spine refers you there.
- **Archived prior versions:** `archive/` — theory_notes.md, literature_review.md, the original phases.md, etc.

If a question can't be answered from this file or the three companion canonicals, the docs need an update. They are the spine; the writeups are commentary.
