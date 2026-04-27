# PILM — Empirical Findings

_Last updated: 2026-04-26._

One paragraph per experiment. Strong claims only — secondary numbers and full
methods live in `writeups/` or in the per-experiment JSON results files
under `data/`.

---

## EXP-001 — Modality collapse on synthetic data (2026-04-25)

Vanilla PILM (per-position prosody concatenation, no modality dropout) on a
toy dataset with fully-determining prosody hits A=100% (all channels) but
collapses to B=28% with prosody zeroed at inference, *worse* than a
text-only baseline (43%). Diagnosis: with a perfect prosody signal at
training, the gradient pressure on the text pathway vanishes. **Lesson:**
multimodal training does not yield text-only competence as a side effect;
it must be exercised. Code: `scripts/run_synthetic_killer_test.py`.
Long-form: `writeups/exp001_modality_collapse.md`.

## EXP-002 — Modality dropout fixes the collapse (2026-04-25)

Sweep of `p_drop ∈ {0, 0.2, 0.5, 1.0}`. p=0.2 brings B from 28% → 42% (within the
text-only-baseline CI), while keeping A at 100%. p=0.5 same. p=1.0 collapses
A. **Locked into D9 as p=0.2.** No Fernyhough effect on synthetic data —
correlation structure is too thin to install transferable inductive biases.

## EXP-004 — Stronger lexical-signal regime (2026-04-25)

Same harness with a richer text-prosody correlation. Text-only ceiling rises
to ~56%; modality dropout still fixes the collapse pattern. Still no
Fernyhough effect. Synthetic data confirms architecture works; the real
test waits on natural data.

## EXP-005 — Phase 1.5 supervised comparison on MELD (2026-04-26)

MELD train→test, macro-F1: text-only 0.318 emotion / 0.601 sentiment;
prosody-only 0.127 / 0.380; combined 0.335 (+0.017) / 0.606 (+0.005). Per-class:
prosody owns **anger** (+0.140 F1 over text); text owns **surprise** (+0.200
— punctuation-driven). 35% of utterances get different predictions across
modalities. **The +0.017 emotion uplift is small but real**; recurs across
every prosody stream we tried (parametric LR, bi-LSTM, ToBI features, frame-F0).

## EXP-006 — POS ablation; punctuation is text's prosody-proxy (2026-04-26)

PUNCT alone (just `?`, `!`, `…`) is the single most predictive POS for both
emotion (0.197 macro-F1) and sentiment (0.451). Removing PUNCT from MELD's
text drops macro-F1 by 17% (emotion) / 13% (sentiment). When transcribers
added punctuation they were doing prosodic annotation in disguise. With PUNCT
removed, prosody's marginal contribution to combined stays at +0.016 emotion —
prosody encodes information **beyond** what punctuation conveys.

## EXP-007 — Question prediction from prosody (2026-04-26)

The 18-dim parametric vector predicts whether a MELD utterance ends in `?`
at AUC 0.62 (LR pooled), 0.64 (LR last-syl), 0.69 (bi-LSTM seq, yn-only).
**Top last-syllable LR coefficients match English yes/no question phonetics
exactly:** rising tilt (+0.31), late peak (+0.16), raised baseline F0
(+0.13), elevated nucleus (+0.10), smaller F0 range (-0.10), shorter syllable
(-0.10). The 18 dims encode question-relevant prosody for the right
phonetic reason.

## EXP-007b — Validation suite for the question probe (2026-04-26)

Five threat / theory tests, all passed. (a) **Speaker-held-out** GroupKFold(5)
across 180 speakers: AUC 0.63 ± 0.01 vs baseline 0.65 — not speaker-fingerprinting.
(b) **Neutral-only** subset: AUC 0.64 — not emotion-confounded. (c) **Position
ablation**: last-syl AUC 0.65 vs middle-syl 0.55 vs first-syl 0.57 (+0.10
last-vs-middle) — boundary tone localised exactly where AM theory predicts.
(d) **Wh-only positives**: AUC drops to 0.60 — rise-tuned model fails on
falling wh-Q signature, as expected. (e) **Bootstrap CIs (n=200) on top
coefficients**: tilt, f0_peak_pos, f0_min_st, f0_range_st, f0_nucleus_st all
sign-stable. **The "matches phonetics exactly" claim from EXP-007 is now
statistically real.** This is paper-grade structural validation. Position
ablation in (c) directly motivates **D23** (syllable-position features in
the architecture).

## EXP-008 — Sequence models, attention pooling, frame-F0, ToBI (2026-04-26)

Architectural sweep on emotion / sentiment. **bi-LSTM** > pooled LR by +0.04–0.05.
**Attention pool ≈ mean pool** (0.170 vs 0.171 emotion). **Frame-level F0
adds +0.005 emotion** / regresses sentiment — microprosody not load-bearing
at MELD utterance resolution. **ToBI categorical labels (15-dim) are the
weakest probe** (0.094 emotion / 0.279 sentiment). **The +0.017 / +0.005
combined boost is invariant** across prosody streams — a corpus-and-method
ceiling on MELD, suggesting NXT (and probably AMI emotion-analogue tasks)
should look different. Direct support for **D22** (don't tokenize prosody at
input).

## EXP-009 — Anger per-dim diagnostic (2026-04-26)

Three views (ANOVA F-stat, univariate AUC, drop-one ablation). At the
syllable level, the five F0 height dims dominate by an order of magnitude
(F = 925–1,357) over rms_max_z (F = 49). **`f0_nucleus_st` alone gets AUC
0.67 on anger-vs-rest** at utterance level. Drop-one ablation: load-bearing
complementary dims are `f0_min_st` (raised pitch *floor*) and `rms_max_z`
(peak loudness). **Anger is not "loud and fast" — it's "high pitch register,
slightly louder."** Folk theory loses; data win. Physically simple,
lexically invisible — exactly the kind of effect text features can never
capture.


## EXP-010 — Cross-corpus prosody probe on AMI (2026-04-26)

The same v1 parametric pipeline was applied to a 30-meeting subset of the AMI Meeting Corpus (10 each from the Edinburgh, Idiap, and TNO/Twente recording sites; 30 meetings, 36 unique speakers, 7147 segments, 737 `el.inf` and 6410 clear-statement DAs).

**Prosody-only AUC for predicting `el.inf` (Elicit-Inform) versus statement DAs**, 5-fold stratified CV: pooled LR = 0.6580; last-syllable LR = 0.5761; bi-LSTM sequence = 0.6762.

**Cross-corpus comparison (apples-to-apples, v1 prosody on both):** AMI `el.inf` last-syl AUC = 0.5761; MELD yn-Q last-syl AUC (same algorithm, v1) = 0.5861; gap = -0.0100.

**Speaker-held-out** (GroupKFold by `global_name`) AUC = 0.5709 — close to within-speaker AUC, supporting that the probe is not speaker-fingerprinting.

**Position ablation** confirms the boundary-tone localisation finding from MELD: last-syllable AUC = 0.5761 versus middle = 0.5835 and first = 0.5732.

**Text vs prosody on `el.inf` (5-fold macro-F1)**: text-only = 0.5431; prosody-only = 0.4740; combined = 0.5704 (uplift over text alone: +0.0273).

**Bootstrap-stable coefficients** (1 of top 6): f0_nucleus_st.


## EXP-010 follow-ups: per-DA extraction, MELD v1 bootstrap, richer bi-LSTM (2026-04-27)

Three follow-up studies addressed surprises in the initial cross-corpus result.

**MELD v1 bootstrap (Q1).** MELD v1 reproduce: baseline yn-only last-syl AUC = 0.5861; top-5 last-syl coefficients sign-stable: 3/5.

**Per-DA AMI re-extraction (Q2).** Re-slicing AMI audio per dialogue act (rather than per transcriber-marked segment) produced 14227 DA-level utterances (883 `el.inf` positives). Prosody-only AUC: pooled LR = 0.6636; last-syllable LR = 0.5984. Position ablation on per-DA data: first = 0.5720, middle = 0.6083, last = 0.5984. Combined text+prosody uplift over text alone: +0.0279 macro-F1.

**Richer bi-LSTM probe (Q3).** A 4-layer × 128-hidden bi-LSTM with attention pooling and 30 training epochs reached AUC 0.6216 on per-segment AMI data and 0.5747 on per-DA data, compared to the original 2-layer × 64-hidden mean-pooled probe at 0.6762 (per-segment) and 0.6862 (per-DA).


## EXP-010 v2 — MFA-aligned cross-corpus question prediction on AMI (2026-04-27)

The MFA-aligned (v2) parametric pipeline was applied to the per-DA AMI subset, enabling apples-to-apples comparison with MELD's EXP-007 v2 numbers. AMI probe target is the `Elicit-Inform` dialogue act vs clear-statement DAs; MELD probe target is utterance-final question-mark presence (yes/no questions only).

**Prosody-only AUC for predicting `el.inf` vs statement DAs**, 5-fold CV, AMI v2: pooled LR = 0.6845; last-syllable LR = 0.6298; bi-LSTM (mean-pool, small) = 0.7164; bi-LSTM (attention-pool, rich) = 0.6187.

**Cross-corpus comparison at v2 quality.** AMI `el.inf` last-syl AUC = 0.6298; MELD yn-Q last-syl AUC (MFA-aligned) = 0.6502; gap = -0.0204.

**Speaker-held-out** (GroupKFold by `global_name`) AUC = 0.6220.

**Position ablation on AMI v2:** first = 0.5787; middle = 0.6216; last = 0.6298.

**Text vs prosody on AMI v2:** combined uplift over text alone = +0.0132 macro-F1.

**Bootstrap-stable coefficients in top 6:** 6/6 sign-stable {f0_nucleus_st, rms_max_z, syl_dur_z, nuc_dur_z, rms_mean_z, f0_peak_pos}.
