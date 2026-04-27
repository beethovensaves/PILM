# Parametric Prosody Pivot — Replacing AM/ToBI Categorical Labels with a Continuous Per-Syllable Vector

_Written 2026-04-25, after the Phase 1 closeout, between MELD download and Phase 1.5 implementation. Captures the rationale, the design space we considered, and the decision._

---

## TL;DR

Through Phase 0–1, PILM was committed to AM/ToBI categorical labels (H\*, L\*, L+H\*, …; break indices 0–4) as both training targets and downstream probes. After examining the literature on inter-annotator agreement and the parametric-prosody tradition that runs alongside ToBI (Tilt, PoLaR, INTSINT, pGSLM), we are pivoting:

- **PILM trains against a continuous 18-dim per-syllable parametric vector**, not categorical labels.
- **ToBI labels are kept, but only as downstream probe targets** — to ask "does the parametric representation recover what AM theory says it should?"
- **The supervised auto-ToBI labeler (formerly Phase 3.2)** is dropped. Replaced by a deterministic Parselmouth-driven extractor that needs no training data of its own.
- **A pGSLM-style frame-level F0 baseline** is added to Phase 5 to confirm that the 18 hand-engineered dimensions capture the linguistic prosody available in F0.

The locked design changes are formalized in revised D4, D5, D6, D7, D8, D9, D10, D18 and new D19, D20 in `docs/design_decisions.md`. This document is the long-form rationale.

---

## 1. The problem with ToBI as a training target

ToBI was designed in 1992 (Silverman et al., ICSLP) as a **transcription standard for human linguists** working on English intonation. Its goals were cross-lab reproducibility and theoretical alignment with autosegmental-metrical (AM) theory. It was never designed as an ML target. It became one by accident — because it was the only large-corpus prosodic annotation available, ML researchers picked it up and treated its categories as ground truth.

This has a real cost.

### 1.1 Inter-annotator agreement is the hard ceiling

Pitrelli, Beckman & Hirschberg (1994) is the canonical study. On the Boston University Radio Corpus, with trained ToBI annotators:

- **Accent presence/absence**: ~80–88% agreement.
- **Accent type** (H\* vs L+H\* vs L\*+H vs H+!H\* etc.): ~60–67% agreement.
- **Break index** (0/1/2/3/4): ~73–93% agreement, with the 3-vs-4 distinction (intermediate vs intonational phrase) being the most-confused.

Yoon, Cole & Hasegawa-Johnson (2004) replicated on a different corpus with similar numbers. More recent assessments (Cole & Shattuck-Hufnagel 2016) on conversational speech show *worse* agreement, especially on accent type.

If we train against ToBI labels as ground truth, our model cannot do better than the humans who produced them disagree with each other. A model achieving "95% accent-type accuracy" is either (a) hallucinating consensus where none exists, or (b) overfitting to one annotator's idiosyncrasies. The same applies to anyone we benchmark against — Roll et al. 2025's 89.8% Japanese accent accuracy is impressive but is also approaching the inter-annotator ceiling, where further progress means progress against noise.

### 1.2 The categories quantize away information

ToBI's accent categories are theoretically motivated but coarsely binned. An H\* and an L+H\* differ in F0 alignment (peak inside the syllable vs delayed peak), but the boundary between them is fuzzy in the acoustic signal — there's a continuum of peak alignments and a categorical decision is imposed by the annotator. The same applies to break indices: the difference between break-3 and break-4 is a matter of F0 reset magnitude and pause duration, both continuous quantities.

When we replace continuous F0 measurements with categorical labels, we throw away the underlying magnitudes — which is exactly the information a richer ML model could use.

### 1.3 ToBI was never claimed to be ML-suited

This is worth saying plainly: the ToBI authors never argued that ToBI was the right format for machine learning. They argued it was the right format for cross-lab linguistic transcription. Adopting it as a training target was an accident of corpus availability, not a principled design decision.

If we were to design PILM from scratch — not reusing existing infrastructure — we would not invent ToBI. We would design something parametric.

---

## 2. The parametric tradition (it's been there the whole time)

ToBI has had visible alternatives since the 1990s. Most of them have not become ML targets simply because they don't have large gold-labeled corpora. But for our purposes — where we don't need labels, only acoustic measurement — they are immediately viable.

### 2.1 Tilt (Taylor 2000)

Each F0 event is parameterized by 4 continuous values:

- **A** — amplitude (how high)
- **D** — duration
- **P** — peak position in the event window
- **T** — tilt = (rise − fall) / (rise + fall) ∈ [−1, +1], capturing the asymmetry of the rise/fall

Tilt is sparse — events occur on accented syllables only, so most syllables get zeros. It's compact (4 dims per event) and theoretically interpretable. It's also harder to feed to a transformer because the event-detection step (which syllables ARE accented?) is itself a categorical decision; you've just pushed the problem one level back.

### 2.2 PoLaR (Mahrt 2018)

Per syllable, dense:

- F0 at multiple time points within the syllable (onset, nucleus, offset)
- F0 max and min within the syllable
- Rhythm features (durations in normalized units)

Every syllable carries a vector — accented or not. This is exactly what a transformer wants: dense, continuous, per-token. PoLaR is theoretically grounded but more recent and less established.

### 2.3 INTSINT (Hirst, multiple papers from the 1990s)

F0 represented as an 8-level ordinal ladder relative to the speaker's range: T (top), H (higher), S (same), L (lower), B (bottom), M (mid), U (upstep), D (downstep). Pseudo-categorical but ordinal — derived from continuous F0 with thresholds, not from human judgment.

INTSINT is closer to ToBI in spirit but solves the inter-annotator problem because the thresholds are deterministic. The price is a coarser representation than full continuous values.

### 2.4 pGSLM frame-level F0 (Kharitonov et al. 2022)

The radical version: just keep raw F0 at frame rate (every 10ms) as a parallel input channel, plus duration. No event abstraction at all. Let the model figure out what's salient.

This is the most ML-native option but loses the syllable-anchored structure that AM theory says matters. It also requires more parameters (a CNN front-end on the F0 stream) and complicates per-syllable analysis.

### 2.5 Wightman & Ostendorf (1990s, classical)

Hand-crafted features per syllable: F0 movement, duration ratio, pause, etc. ~10–20 features. Used by classical pitch-accent classifiers. Predates the ToBI-as-ML-target tradition; explicit precedent for what we're now building.

---

## 3. The PILM parametric vector (locked in D19)

The design that fell out of this exercise combines PoLaR-style dense per-syllable F0 geometry with Tilt-style event shape parameters and explicit boundary features. All speaker-normalized.

**18 continuous dimensions per syllable**, plus 1 voicing flag carried alongside:

| Group | Dims | Purpose |
|---|---|---|
| Pitch geometry | 7 | F0 at onset / nucleus / offset / max / min, range, slope. Captures the F0 shape across the syllable. |
| Tilt-style event geometry | 4 | Peak position, rise amplitude, fall amplitude, tilt asymmetry. Captures *when* the F0 peak occurs and how symmetric the contour is. |
| Energy | 2 | RMS max and mean, speaker-z. |
| Duration | 2 | Syllable and nucleus duration, speaker-z. |
| Boundary | 3 | Pause after, final-lengthening ratio, F0 reset to next syllable. Captures phrasing structure. |
| **Voicing flag** | (+1) | Fraction of frames with reliable F0. Carried alongside; F0-derived dims are masked when this is 0. |

All values are speaker-normalized (D6): F0 in semitones relative to speaker median; energy and duration as z-scores against speaker distributions.

The full per-dim spec is in D19 (`docs/design_decisions.md`).

### 3.1 Why these specific dimensions

- **Pitch geometry (7 dims)**: PoLaR-inspired. Five "where is the F0 right now?" measurements plus two derived (range, slope). A transformer can recover any other monotonic statistic from these.
- **Tilt-style event geometry (4 dims)**: needed because pitch geometry alone doesn't disambiguate peak alignment within the syllable (H\* vs L+H\* equivalents). Tilt's `f0_peak_position_norm` and rise/fall amplitudes are precisely the AM-theoretic features the H\*/L+H\*/L\*+H distinction tries to capture, in continuous form.
- **Energy (2 dims)**: prominence isn't just F0; loudness contributes. Two dims is enough — peak and mean. Higher-order spectral stuff (centroid, tilt) is secondary; defer.
- **Duration (2 dims)**: rate is a prominence cue and a phrase-level rhythm cue. Two dims (whole-syllable + vowel-only) lets the model separate stress from final lengthening.
- **Boundary (3 dims)**: the only place where the parametric vector reaches *across* syllables. Critical because phrasing structure is the part of ToBI that's hardest to capture from per-syllable features alone.
- **Voicing flag**: F0 is undefined on voiceless segments (e.g., /s/). We mask the F0-derived dims at those positions and tell the encoder via this flag.

### 3.2 What we're not including (and why)

- **Spectral tilt / brightness** — could be added but is a third-order prominence cue. Defer.
- **Per-utterance / phrase-level features** (declination slope, mean F0, speech rate) — derivable from per-syllable + boundary + self-attention. The model can compute them itself. Adding them as explicit features risks redundancy without clear gain.
- **Frame-level F0 contour** — compared as a baseline (D20), not as input. See §5 below.

---

## 4. Architectural changes that fall out

The 18-dim parametric vector replaces the categorical accent + boundary slots in the input embedding. Cleanly, this means:

### 4.1 Input embedding (D7 revised)

Was: `[phone_embed ⊕ accent_embed ⊕ boundary_embed ⊕ continuous_proj]`

Now: `[phone_embed ⊕ syllable_param_proj]` where `syllable_param_proj` is a learned linear projection of the 18-dim vector for the syllable that contains this phone, **replicated across all phones in that syllable**.

The replication is the simplest possible coupling. It keeps the architecture phone-tokenized (no separate syllable-stream encoder) and preserves the masked-phone-prediction loss without modification. We accepted that the same parametric values appear at every phone in a syllable, which is fine because syllable-level prosody is constant across the phones it contains by definition.

### 4.2 Pretraining loss (D9 revised)

Was: masked phone + masked accent + masked break + masked continuous regression.

Now: **masked phone + masked parametric regression** (MSE on the 18 dims, BCE on voicing). Two losses instead of four. Cleaner, no inter-annotator-agreement ceiling, and the regression target is much richer per-syllable than two categorical decisions.

### 4.3 Probing (D5 revised)

ToBI labels survive as **downstream probe targets**. After pretraining, we ask: "given PILM's hidden states, can a linear or small-MLP probe recover the ToBI categories that AuToBI (or NXT-gold annotation) assigned?" If yes, the parametric representation captures what AM theory says it should — sanity check passed. If no, that's also informative — either our parametric vector is missing something or AM theory isn't reflected in the representation.

### 4.4 Auto-ToBI labeler (D10 revised)

The Roll-et-al-style supervised labeler we were going to build in Phase 3.2 is dropped. The deterministic Parselmouth-driven extractor needs no training data and runs in CPU-seconds per minute of audio. AuToBI plays the probe-target-generator role at evaluation only — converting parametric outputs to ToBI categories so we can compute a probe F1.

The community contribution (formerly "auto-ToBI labeler") becomes `pilm-prosody-frontend`: a pip package that produces the parametric vector from any wav + word-aligned text. Same role, simpler implementation, no training cost.

---

## 5. The pGSLM frame-level baseline (locked in D20)

Felipe's argument for keeping a frame-level F0 channel was the right one even if we didn't end up adopting it as the architecture: **pGSLM-style frame-level F0 contains strictly more information per second** than our 18-dim aggregated vector. We should know empirically whether that extra information is load-bearing for pragmatic inference.

So Phase 5 includes a same-architecture, same-compute control where the parametric vector is replaced by a 100-Hz F0 + energy contour processed by a 1D CNN front-end pooled to phone level.

The comparison answers a sharp question:

- **If parametric ≈ frame-level on all probes** → D19's 18 dims capture the linguistic prosody available in F0. We ship parametric for its interpretability and lower compute footprint. The Phase 6 scaling bet is on parametric.
- **If frame-level beats parametric by a meaningful margin** → we expand D19 (more dimensions, possibly micro-prosodic features) or move to a hybrid (parametric + frame-level both as input streams). This blocks Phase 6 scaling commitments until resolved.

This is the kind of ablation that's cheap to run alongside the killer experiment and would be expensive to skip and regret later. ~1 additional pretraining run at Phase 5 budget — small relative to the value of knowing.

---

## 6. Risks and what could go wrong

### 6.1 The 18 dims might not be enough

We deliberately resisted adding spectral, micro-prosodic, or phrase-level features in the first cut. If the parametric channel underperforms in Phase 4 probing — e.g., the linear probe fails to recover NXT-gold ToBI accent presence at F1 ≥ 0.75 — we revisit. Likely additions in priority order: spectral tilt, per-utterance declination slope, micro-prosodic shape parameters.

### 6.2 Replication might leak inappropriately across syllables

Per-position concatenation with replication (D4/D7) makes every phone in a syllable see the same prosody vector. That's by design. But it also means the model could learn syllable boundaries through this signal alone — which would be fine (it's a real prosodic signal) but might shortcut around the segmental tier. If Phase 4 probing shows the segmental representation has degraded, switch to cross-attention.

### 6.3 ToBI as probe might be the wrong target

We commit to ToBI as a probe target on the assumption that AM theory's categorical events are real things our parametric vector should encode. If that turns out to be the wrong frame — e.g., the parametric vector encodes prosody perfectly but doesn't cluster around ToBI categories because ToBI's categories aren't natural kinds — we'd see low probe F1 alongside good downstream task performance. That's not a problem, just informative: ToBI as a probe target was wrong, the representation is fine. Either way, we learn something.

### 6.4 The pGSLM baseline could win

If frame-level beats parametric meaningfully, the architecture revision is real (hybrid streams, more parameters, scaling-cost implications). We'd want to know this before Phase 6 commits to compute. That's exactly why the ablation lives in Phase 5 — not deferred.

---

## 7. What this changes about the killer experiment

The headline test (Phase 5) is unchanged in spirit. We still ask: does PILM, having seen prosody during training, perform better on text-only inference than a same-compute text-only baseline? The Fernyhough hypothesis stands or falls on this comparison.

What changes is the *what* of "having seen prosody": it's now a 18-dim continuous parametric vector per syllable, not a categorical accent/break-index assignment. The hypothesis test is *cleaner* — we're not asking the model to absorb noisy human-categorical decisions, we're giving it the underlying acoustic measurements and letting it find structure.

If the hypothesis is true, this should make it *easier* to demonstrate, because the signal is less noise-corrupted. If the hypothesis is false, it makes the negative result *harder* to dismiss as "just a label-noise problem."

Either way, the test is sharper. That's the point.

---

## 8. Implementation order

1. **Phase 1.5** (active, on MELD) — implement `scripts/extract_parametric_prosody.py`; run on MELD dev split; validate against AuToBI as probe target. Gate: linear probe over parametric vectors recovers AuToBI accent presence at F1 ≥ 0.65.
2. **Phase 2** (pinned on LDC) — port the same extractor to NXT data with stereo per-channel speaker baselines. Gate: linear probe recovers NXT-gold ToBI accent at F1 ≥ 0.75.
3. **Phase 3.2** — package the extractor as `pilm-prosody-frontend`.
4. **Phase 4** — pretrain PILM with the new parametric input channel.
5. **Phase 5** — killer experiment + pGSLM ablation.

The parametric pivot doesn't change the phase boundaries; it changes what happens inside each phase. The clock didn't move, but the bet got cleaner.

---

## 9. Open questions for future-Felipe

These are not blocking for Phase 1.5 but are worth tracking:

1. **Dimensionality reduction.** 18 dims per syllable is small but redundant — `f0_max_st` and `f0_rise_amplitude_st` are correlated by construction. Run PCA on the parametric vector across MELD dev; report effective rank. If effective rank < 12, prune. This is a Phase 4 cleanup task.
2. **Should `voiced_fraction` be inside the 18, not alongside?** Practical question: it's a meaningful continuous feature in its own right (creaky voice, breathiness, laughter all reduce voiced fraction). Currently outside because it has a special masking role. Could be both.
3. **Phrase-level digest features at boundaries** — if Phase 4 probing shows poor dialog-act recovery, add a 4-dim "phrase digest" at every break point (phrase mean F0, range, declination slope, speech rate). Cheap to add post-hoc.
4. **Multilingual extension.** The parametric vector is intended to be language-agnostic, but Mandarin tones add a lexical dimension that English doesn't. Phase 8 question.
5. **Validation against pGSLM.** D20 ablation answers this empirically. Until then, the parametric design is a theoretical bet.

---

## Sources

- Silverman, K., et al. (1992). *ToBI: a standard for labeling English prosody*. ICSLP.
- Pitrelli, J., Beckman, M., Hirschberg, J. (1994). *Evaluation of prosodic transcription labeling reliability in the ToBI framework*. ICSLP.
- Yoon, T., Cole, J., Hasegawa-Johnson, M. (2004). *On the edge: acoustic cues to layered prosodic domains*. ICPhS.
- Cole, J., Shattuck-Hufnagel, S. (2016). *New methods for prosodic transcription: capturing variability as a source of information*. Laboratory Phonology 7(1).
- Taylor, P. (2000). *Analysis and synthesis of intonation using the Tilt model*. JASA 107(3).
- Mahrt, T. (2018). *PoLaR: Pitch and Rhythm — a parametric prosody annotation system*. (PhD-era technical note + Praat plugin.)
- Hirst, D., Di Cristo, A. (1998). *Intonation Systems: A Survey of Twenty Languages*. Cambridge UP. (INTSINT background.)
- Wightman, C., Ostendorf, M. (1994). *Automatic labeling of prosodic patterns*. IEEE TSAP 2(4).
- Kharitonov, E., et al. (2022). *Text-Free Prosody-Aware Generative Spoken Language Modeling* (pGSLM). ACL.
- Roll, N., et al. (2025). *Prosody Labeling with Phoneme-BERT and Speech Foundation Models*. arXiv 2507.03912.
- Rosenberg, A. (2010). *AuToBI: a tool for automatic ToBI annotation*. Interspeech.
- Honorof, D., Whalen, D. (2005). *Perception of pitch location within a speaker's F0 range*. JASA. (Speaker-relative pitch perception.)
