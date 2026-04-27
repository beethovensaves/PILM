# PILM — Theory Notes

_Last updated: 2026-04-25_

This file is the conceptual statement of the project. It records why PILM exists, the theoretical commitments that distinguish it from adjacent work, and the empirical predictions that count as success or failure.

For citations and the broader literature map, see `literature_review.md`. For phased delivery, see `phases.md`. For locked architectural choices, see `design_decisions.md`.

---

## 1. The thesis

**Humans internalize prosody during first-language acquisition and use it as a structural layer of meaning even when they are not speaking aloud.**

This is a Vygotsky → Fernyhough claim. Inner speech is internalized social speech, and expanded inner speech retains phonological and prosodic properties of external dialogue (Alderson-Day & Fernyhough 2015). Fodor's Implicit Prosody Hypothesis (1998, 2002) provides the empirical bridge: silent readers project prosody onto text and this measurably affects parsing and disambiguation (Breen 2014).

If this is true of humans and matters for human language understanding, then text-only language models — which never had access to prosody — are missing a layer of structure that humans use. **PILM is built to test whether installing that layer at training time produces inductive biases that persist at inference, even on text-only inputs.**

This is the thesis. Everything in the architecture and the evaluation plan exists to test it.

---

## 2. Why ProsodyLM is structurally wrong

ProsodyLM (Lin et al. 2025) tokenizes as `text → word-level prosody token → text → word-level prosody token`. It treats prosody as a *sequential* element interleaved with text at the *word* level.

This is wrong for two structural reasons:

- **Granularity.** A word like "tomorrow" carries pitch movement *across its syllables* (to-MOR-row, with the contour localized on the final syllable). At word level, this contour collapses into a single tag and the tone-syllable association is lost. Autosegmental-Metrical theory (Pierrehumbert 1980; Beckman & Pierrehumbert 1986; Ladd 2008) and the ToBI annotation framework explicitly encode the sub-word association between tones and metrical positions.
- **Channel.** Prosody is not a *next* element in the segmental sequence; it is a *parallel tier* that runs concurrently with the segmental tier and binds to it via timing alignment. Tokenizing prosody in-line forces the model to learn "now I am in a text context, now I am in a prosody context" rather than "at this metrical position, the segment is X *and simultaneously* the tone is Y."

The structural fix: **phone-level parallel-tier representation.** Each phone position carries (i) phone identity, (ii) categorical AM/ToBI features (accent class, boundary class), and (iii) continuous features (log-F0 relative to speaker baseline, energy relative to speaker baseline, duration relative to speaker rate). All three are in the same input embedding at the same position.

---

## 3. Pitch is relative — the speaker-baseline commitment

A given F0 in Hz means different things in different speakers (Honorof & Whalen 2005). The same 220 Hz is high for a low male voice and low for a high female voice. *Linguistic and pragmatic interpretation depends on the F0's location within the speaker's range, not on its absolute value.*

PILM is therefore committed to per-speaker normalization on every continuous channel:

- log-F0 z-scored within speaker (or, equivalently, expressed in semitones above the speaker's median).
- Energy log-z-scored within speaker.
- Duration expressed as a ratio to the speaker's mean duration for that phone identity.

The user's example — "higher pitch at the end of *tomorrow* compared to F0, expressing surprise on top of the question" — is *only* interpretable in this framework. The "surprise" reading depends on the F0 peak being unusually high *for this speaker*, not unusually high in absolute Hz.

A non-lossy approach: where we have speaker labels (LibriTTS), use them directly. Where we don't, cluster utterances acoustically (ECAPA-TDNN) and use cluster-level baselines. We do not collapse to per-utterance baselines except as a fallback.

---

## 4. Continuous and symbolic, both — and tightly coupled

PILM holds both representations because they encode different things:

- **Symbolic (AM/ToBI categories).** Captures the categorical, phonological structure that linguistic theory has been refining for 40 years. H\*, L\*, L+H\*, etc. are not arbitrary; they correspond to perceptually distinct accent types with consistent pragmatic correlates.
- **Continuous (log-F0 z, energy z, duration ratio).** Captures the gradient detail that the categorical labels collapse. The difference between "ordinary question rise" and "surprised question rise" is a gradient F0 peak height; both are H% but the magnitude carries the affective overlay.

If we kept only one, we would either lose the linguistic grounding (continuous-only) or lose the gradient detail that powers emotion / pragmatic nuance (symbolic-only). We keep both.

The architecture choice that follows: **per-position concatenation at input.** Each phone position has its embedding fattened by the prosody slice. The model is forced to integrate both at every layer, not as a side channel that can be ignored. This is the tightest sensible coupling and the best match for the Fernyhough framing — prosody is *part of* the representation, not a separate stream attended to occasionally.

---

## 5. The killer experiment

This is the experiment whose result determines whether PILM's thesis is supported.

**Setup.** Pretrain PILM with all channels (phone + accent + continuous prosody, all at sub-word level). Pretrain a matched text-only baseline on the same data, with only the phone channel.

**Three inference conditions for PILM:**

| Condition | Input | Probe |
|---|---|---|
| A | Speech (all channels) | Pragmatic label |
| B | Text only (prosody channel zeroed) | Pragmatic label |
| C | Text only with prosody re-imagined by decoder | Pragmatic label (Phase 7) |

**Comparison:** Condition B vs text-only baseline.

**Predictions:**

- If PILM-B beats text-only baseline on pragmatic tasks: **the Fernyhough prediction is supported.** Prosody pretraining produced inductive biases that persist when prosody is unavailable at inference. This is the result.
- If PILM-B is at or below text-only baseline: PILM is "just" a multimodal model. Still useful, but the strong claim falls.

**Why this matters.** Nobody has run this experiment on speech LMs. ProsodyLM, pGSLM, SpiritLM, and Moshi all evaluate with prosody available at inference. The text-only-inference test is what separates "PILM uses the extra channel" from "PILM internalized prosodic structure into its representations."

This experiment is also Fodor-flavored: implicit prosody work shows that *humans* exhibit prosodic effects when reading silently. If PILM exhibits analogous behavior on text-only inputs, we have a computational analogue of an established psycholinguistic phenomenon. That is publishable in both ML and cognitive-science venues.

---

## 6. What "pragmatic inference" means here, scoped to v1

Pragmatics is huge. v1 narrows to four task families where prosody has the largest causal effect on interpretation:

- **Speech act type.** Statement / question / command / exclamation. The most-studied case (Wilson & Wharton 2006, the 2025 gating study).
- **Contrastive focus location.** Which word is contrastively stressed in a sentence with otherwise identical lexical content. Operationalized via L+H\* placement and minimal pairs.
- **Affect / emotion.** Categorical (neutral / happy / sad / angry / surprised) and dimensional (arousal × valence; Russell 1980; Scherer 2009). Strongly conditioned by F0 range and energy.
- **Question vs assertion with identical surface form.** "It will rain tomorrow." with falling vs rising terminal contour. The user's motivating example. Tests minimal-pair sensitivity directly.

v2 and beyond can extend to: irony / sarcasm (Cheang & Pell 2008), turn-taking and backchanneling, discourse-level focus and information structure, presupposition triggers, deixis. None of these are out of scope theoretically; they are simply not v1 scope.

---

## 7. Emotion as cognition — the framing

The user's phrase "emotion as cognition" aligns with two theoretical traditions:

- **Component process model (Scherer 2009).** Emotions are synchronized appraisal-driven responses across cognitive subsystems. Treating emotion as cognition means treating prosodic affect as an *informative* signal about the speaker's appraisal state, not as a separate noise channel layered on top of "real" content.
- **Theory of constructed emotion (Barrett 2017).** Emotions are not biological natural kinds; they are constructions from interoceptive and conceptual primitives. This means a model that learns to read prosodic affect is learning something about human conceptual structure, not just about acoustic patterns.

For PILM, this matters because the *labels* we train on (categorical emotions, dimensional affect) are themselves contestable constructs. We use them as v1 conveniences while remaining aware that the deeper goal — capturing the cognitive load that prosody carries — is broader than any specific emotion taxonomy.

---

## 8. What would falsify the thesis

The killer experiment is the central falsification, but here is a more complete failure-mode list:

1. **PILM at inference (Condition B) does not beat text-only baseline on any pragmatic task.** Thesis falsified for this regime; PILM remains a multimodal model.
2. **PILM at inference (Condition A) does not beat text-only baseline on any pragmatic task.** Architecture is broken; prosody channel is not contributing even when present.
3. **PILM does not learn meaningful AM/ToBI categories during pretraining (low probing accuracy).** Either the auto-labeler is too noisy or the model capacity is mismatched.
4. **PILM matches but does not exceed pGSLM (the closest neighbor) on shared evaluations at matched scale.** Thesis is "true but unimportant"; the parallel-tier-with-AM-symbolic angle does not pay rent over an alternative continuous-only approach.

We commit in writing now: if (1) holds at scale (Phase 6 with proper compute), we publish the negative result and pivot to whichever Phase 9 bet survives.

### Phase 1 addendum (2026-04-25): a fifth failure mode discovered

**5. Vanilla multimodal training does not produce text-only competence as a side effect.** Phase 1 (EXP-001) showed that an encoder trained with prosody available at every step collapses to a degenerate prediction (one label for everything) when prosody is zeroed at inference. Without modality dropout, "Condition B vs text-only baseline" measures the wrong thing — it measures whether multimodal training spontaneously produces text-only behavior, not whether prosody pretraining installs useful priors that transfer.

The corrective: modality dropout p=0.2 during pretraining (locked in `design_decisions.md` D9). The corrected killer-experiment question is sharper than the original: given that both PILM (with dropout) and a text-only baseline have text-only competence, **does PILM's text-only inference benefit from having seen prosody during training, beyond what the text-only training alone would have given?** That is the genuine Fernyhough/Fodor test, and it is what Phase 5 will measure on Switchboard NXT.

Full analysis in `docs/writeups/exp001_modality_collapse.md`.

---

## 9. What is genuinely new

The novel claim PILM is positioned to make is not "prosody helps speech LMs" (settled) and not "sub-word prosody beats word-level prosody" (technical, not surprising). It is:

> **A pretraining setup with parallel sub-word prosodic tiers and per-speaker continuous features produces representations whose pragmatic-inference performance retains a measurable benefit even when the prosody channel is zero at inference. This is a computational analogue of Fodor's implicit prosody phenomenon and a step toward Fernyhough's claim that prosody is internalized as part of human linguistic cognition.**

If that survives Phase 5 with statistical significance, the paper writes itself.

---

## 10. Open theoretical questions to revisit

These are not blockers for v1 but should be tracked:

- **What is the right symbolic granularity?** AM/ToBI assumes a phone/syllable-anchored tier system. Some languages (Mandarin) embed tone at the syllable level as a primary lexical feature, not as a suprasegmental overlay. Phase 8 will force this question.
- **Is per-speaker baseline enough?** Pitch perception also depends on context (preceding F0, declination). A more sophisticated framework (declination-corrected, context-relativized) may be needed. Defer to v2.
- **What is the right loss for "prosody internalization"?** v1 uses standard masked prediction + downstream probing. v2 might benefit from explicit contrastive losses on minimal pairs (same text, different prosody → different embedding).
- **Can we separate prosody from speaker identity cleanly?** Speaker identity is in the audio, and even per-speaker normalization may leave residual speaker info in the prosody features. Disentanglement losses (cf. MSR-Codec 2025) may be needed.
- **What does "prosody" even mean for written text in inference?** If PILM is given text-only input, it has no prosody to encode. The thesis is that pretraining left a prosody-shaped *prior* in the encoder's parameters, such that text-only inputs activate prosodically-informed representations. This is a strong claim that needs careful operationalization at evaluation time.
