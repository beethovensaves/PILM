# PILM — Design Decisions

_Last updated: 2026-04-25 (parametric prosody pivot — D4, D5, D6, D7, D8, D9, D10, D18 revised; D19, D20 added)_

This file records architectural and methodological decisions that are **locked** for v1, with the reasoning behind each. Open questions live in `phases.md` (decision log section).

For the rationale behind the parametric prosody pivot specifically, see `docs/writeups/parametric_prosody_pivot.md`.

---

## Locked decisions

### D1. Project name

- **Project:** Prosody-Internalized Language Model (PILM).
- **Directory:** `PILM/`. (Renamed from `PAT/` on 2026-04-25.)
- **Rationale.** "PAT" was committed to a Pynini WFST architecture that is no longer central. PILM names the actual scientific commitment: prosody internalization, in the Fernyhough/Fodor sense.

### D2. WFST dropped

- **Decision.** No Pynini / OpenFST in v1.
- **Rationale.** The original PAT framing assumed a hand-written WFST as the tokenizer. Given the new framing (parallel sub-word tiers, continuous + symbolic features), the WFST has no operational role beyond what an embedding lookup provides. We may revisit if we ever need a deterministic compiled accent grammar.
- **Cleanup.** `fst/` directory archived with note; not deleted in case we revisit.

### D3. Architecture: encoder-decoder, encoder-only training in v1

- **Decision.** Architecture is encoder-decoder. v1 trains the **encoder + classification heads only**; the decoder exists in the checkpoint but is not active until Phase 7.
- **Rationale.** Encoder-decoder is a flexible foundation for both understanding (Phase 5) and generation (Phase 7). Training only the encoder in v1 keeps compute contained and aligns with the Phase 5 killer-experiment goal, which is a probing experiment over hidden states.
- **Alternative considered.** Decoder-only (GPT-style) is the modern default. Rejected because (a) the killer experiment is naturally an encoder-side probing test, and (b) the eventual generation use cases (TTS, prosody-aware text gen) split cleanly across two streams that an encoder-decoder serves well.

### D4. Sub-word unit: hybrid — phone for segmental, syllable for prosodic

- **Decision.** Two alignment units, used in parallel:
  - **Segmental tier** is anchored at the **phone** (unchanged from before). Phones come from forced alignment.
  - **Prosodic tier** is anchored at the **syllable**. The 18-dim parametric prosody vector (D19) is computed once per syllable.
- **Combination at input.** The syllable parametric vector is **replicated across all phones in the syllable**, then concatenated per-phone with the segmental tier (D7). This keeps the architecture phone-tokenized while letting prosody live at its natural unit.
- **Syllable boundaries** are derived from phone alignment plus syllabification rules (CMUdict for English).
- **Rationale.** Prosodic events (peaks, boundaries, lengthening) are syllable-aligned in AM theory and in the human perceptual literature. Anchoring prosody at phone level forces us to spread or repeat values awkwardly. Anchoring it at the syllable matches the physics of the signal. The replication-to-phones pattern lets us keep phone-tokenized self-attention with no extra encoder branch.
- **Voiceless segments.** F0 is masked (not zero-filled) at the phone level via the voicing-fraction component of the syllable parametric vector (D19); the encoder receives an explicit voicing indicator.
- **Alternative considered.** Cross-attention between a phone-stream and a syllable-stream. Rejected for v1 because it adds parameters and complicates the modality-dropout ablation. Revisit if Phase 4 probing shows the replication pattern leaks information across syllable boundaries inappropriately.

### D5. ToBI labels: probe target only, never training signal

- **Decision.** AM/ToBI categorical labels (pitch accents H\*/L\*/L+H\*/…, break indices 0–4) are **not** used as PILM training targets. They serve **only as a downstream probe target** during evaluation.
- **What this means concretely:**
  - The pretraining loss (D9) does not include masked-accent or masked-break-index categorical prediction.
  - The encoder is never given a ToBI label as input.
  - At evaluation time, we ask: "given the parametric prosody representation that PILM learned, can a small linear/MLP probe recover the ToBI labels a human would have assigned?" Recovery is a sanity check that our parametric vector encodes what AM theory says it should.
- **Rationale.** ToBI inter-annotator agreement is ~80% on accent presence/absence and ~60% on accent type (Pitrelli et al. 1994; Yoon et al. 2004). Using ToBI as a training target caps any model at the human-agreement ceiling and bakes annotation noise into the representation. ToBI was designed in 1992 as a transcription standard for human linguists, not as an ML target — adopting it as ground truth was an accident of corpus availability, not a principled choice. Replacing categorical labels with a parametric vector (D19) avoids the agreement-ceiling problem and gives the model a richer, finer-grained, ML-native signal.
- **What we keep from ToBI.** The *theoretical* commitments (AM theory of intonation: tonal targets aligned to syllable nuclei, hierarchical phrasing, speaker-relative pitch perception) inform the parametric vector design (D19). We use ToBI's framework, just not its labels.
- **Probe wrapper.** AuToBI (Rosenberg 2010) is the categorical decoder we use to convert parametric outputs into ToBI categories for the probe. AuToBI runs once on each evaluation set and its outputs become probe targets, never inputs.
- **Replaces.** Earlier D5 commitment to AM/ToBI as the symbolic vocabulary. Earlier formulation moved to Decisions explicitly rejected.
- **Extension.** Language-specific probes may swap in Mandarin tone categories etc. in Phase 8; the parametric vector itself is language-agnostic.

### D6. Per-speaker baselining (still load-bearing)

- **Decision.** All quantities in the parametric prosody vector (D19) that have a meaningful speaker-relative interpretation are computed against a **per-speaker baseline**:
  - F0 expressed as **semitones relative to the speaker's median F0**.
  - Energy z-scored within speaker.
  - Duration ratios computed against speaker-specific phone-identity / syllable-duration distributions.
- **Speaker baseline computation.** Non-lossy:
  - Where speaker labels are available (Switchboard NXT stereo, MELD speaker IDs, LibriTTS), compute baseline directly.
  - Where they are not, cluster utterances with ECAPA-TDNN embeddings and use cluster-level baselines.
  - Per-utterance fallback only as last resort.
- **Rationale.** Pitch perception is relative to speaker range (Honorof & Whalen 2005). H\* perceived as "high" in a low-pitched speaker is a different acoustic event than H\* in a high-pitched speaker — the *psycholinguistic* event is the same. Per-speaker normalization preserves this by construction.
- **Note.** D6 used to specify the full continuous-feature spec at phone level (log-F0 z, energy z, duration ratio). The full spec is now in D19 at syllable level. D6 retains *only* the speaker-baselining commitment — the **how** of normalization, not the **what** of the feature vector.

### D7. Per-position concatenation at input (with replicated parametric tier)

- **Decision.** Each phone position has input embedding `[phone_embed ⊕ syllable_param_proj]`, where `syllable_param_proj` is a learned linear projection of the 18-dim parametric prosody vector (D19) for the syllable that contains this phone. All phones in the same syllable receive the *same* `syllable_param_proj`. After embedding, a single transformer stack processes the unified phone-tokenized representation.
- **Rationale.** Tightest practical coupling at the unit where prosody actually lives (D4). Matches autosegmental theory: tones and segments are co-temporaneous, not sequential. Matches the Fernyhough framing: prosody is part of the representation, not a side channel. Replication makes the modality-dropout ablation (D8, D9) clean — zeroing the parametric slice in all phones simultaneously zeroes prosody for the syllable.
- **Alternative considered.** Cross-attention between phone and syllable streams (CHiVE-style). Rejected for v1 because (a) it adds parameters, (b) it complicates the killer experiment (zeroing the prosody channel becomes "zeroing the cross-attention input," which has different gradient dynamics from zeroing a portion of the input embedding).
- **Replaces.** Earlier formulation `[phone ⊕ accent ⊕ boundary ⊕ continuous_proj]` (categorical accent/boundary slots). Categorical slots removed per D5.

### D8. Masking for the killer experiment

- **Decision.** Condition B of the killer experiment is implemented by *zeroing* the `syllable_param_proj` slice of the input embedding at every phone position at inference. Phone identity remains untouched.
- **Rationale.** Per-position concatenation with replication (D7) makes this a clean ablation: one slice of the embedding goes to zero across all positions, the rest is unchanged. This keeps the test honest and matches what modality dropout (D9) does during training, just held at p=1.0 for inference.
- **Replaces.** Earlier formulation that zeroed accent / boundary / continuous slices separately.

### D9. Pretraining objective: masked phone prediction + masked parametric regression with modality dropout

- **Decision.** v1 pretraining objectives:
  1. **Masked phone prediction** (BERT-style on the segmental tier, cross-entropy).
  2. **Masked parametric prosody regression** — at masked syllables, predict the full 18-dim parametric vector from D19 with MSE loss. Voiced/unvoiced indicators are predicted with binary cross-entropy in the same head.
- **Modality dropout.** During training, with probability `p_drop = 0.2` per example, the entire `syllable_param_proj` slice is zeroed at every phone's input embedding. This forces the encoder to develop genuine text-only competence rather than relying entirely on the prosody channel.
- **Optional v1 add-on:** contrastive loss on synthetic minimal pairs from Phase 1 (same text, different prosody → different embedding).
- **Rationale.** Without modality dropout, multimodal training does not yield text-only competence as a side effect — once a fully-diagnostic prosody signal is available, gradient pressure on the text pathway vanishes and the model collapses degenerately when prosody is removed at inference. Documented empirically in EXP-001 (`docs/experiments.md`) where vanilla PILM fell to 28% with prosody zeroed vs a 43% always-without-prosody baseline. p=0.2 is the literature default for modality dropout; we may sweep it again on real data if Phase 4 shows sensitivity.
- **Categorical accent/break-index losses dropped.** Replaced by the parametric regression loss. This avoids the ToBI inter-annotator-agreement ceiling (D5).
- **Killer experiment implication.** Unchanged in spirit. The Phase 5 test asks: given that both PILM (with dropout) and a text-only baseline can do text-only inference, does PILM's text-only inference benefit from having seen parametric prosody during training?

### D10. Prosody extraction: deterministic parametric extractor; AuToBI as probe wrapper only

- **Decision.** v1 prosody extraction is **deterministic** — Parselmouth-driven F0 / voicing / energy / duration measurements aggregated into the 18-dim per-syllable parametric vector (D19). No supervised training required for the extractor itself.
- **AuToBI** (Rosenberg 2010, accent F1 ~0.78 on read speech / ~0.73 on spontaneous BDC) is run only as a **probe target generator** at evaluation time — it converts the parametric outputs of *any* model checkpoint into ToBI categories so we can ask "does this representation recover ToBI?" AuToBI is **never** used as a label source for pretraining.
- **Rationale.** Once D5 demoted ToBI to probe-only status, the rationale for training a custom auto-ToBI labeler evaporated. The deterministic parametric extractor has zero training cost, zero inter-annotator-noise floor, and is reproducible across corpora. AuToBI is mature, free, and adequate for the probe role even at its modest spontaneous-speech accuracy because we only need its outputs as a coarse comparator, not as truth.
- **Phase reshuffle.** The "auto-ToBI labeler as standalone Interspeech paper" milestone (formerly Phase 3.2) is dropped. The community-resource role it would have played is replaced by **releasing the parametric prosody extractor as a Hugging Face / pip package** (`pilm-prosody-frontend`).
- **Replaces.** Earlier D10 commitment to building a Roll-et-al-style supervised labeler with WavLM features. Earlier formulation moved to Decisions explicitly rejected.

### D11. Pretraining corpus: LibriTTS first

- **Decision.** v1 pretrains on LibriTTS clean-100 + clean-360 (~460 hr). Scaling to LibriSpeech 960 + GigaSpeech is Phase 6.
- **Rationale.** LibriTTS has speaker labels and alignment-friendly text. Clean read speech is appropriate for v1; conversational speech adds noise that we should not absorb until the architecture is stable.
- **CMU ARCTIC** stays in the repo for sanity / spot-checking, not as primary training.

### D12. Evaluation: probing, not fine-tuning

- **Decision.** All evaluations in Phase 5 use linear probes over frozen representations. No fine-tuning of the encoder for downstream tasks.
- **Rationale.** Fine-tuning conflates "what did pretraining produce" with "what can a small fine-tune fix." Probing answers the cleaner question. Fine-tuning numbers can come later as a separate experiment if needed.

### D13. Compute strategy: constrained-first

- **Decision.** Phases 0–5 run on laptop / Colab Pro / single-GPU paid runs. Phase 6 scaling triggers only after Phase 5 succeeds.
- **Rationale.** The Phase 5 killer experiment is decisive; spending GPU-weeks before that is wasteful. After Phase 5, the case for compute is strong (UW department, paid GPUs, or a grant) and the spending is justified.

### D14. Honesty in evaluation

- **Decision.** v1 publishes raw numbers, bootstrap CIs, and the negative result if it lands. We do not cherry-pick test sets, we do not retroactively redefine the killer experiment, and we publish the auto-labeler's failure modes.
- **Rationale.** The thesis is testable and interesting whether it is supported or falsified. Either result deserves a clean writeup.

### D15. Dataset stack — Switchboard NXT first, then MELD, then real-world

- **Decision.** v1 uses a staged data stack:
  1. **Switchboard NXT** (LDC2009T26 + LDC97S62 audio) — primary anchor. ~63 conversations have full or near-full ToBI annotation (45 Ostendorf-style + 18 Calhoun-style). Plus dialog acts, focus/contrast (kontrast), syntax, animacy, info status, coreference, and disfluencies. Stereo audio = one channel per speaker, which makes per-speaker baseline computation straightforward.
  2. **MELD** (Friends TV show, ~14 hr, multi-party scenes, categorical emotion + sentiment labels) — added once Switchboard pipeline works. Will be auto-labeled for ToBI using the Phase 2 labeler trained on Switchboard.
  3. **Real-world / self-curated** — the third tier (NPR / podcasts / similar). Annotated by hand via the annotation tool; used for diversity and as a gold pragmatic-inference set.
- **Rationale.** Switchboard NXT is the unique English corpus combining ToBI gold with rich dialog-act + focus/contrast annotations on conversational dialogue. Starting there gets us the fewest moving parts for the modality-dropout-corrected killer experiment. MELD adds emotion labels and movie-style dialogue at scale once the auto-labeler is trustworthy. Self-curated NPR/podcast clips give us diversity and human-validated pragmatic-inference labels.
- **Replaces.** Original D11 plan (LibriTTS as primary v1 corpus). LibriTTS read-speech monologue is poorly suited for testing pragmatic inference in dialogue. Switchboard NXT is the better fit for our actual question.
- **Access.** Both LDC97S62 and LDC2009T26 are LDC-licensed. UW co-developed NXT-Switchboard (Edinburgh + Stanford + UW per Calhoun et al. 2010), so UW institutional access is expected. **Status as of session close: LDC access being confirmed.**

### D16. Refactored label space — two axes plus optional focus

- **Decision.** Replace the conflated 4-label set (STATEMENT / QUESTION / SURPRISED_QUESTION / FOCUS) used in EXP-001/002/004 with a two-axis tag set:
  - `speech_act ∈ {STATEMENT, QUESTION, COMMAND, EXCLAMATION, BACKCHANNEL, REPAIR}`
  - `affect ∈ {NEUTRAL, SURPRISED, AMUSED, IRRITATED, INCREDULOUS, RHETORICAL, ...}` (extensible; categorical-emotion labels from MELD slot directly here)
  - Optional: `contrastive_focus_word_idx` (integer, position of contrastive focus if present)
  - Optional: `confidence ∈ {hedged, neutral, confident}` (relevant for Phase 9.6 alignment work)
- **Rationale.** The synthetic v1 conflated speech-act and affect into a single axis (e.g., SURPRISED_QUESTION = QUESTION + surprise-overlay). Real dialogue routinely combines them independently — a STATEMENT can be neutral or amused or rhetorical, a QUESTION can be neutral or surprised or incredulous. Two axes capture this without an exponential label explosion.
- **Compatibility.** Switchboard NXT's existing dialog acts map onto `speech_act` (with mapping table to be written). MELD's categorical emotion labels map onto `affect`. Self-annotated clips use both axes from the tool's UI.

### D19. Parametric prosody vector specification (per-syllable, 18 dims)

- **Decision.** Each syllable carries an 18-dimensional continuous parametric vector. All values are speaker-normalized per D6.

**Pitch geometry (7 dims, semitones relative to speaker median):**

| Dim | Name | Definition |
|---|---|---|
| 1 | `f0_onset_st` | F0 at syllable start |
| 2 | `f0_nucleus_st` | F0 at vowel center |
| 3 | `f0_offset_st` | F0 at syllable end |
| 4 | `f0_max_st` | Peak F0 within syllable |
| 5 | `f0_min_st` | Minimum F0 within syllable |
| 6 | `f0_range_st` | `f0_max_st − f0_min_st` |
| 7 | `f0_slope_st_per_ms` | `(f0_offset_st − f0_onset_st) / duration_ms` |

**Tilt-style event geometry (4 dims, Taylor 2000):**

| Dim | Name | Definition |
|---|---|---|
| 8 | `f0_peak_position_norm` | Peak time / syllable duration ∈ [0, 1] |
| 9 | `f0_rise_amplitude_st` | `f0_max_st − f0_onset_st` |
| 10 | `f0_fall_amplitude_st` | `f0_max_st − f0_offset_st` |
| 11 | `tilt` | `(rise − fall) / (rise + fall) ∈ [−1, +1]` |

**Energy (2 dims, speaker-z):**

| Dim | Name | Definition |
|---|---|---|
| 12 | `rms_max_z` | Peak RMS energy, z-scored against speaker |
| 13 | `rms_mean_z` | Mean RMS energy, z-scored against speaker |

**Duration (2 dims, speaker-z):**

| Dim | Name | Definition |
|---|---|---|
| 14 | `syllable_duration_z` | Syllable duration vs. speaker syllable-duration distribution |
| 15 | `nucleus_duration_z` | Vowel-only duration vs. speaker vowel-duration distribution |

**Boundary (3 dims, computed at the syllable's right edge):**

| Dim | Name | Definition |
|---|---|---|
| 16 | `pause_after_ms` | Silence duration immediately following the syllable |
| 17 | `final_lengthening_ratio` | This syllable's duration / mean of preceding word's syllables |
| 18 | `f0_reset_st` | Next-syllable `f0_onset_st` − this-syllable `f0_offset_st` |

**Voicing flag (companion, 1 dim, not part of the 18 but always carried alongside):**
- `voiced_fraction ∈ [0, 1]` — fraction of frames within the syllable where F0 was reliably extracted. When `voiced_fraction = 0` the F0-derived dims (1–11) are masked; the energy + duration + boundary dims are still defined.

- **Rationale.** Combines PoLaR-style dense per-syllable F0 geometry (Mahrt 2018), Tilt-style event-shape parameters (Taylor 2000), explicit boundary features (Wightman & Ostendorf), and per-speaker normalization (D6). Designed so a small linear/MLP probe can recover ToBI categories from the 18-dim vector if our theoretical commitment to AM-style events is correct.
- **Voicing handling.** Per D6, F0 is undefined on voiceless segments; the voicing flag tells the encoder when the F0-derived dims are interpolated and should be down-weighted.
- **Implementation.** `scripts/extract_parametric_prosody.py` (Phase 1.5), Parselmouth-driven, deterministic.
- **Bench.** D20 specifies a frame-level F0 baseline ablation that lets us measure how much linguistic prosody is captured by these 18 dims vs. the raw 100-Hz F0 contour.

### D20. pGSLM frame-level baseline as Phase 5 ablation

- **Decision.** Phase 5 includes a same-architecture, same-compute control where the per-syllable parametric vector (D19) is replaced by a **raw frame-level F0 + energy contour stream** (pGSLM-style; Kharitonov et al. 2022) processed by a 1D CNN front-end and pooled to phone level.
- **What the comparison answers.** Whether the 18 hand-engineered dimensions in D19 capture all the linguistic prosody available in F0, or whether the model benefits from microprosodic detail that aggregation discards.
- **Decision logic.**
  - If parametric ≈ frame-level on all probes → D19 is sufficient; ship.
  - If frame-level beats parametric on a meaningful margin → either expand D19 or move to a hybrid (parametric + frame-level both as input streams).
- **Rationale.** Felipe's observation: pGSLM-style frame-level F0 contains strictly more information per second than the parametric vector. We should know, empirically, whether that extra information is load-bearing for pragmatic inference, before committing to a parametric-only architecture for Phase 6 scaling.
- **Cost.** ~1 additional pretraining run at Phase 5 budget. Cheap relative to the value of the answer.

### D17. Annotation tool — web-based, TextGrid-compatible, with active-learning hook

- **Decision.** Build a Streamlit-based web annotation tool that:
  - Plays audio with synchronized waveform + F0 contour + spectrogram (via Parselmouth).
  - Displays existing word/phone alignment from MFA.
  - Lets the user place AM/ToBI tones on syllable nuclei, mark break indices on word boundaries, and tag utterance-level (speech_act, affect, focus_word_idx).
  - Saves annotations as Praat TextGrid (industry standard) and JSON (PILM-native).
  - Supports a "model pre-annotates → user corrects" mode that loads model predictions as initial labels.
- **Rationale.** Streamlit is the lowest-effort path to a usable web UI for a researcher who wants to listen and click. TextGrid output is the lingua franca of speech-prosody research — Praat reads/writes it natively, MFA uses it, every related toolkit understands it. Active-learning support is critical because we expect to use this tool both for ground-truth annotation and for correcting auto-labeler outputs.
- **Deferred.** Multi-annotator workflow, IAA computation, schema evolution. v1 is single-user.

### D18. Pretraining is fully unsupervised; pragmatic labels are probing-only

- **Decision.** v1 pretraining is fully unsupervised:
  1. Masked phone prediction (segmental tier).
  2. Masked parametric prosody regression — 18-dim per-syllable vector from D19.
  3. Modality dropout p=0.2 (D9).
- **Pragmatic / affect labels (`speech_act`, `affect`, `focus_word_idx`) are NEVER seen during pretraining.** They are used only for downstream linear probing of frozen representations.
- **ToBI categorical labels are also NEVER seen during pretraining.** They are also probing-only (D5).
- **Rationale.** This sharpens the Fernyhough test: any text-only-inference advantage PILM shows over a same-compute text-only baseline reflects representation quality, not label leakage. It also matches the regime that scales naturally to unlabeled data (eventual Phase 6 scaling).
- **Replaces.** EXP-001/002/004 used label-supervised classification. That regime was fine for harness validation but not for the core hypothesis test. Earlier D18 also listed masked AM/ToBI accent/boundary categorical prediction in the loss; those targets are removed per the parametric-pivot revisions to D5 and D9.

---

## Decisions explicitly deferred

These have been raised and explicitly deferred to a later phase rather than left ambiguous:

- **Cross-attention vs concatenation, revisited.** If concatenation underperforms in Phase 4 probing, revisit cross-attention as a second architecture run.
- **Decoder activation.** Phase 7.
- **Multilingual extension.** Phase 8.
- **Disentanglement losses (speaker vs prosody).** Revisit if Phase 4 probing shows the prosody channel still encodes speaker identity.
- **Curated pragmatic minimal-pair corpus** as a community resource. Phase 9.8.
- **Inner-speech / silent-reading psycholinguistics test** (predict reading times). Phase 9.10.

---

## Decisions explicitly rejected

- **Word-level prosody tokens (ProsodyLM-style).** Granularity and channel arguments in `theory_notes.md` §2.
- **Textless model (pGSLM-style) as the architecture.** PILM specifically tests text-with-internalized-prosody; removing text removes the killer experiment. (Note: pGSLM-style frame-level F0 returns as a *baseline ablation* in D20, not as the architecture.)
- **Fine-tuning-based evaluation as primary.** See D12.
- **Per-utterance F0 normalization as default.** Lossy with respect to speaker information; see D6.
- **Hand-written WFST tokenizer.** D2.
- **LibriTTS as v1 primary corpus.** Read-speech monologue does not exercise dialogue-level pragmatic inference. Replaced by Switchboard NXT (D15). LibriTTS may return as a Phase 6 scaling corpus.
- **Conflated single-axis pragmatic labels (e.g., SURPRISED_QUESTION).** Replaced by 2-axis (speech_act, affect) per D16.
- **Label-supervised pretraining.** Replaced by fully unsupervised pretraining per D18. Labels are now strictly probing-only.
- **AM/ToBI categorical labels as training target.** Hard ceiling at human inter-annotator agreement (~80% accent presence, ~60% accent type); ToBI was designed as a transcription standard for human linguists, not as ML targets. Replaced by parametric prosody vector (D19) as the trained-against representation; ToBI labels survive only as a downstream probe target (D5). Earlier D5 / D9 / D18 formulations moved here.
- **Custom auto-ToBI labeler (Roll-et-al-style WavLM + Phoneme-BERT supervised classifier) as a deliverable.** Once D5 demoted ToBI to probe-only, the rationale for training a custom labeler evaporated. Replaced by deterministic Parselmouth-driven parametric extractor + AuToBI as probe wrapper (D10). Earlier D10 formulation moved here.
