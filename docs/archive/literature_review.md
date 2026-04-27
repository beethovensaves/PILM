# PILM — Literature Review

_Last updated: 2026-04-25_

This is a working literature review for the **Prosody-Internalized Language Model (PILM)** project. The aim is twofold: (i) ground the project's theoretical commitments, and (ii) make it easy to position PILM against existing work. Sections are organized so that each can be read independently.

For each work cited, I include the role it plays for PILM (e.g., "evidential," "competing approach," "method we may borrow"). Where I am uncertain about a citation, I note it explicitly rather than fabricate.

---

## 1. Inner speech, internalization, and the Fernyhough framing

The motivating theoretical claim of PILM is that humans internalize prosody during first-language acquisition and use it as a structural layer of meaning even when not speaking aloud. This is grounded in the Vygotsky–Fernyhough tradition.

- **Vygotsky (1934 / English 1962, *Thought and Language*).** Inner speech as the internalization of social/external dialogue. Foundation for the entire tradition.
- **Fernyhough, C. (2004). *Alien voices and inner dialogue: towards a developmental account of auditory verbal hallucinations*. New Ideas in Psychology, 22(1), 49–68.** Develops the dialogic theory of inner speech.  
  → [PDF](https://hearingthevoice.org/wp-content/uploads/2014/01/Fernyhough-2004.pdf)
- **Alderson-Day, B., & Fernyhough, C. (2015). *Inner Speech: Development, Cognitive Functions, Phenomenology, and Neurobiology*. Psychological Bulletin.** Comprehensive review distinguishing **expanded inner speech** (retains phonological/turn-taking properties of external dialogue) from **condensed inner speech** ("thinking in pure meanings"). PILM's claim leans on expanded inner speech retaining prosodic information.  
  → [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4538954/)
- **Fernyhough, C. (2016). *The Voices Within*. Basic Books.** Trade-press synthesis. Useful for framing.
- **Inner Speech (Stanford Encyclopedia of Philosophy).** Standard reference for the philosophical landscape.  
  → [SEP entry](https://plato.stanford.edu/entries/inner-speech/)
- **Fossa, P., & Pachecho-Montoya, R. (2024).** Recent reviews suggest inner speech carries prosodic information; useful as recent evidence that this is not a fringe claim. (Cited indirectly via the SEP and PMC reviews above.)
- **Lœvenbruck, H., et al. (2018). *Inner Speech as Language Process and Cognitive Tool*. Trends in Cognitive Sciences.** Argues inner speech is multimodal, including prosodic and articulatory features.  
  → [Cell Press](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(23)00210-3)
- **Frontiers (2019). *A Penny for Your Thoughts: Children's Inner Speech and Its Neuro-Development*.** Developmental data; relevant to the L1-acquisition framing.  
  → [Frontiers](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.01708/full)

**Role for PILM.** Provides the theoretical motivation for the killer experiment: if humans internalize prosody such that it shapes their cognition even on text-only inputs, a prosody-pretrained model should retain prosodic inductive biases at text-only inference. This is how PILM differentiates from "just another multimodal model."

---

## 2. Implicit prosody (Fodor) — the empirical bridge

Fodor's Implicit Prosody Hypothesis is the empirical link between Fernyhough's general theory and a measurable behavioral signature. Silent readers project prosody onto text and this affects parsing.

- **Fodor, J. D. (1998). *Learning to parse?* Journal of Psycholinguistic Research, 27(2), 285–319.** Original IPH proposal.
- **Fodor, J. D. (2002). *Prosodic Disambiguation in Silent Reading*. Proceedings of NELS 32.** The most-cited statement of the hypothesis.  
  → [PDF](https://janetdeanfodor.wordpress.com/wp-content/uploads/2016/06/fodor-2002-prosodic-disambiguation-in-silent-reading.pdf)
- **Breen, M. (2014). *Empirical Investigations of the Role of Implicit Prosody in Sentence Processing*. Language and Linguistics Compass, 8(2), 37–50.** Review article. Best single entry point.  
  → [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/lnc3.12061)
- **Fodor, J. D. (2007). *Could implicit prosody also be applied?*** Discusses cross-linguistic relative clause attachment differences traced to prosodic phrasing.
- **Webman-Shafran, R. (2018). *Implicit prosody and parsing in silent reading*. Journal of Research in Reading.**  
  → [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9817.12124)
- **Yao & Scheepers (2018). *Direct speech quotations promote low relative-clause attachment in silent reading of English*. Cognition.** Shows that even *manipulating* implicit prosody (via punctuation/typography) shifts parsing.  
  → [PubMed](https://pubmed.ncbi.nlm.nih.gov/29609099/)
- **Bishop (2016). *Empirical investigations of implicit prosody*.** Methodological review.

**Role for PILM.** This is the *operationalization* of the Fernyhough claim. The classic IPH probes (relative clause attachment, garden paths, focus interpretation) become candidate evaluation tasks: if PILM beats a text-only baseline on these *with text-only inputs*, the prosody-internalization hypothesis is supported.

---

## 3. Autosegmental-Metrical theory and ToBI

PILM's symbolic prosody inventory will be drawn from AM/ToBI (per design decision in this conversation).

- **Pierrehumbert, J. (1980). *The Phonology and Phonetics of English Intonation*. PhD thesis, MIT.** The foundational AM thesis.
- **Beckman, M., & Pierrehumbert, J. (1986). *Intonational structure in Japanese and English*. Phonology Yearbook 3.** Cross-linguistic extension.
- **Ladd, D. R. (2008). *Intonational Phonology*, 2nd ed. Cambridge University Press.** The standard textbook on AM theory.
- **Silverman, K., et al. (1992). *ToBI: a standard for labeling English prosody*. ICSLP.** The original ToBI specification.
- **Beckman, M., Hirschberg, J., & Shattuck-Hufnagel, S. (2005). *The original ToBI system and the evolution of the ToBI framework*.** Standard reference for the annotation scheme.
- **Dilley, L. C., & Breen, M. (2018). *An enhanced autosegmental-metrical theory (AM+) facilitates phonetically-transparent prosodic annotation*.**  
  → [PDF](https://speechlab.cas.msu.edu/PDF/Dilley_Breen_Symp_2018.pdf)

**Role for PILM.** Symbolic backbone. Pitch accents (H\*, L\*, L+H\*, L\*+H, H+!H\*), phrase accents (H-, L-), boundary tones (H%, L%), break indices 0–4. This vocabulary is what the auto-ToBI labeler will produce.

---

## 4. F0 normalization and speaker baselines

PILM commits to per-speaker F0 baselines (non-lossy). Relevant work:

- **Honorof, D. N., & Whalen, D. H. (2005). *Perception of pitch location within a speaker's F0 range*. Journal of the Acoustical Society of America.** The canonical reference for "F0 is interpreted relative to speaker range, not in absolute Hz."  
  → [PDF](https://personal.utdallas.edu/~assmann/hcs6367/honorof_whalen05.pdf)
- **De Looze, C., & Hirst, D. (2008). *Detecting changes in key and range for the automatic modelling and coding of intonation*.** Methodological — automatic estimation of speaker pitch range.
- **Hirst, D. (2011). *The analysis by synthesis of speech melody: from data to models*. Journal of Speech Sciences.** MOMEL/INTSINT framework for F0 modelling, useful as practical reference.
- **Mertens, P. (2004). *The Prosogram: semi-automatic transcription of prosody based on a tonal perception model*.** Practical algorithm for stylized F0 perception-relevant features.
- **Tone normalization comparison (Zhang & Ye, 2018).** Compares z-score, log-z, semitone, and Lobanov-style normalizations.  
  → [PDF](https://aclanthology.org/Y18-1095.pdf)
- **Estimating underlying F0 range from spectral features (MDPI, 2022).** Methods to estimate speaker baseline from short audio.  
  → [MDPI](https://www.mdpi.com/2076-3417/12/13/6494)
- **Adaptive Baseline Calibration for Voice Stress Assessment.** Engineering Archive preprint with practical approach to per-speaker baseline drift.  
  → [EngrXiv](https://engrxiv.org/preprint/view/6768)

**Role for PILM.** Determines preprocessing. v1 plan: per-speaker log-F0 z-scoring using LibriTTS speaker IDs; speaker clustering as fallback. Continuous prosody features will be reported relative to speaker baseline + range, not raw Hz.

---

## 5. Prosodic pragmatics — speech acts, focus, contrast

The downstream evaluation domain.

- **Wilson, D., & Wharton, T. (2006). *Relevance and prosody*. Journal of Pragmatics.** Influential framework linking prosody to relevance theory.
- **Cole, J. (2015). *Prosody in context: a review*. Language, Cognition and Neuroscience.** Standard review.
- **Prieto, P. (2015). *Intonational meaning*. WIREs Cognitive Science.** Survey of how intonation conveys speaker meaning.
- **Cambridge Handbook of Language in Context, Ch. 14: *Prosodic Pragmatics in Context*.**  
  → [Cambridge](https://www.cambridge.org/core/books/abs/cambridge-handbook-of-language-in-context/prosodic-pragmatics-in-context/6F468876BB6B0E3301F873D5486DE466)
- **Kurumada, C., & Clark, E. V. (2017). *Pragmatic inferences in context: learning to interpret contrastive prosody*. Journal of Child Language.** Empirical work on how listeners (including children) make pragmatic inferences from contrastive prosody.  
  → [PDF](https://kinderlab.bcs.rochester.edu/papers/KurumadaClark2016.pdf)
- **Kurumada, C., et al. (2014). *Rapid adaptation in online pragmatic interpretation of contrastive prosody*. CogSci.**  
  → [PDF](https://kinderlab.bcs.rochester.edu/papers/KurumadaEtAl_CogSci2014.pdf)
- **PNAS (2025). *Three distinct components of pragmatic language use: Social conventions, intonation, and world knowledge–based causal reasoning*.** Argues intonation is one of three structurally distinct pragmatic capacities.  
  → [PNAS](https://www.pnas.org/doi/10.1073/pnas.2424400122)
- **Speech act recognition gating study (2025). *The time course of speech act recognition conveyed by speech prosody*. Language, Cognition and Neuroscience.** Online evidence that listeners recognize speech acts (questions vs. statements) primarily from terminal pitch movement, often before lexical material completes.  
  → [Tandfonline](https://www.tandfonline.com/doi/full/10.1080/23273798.2025.2506641)
- **Cole, J., et al. (2017). *Crowd-sourcing prosodic annotation*.** Relevant if we ever want non-expert annotation of prosodic prominence.

**Role for PILM.** Defines the evaluation tasks: speech act type (statement / question / command / exclamation), contrastive focus location, and discourse-level effects.

---

## 6. Sarcasm, irony, emotion in prosody

Direct relevance to the user's "rising + extra-high F0 = question + surprise" example.

- **Cheang, H. S., & Pell, M. D. (2008). *The sound of sarcasm*. Speech Communication, 50(5), 366–381.** Identifies acoustic correlates: lowered mean F0, reduced F0 range and intensity, slower speaking rate. The single most-cited paper on sarcasm acoustics.
- **Cheang & Pell (2009). *Acoustic markers of sarcasm in Cantonese and English*.** Cross-linguistic.
- **Bryant, G. A., & Fox Tree, J. E. (2005). *Is there an ironic tone of voice?* Language and Speech.**
- **Rockwell, P. (2000). *Lower, slower, louder: vocal cues of sarcasm*. Journal of Psycholinguistic Research.**
- **Banse, R., & Scherer, K. R. (1996). *Acoustic profiles in vocal emotion expression*. Journal of Personality and Social Psychology.** Classic emotion-acoustic mapping.
- **Juslin, P. N., & Laukka, P. (2003). *Communication of emotions in vocal expression and music performance*. Psychological Bulletin.** Meta-analysis. Shows surprising robustness of emotion-acoustic mappings cross-culturally.
- **Mauchand, M., & Pell, M. D. (2021). *Emotion and the prosodic cloak: how prosody constrains the inferential interpretation of utterances*.** Recent integration with pragmatic inference.

**Role for PILM.** Tells us which acoustic features matter for emotion/affect labeling and what the expected effect sizes are. "Extra-high F0 peak adds surprise" is consistent with widened F0 range being characteristic of high-arousal positive emotions.

---

## 7. Emotion as cognition — theoretical framing

The user mentioned "emotion as cognition." Three relevant traditions:

- **Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology.** Two-dimensional valence × arousal space. Standard for dimensional emotion modeling in NLP.
- **Scherer, K. R. (2009). *The dynamic architecture of emotion: Evidence for the component process model*. Cognition and Emotion.** Argues emotion is the synchronized response of multiple subsystems including appraisal, expression, action tendency. Most useful theoretical framework for treating emotion as a *cognitive* (not merely affective) phenomenon.
- **Barrett, L. F. (2017). *How Emotions Are Made: The Secret Life of the Brain*. Houghton Mifflin Harcourt.** Theory of constructed emotion. Argues emotions are not biological natural kinds but constructions from interoceptive and conceptual primitives. Strong theoretical alignment with "prosody is part of how meaning is constructed."
- **Pessoa, L. (2008). *On the relationship between emotion and cognition*. Nature Reviews Neuroscience.** Argues emotion and cognition are integrated, not separable systems.

**Role for PILM.** Provides the conceptual frame for treating emotion-bearing prosody as a *cognitive* signal that participates in meaning construction, not as a separate affective overlay. This aligns with the Fernyhough framing.

---

## 8. Self-supervised speech representations

These are the candidate audio encoders.

- **Baevski, A., et al. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations*. NeurIPS.** Currently used in PILM's stub frontend.  
  → [arXiv](https://arxiv.org/abs/2006.11477)
- **Hsu, W.-N., et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*. IEEE/ACM TASLP.**  
  → [arXiv](https://arxiv.org/abs/2106.07447)
- **Chen, S., et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*. JSTSP.** Strong on prosody-related tasks per layer-wise analyses.  
  → [arXiv](https://arxiv.org/abs/2110.13900)
- **Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). ICML.**  
  → [arXiv](https://arxiv.org/abs/2212.04356)
- **Pasad, A., Chou, J.-C., & Livescu, K. (2021). *Layer-wise analysis of a self-supervised speech representation model*. ASRU.** Shows mid-layer features carry phonetic info; top layers re-encode semantic content. Relevant when picking which layers to read for prosody.  
  → [arXiv](https://arxiv.org/abs/2107.04734)

**Role for PILM.** WavLM is probably the right starting point given the literature on its prosodic content. We will not train a SSL encoder from scratch.

---

## 9. Computational prosody for TTS — closest architectural precedents

These are the works whose architectures most resemble what PILM is being asked to build.

- **Skerry-Ryan, R. J., et al. (2018). *Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron*. ICML.** Reference encoder → prosody embedding → TTS. The original "prosody embedding" paper.  
  → [PDF](https://proceedings.mlr.press/v80/skerry-ryan18a/skerry-ryan18a.pdf)
- **Kenter, T., Wan, V., et al. (2019). *CHiVE: Varying Prosody in Speech Synthesis with a Linguistically Driven Dynamic Hierarchical Conditional Variational Network*. ICML.** Multi-rate RNNs reading frame-level features and emitting at syllable/word/sentence boundaries. **Architecturally the closest precedent for PILM's parallel-tier design.**  
  → [PDF](http://proceedings.mlr.press/v97/kenter19a/kenter19a.pdf)
- **PhonemeVec (2024). *A Phoneme-Level Contextual Prosody Representation For Speech Synthesis*. ACM TALLIP.** Phone-level prosody embeddings, directly aligned with PILM's chosen unit.  
  → [ACM](https://dl.acm.org/doi/10.1145/3711828)
- **StyleTTS 2 (Li et al., 2023).** Controllable expressive TTS. Useful as the engine for minimal-pair stress tests in PILM Phase 5.  
  → [GitHub](https://github.com/yl4579/StyleTTS2)
- **MSR-Codec (2025). *Multi-Stream Residual Codec for High-Fidelity Speech Generation with Information Disentanglement*.** Disentangles speech into timbre / semantics / prosody / residual.  
  → [arXiv](https://arxiv.org/html/2509.13068)

**Role for PILM.** CHiVE is the architectural template, with two key changes: (a) PILM operates at phone, not syllable, as primary unit; (b) PILM's purpose is *understanding* (encoder + classifier) before generation, not TTS.

---

## 10. Speech language models with prosody — competitors and adjacent work

The competitive landscape PILM will be measured against.

- **Lakhotia, K., et al. (2021). *Generative Spoken Language Modeling from Raw Audio* (GSLM). TACL.** Foundational textless speech LM. No explicit prosody.  
  → [arXiv](https://arxiv.org/abs/2102.01192)
- **Kharitonov, E., et al. (2022). *Text-Free Prosody-Aware Generative Spoken Language Modeling* (pGSLM).** Adds F0 and duration as additional channels in a textless speech LM. **Architecturally close to PILM.** Worth careful comparison.  
  → [arXiv](https://arxiv.org/abs/2109.03264)
- **Borsos, Z., et al. (2023). *AudioLM: a Language Modeling Approach to Audio Generation*. IEEE/ACM TASLP.**  
  → [arXiv](https://arxiv.org/abs/2209.03143)
- **Nguyen, T. A., et al. (2024). *SpiritLM: Interleaved Spoken and Written Language Model*. Meta AI.** Interleaves speech and text tokens at training time.  
  → [arXiv](https://arxiv.org/abs/2402.05755)
- **Défossez, A., et al. (2024). *Moshi: a speech-text foundation model for real-time dialogue*. Kyutai Labs.** Full-duplex speech-text foundation model. "Inner Monologue" stream of time-aligned text as prefix to audio. Important contemporary reference.  
  → [arXiv](https://arxiv.org/abs/2410.00037) | [GitHub](https://github.com/kyutai-labs/moshi)
- **Lin, K., et al. (2025). *ProsodyLM: Uncovering the Emerging Prosody Processing Capabilities in Speech Language Models*. arXiv 2507.20091.** **The headline competitor.** Sequential text + word-level prosody tokens. PILM's structural critique targets this paper specifically.  
  → [arXiv](https://arxiv.org/abs/2507.20091)
- **Phonological Tokenizer (2026, arXiv 2601.19781). *Prosody-Aware Phonetic Token via Multi-Objective Fine-Tuning with Differentiable K-Means*.** Learned phonetic tokens that retain phonological/prosodic info. Different mechanism, similar goal.  
  → [arXiv](https://arxiv.org/abs/2601.19781)
- **MOSS-Speech (2025). *Towards True Speech-to-Speech Models Without Text Guidance*.**  
  → [arXiv](https://arxiv.org/html/2510.00499)
- **When Large Language Models Meet Speech: A Survey on Integration Approaches (2025, ACL Findings).** Survey of the field. Useful map.  
  → [PDF](https://aclanthology.org/2025.findings-acl.1041.pdf)

**Role for PILM.** The competitive landscape. The two papers we must beat or substantively differ from are **ProsodyLM** (architectural target of our critique) and **pGSLM** (closest in approach — adds F0/duration channels but is textless). The differentiation is:

- vs. ProsodyLM: parallel sub-word tier, not sequential word-level interleaving.
- vs. pGSLM: text + prosody (not textless), pragmatic inference target (not just generation).
- vs. SpiritLM/Moshi: explicit prosodic structure (AM/ToBI + continuous), not raw codec tokens.

---

## 11. Auto-ToBI labeling — the chokepoint

PILM needs ToBI-style labels at scale; hand annotation is impossible. This subsection is the most operationally important.

- **Wightman, C. W., & Ostendorf, M. (1994). *Automatic labeling of prosodic patterns*. IEEE Transactions on Speech and Audio Processing.** Classical statistical approach.
- **Sun, X. (2002). *Pitch accent prediction using ensemble machine learning*.** Older but still relevant baseline.
- **Ananthakrishnan, S., & Narayanan, S. (2008). *Automatic prosodic event detection using acoustic, lexical, and syntactic evidence*. IEEE TASLP.** Classical multi-feature approach.  
  → [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2709295/)
- **Rosenberg, A. (2010). *AuToBI – A Tool for Automatic ToBI annotation*. Interspeech.** Open-source toolkit. v1 baseline candidate.
- **Sundararaman, M. N., et al. (2019). *Investigation of unsupervised methods for the lexical placement of prominence and boundary in English prosody*.**
- **Roll, N., et al. (2025). *Prosody Labeling with Phoneme-BERT and Speech Foundation Models*. arXiv 2507.03912.** **The most relevant recent work.** Combines phoneme-BERT with WavLM/HuBERT features to predict ToBI categories. Numbers to beat or match.  
  → [arXiv](https://arxiv.org/html/2507.03912v1)

**Plan.** Phase 2 builds an auto-labeler in the style of Roll et al. (2025), trained on whatever ToBI-gold corpora are accessible (BURNC, Switchboard-NXT subset). If we can't reach ~80% accent F1 in 4 weeks, we fall back to AuToBI or commercial labelers.

---

## 12. Datasets — pretraining, supervised eval, minimal pairs

### Pretraining (large, naturalistic)

- **LibriTTS (Zen et al., 2019)** — ~585 hr, multi-speaker, has alignments. **Primary v1 pretraining corpus.**  
  → [openslr.org/60](https://www.openslr.org/60/)
- **LibriSpeech (Panayotov et al., 2015)** — ~960 hr.
- **GigaSpeech / People's Speech / VoxPopuli** — for scaling.
- **CMU ARCTIC (already in repo)** — ~1132 utterances, single speaker awb. Useful for sanity testing because it has phone alignments and pitch marks.

### ToBI gold labels (small, expensive)

- **Boston University Radio News Corpus (BURNC, Ostendorf et al., 1995).** ~3 hr ToBI-labeled. The gold standard for English ToBI.  
  → [LDC](https://catalog.ldc.upenn.edu/LDC96S36)
- **Switchboard NXT (Calhoun et al., 2010).** Subset has ToBI + dialog acts + syntactic parses.  
  → [Switchboard NXT](https://groups.inf.ed.ac.uk/switchboard/)
- **Boston Directions Corpus.** Smaller but ToBI-labeled.

### Pragmatic / dialog-act / emotion supervised eval

- **Switchboard NXT** (above) — for dialog acts.
- **IEMOCAP (Busso et al., 2008)** — acted emotion, 12 hr, 5 dyadic sessions.
- **MSP-Podcast (Lotfian & Busso, 2017+)** — naturalistic emotional speech, large.
- **MELD (Poria et al., 2019)** — multimodal emotion in TV-series dialogues.
- **SLUE (Speech Language Understanding Evaluation, Shon et al., 2022).** Includes some prosody-relevant tasks.  
  → [arXiv](https://arxiv.org/abs/2111.10367)

### Minimal pairs (Phase 5 demo)

No off-the-shelf corpus exists. Plan: synthesize 100–300 sentences in 3–5 prosodic variants each via StyleTTS 2 / F5-TTS, manually verify prosody, use as held-out demo set. This is acknowledged as TTS-biased; it is a *demo*, not the headline metric.

---

## 13. Probing benchmarks — how to measure "prosody internalization"

PILM's evaluation will draw from these:

- **de Seyssel, M., et al. (2023). *ProsAudit, a prosodic benchmark for self-supervised speech models*. Interspeech.** Two tasks: protosyntax (strong vs weak prosodic boundaries) and lexical (within-word vs between-word pauses). **Direct fit for evaluating PILM's encoder.**  
  → [arXiv](https://arxiv.org/abs/2302.12057)
- **Zero Resource Speech Benchmark (Dunbar et al., 2021).** Broader textless speech evaluation.
- **SLUE Phase 2 (2024).** Some tasks involve prosodic content.
- **Pasad et al. (2021)** layer-wise probing methodology — what PILM should adopt for "is the prosody actually represented?" probes.

---

## 14. Multilingual prosody — for the long tail

For Phase 8 / future work.

- **Ladd (2008)** — already cited; covers cross-linguistic AM.
- **Jun, S.-A. (Ed.) (2014). *Prosodic Typology II: The Phonology of Intonation and Phrasing*.** Cross-linguistic reference.
- **Mandarin tone literature** — Xu (1997), Prom-on et al. (2009).
- **Yoruba and African tonal languages** — Connell, Akinlabi.
- **Pitch-accent languages (Japanese, Swedish, Serbian).**
- **Tone normalization comparison (Zhang & Ye, 2018).**  
  → [PDF](https://aclanthology.org/Y18-1095.pdf)

---

## 15. Where PILM fits — and what is genuinely new

The space PILM is trying to occupy:

| Existing approach | What it does | What it lacks (vs PILM aim) |
|---|---|---|
| ProsodyLM | Sequential text + word-level prosody tokens | Sub-word granularity; parallel-tier structure |
| pGSLM | Adds F0 + duration channels; textless | Text co-modeling; pragmatic inference target |
| SpiritLM | Interleaves speech-text tokens | Explicit symbolic prosody; per-speaker baselines |
| Moshi | Real-time dialogue, text-prefix audio | Pragmatic-inference probing; sub-word prosodic structure as primary feature |
| AuToBI / Roll et al. 2025 | ToBI labeling | Not language modeling; no inference downstream |
| CHiVE | Multi-rate prosody for TTS | Generation only, not understanding; not pretrained at scale |

**The novel claim** PILM is positioned to make is the Fernyhough/Fodor-flavored one: pretraining with parallel sub-word prosodic tiers + per-speaker-baselined continuous features should produce representations that retain prosodic inductive bias *even when prosody is masked at inference*. To my knowledge, no one has tested this directly. This is the headline experiment.

If that result comes through, the secondary contribution is the principled architecture (per-position concatenation of segmental + suprasegmental + continuous, anchored at phone, AM/ToBI vocabulary) and the demonstration that it scales meaningfully.

---

## Reading priority for the user

If time is limited, read in this order:

1. **Breen 2014** (implicit prosody review) — operationalizes the killer experiment.
2. **Alderson-Day & Fernyhough 2015** (PMC) — the theoretical foundation.
3. **Lin et al. 2025 (ProsodyLM)** — the paper PILM is critiquing.
4. **Kharitonov et al. 2022 (pGSLM)** — closest architectural neighbor.
5. **Roll et al. 2025 (Phoneme-BERT auto-ToBI)** — Phase 2 reference implementation.
6. **Cole 2015 (Prosody in context)** — general orientation for prosodic pragmatics.
7. **Honorof & Whalen 2005** — F0 normalization basics.
8. **Kenter et al. 2019 (CHiVE)** — architectural precedent.

---

_Open issues to track in this file as we proceed:_
- Verify whether Roll et al. 2025 release code/labelers we can reuse.
- Locate ToBI-gold subset of Switchboard-NXT (access).
- Track new ProsodyLM follow-ups; recheck quarterly.
- Find or build a pragmatic-inference minimal-pair corpus the field can adopt (potential publication on its own).
