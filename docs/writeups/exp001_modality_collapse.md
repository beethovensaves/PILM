# EXP-001 — Why Vanilla Multimodal Training Collapses Without Modality Dropout

_A long-form writeup of the Phase 1 synthetic killer-experiment baseline._

_Date: 2026-04-25_
_Code: `scripts/run_synthetic_killer_test.py`, `models/pilm_toy.py`, `models/synthetic_dataset.py`, `scripts/gen_synthetic_prosody.py`_
_Results: `data/synthetic/killer_test_results.json`_

---

## 1. TL;DR

We trained a small transformer encoder (842K parameters) on synthetic prosody-text data designed to test the Fernyhough/Fodor hypothesis that prosody pretraining installs inductive biases that persist at text-only inference. **A vanilla setup — train with all channels, evaluate with prosody zeroed — does not test the hypothesis.** It tests, instead, whether multimodal training spontaneously produces text-only competence as a side effect. The answer is no, and informatively no:

- With prosody at inference, the encoder hits **100% accuracy** on a 4-way pragmatic-label task.
- With prosody zeroed at inference but available during training, the encoder collapses to **28.4% overall** — *worse than chance among informative labels*, and per-label decomposition shows it predicts QUESTION for every input.
- A control trained without prosody throughout reaches **42.9%** by attending to the small lexical signal (~50% of QUESTION/SURPRISED_QUESTION begin with a question-typical filler word).

The PILM-with-prosody-zeroed model is **−14.5 pp** below the never-saw-prosody control. This is a degenerate failure, not a graceful fallback.

The lesson is that the killer experiment as originally written measured the wrong thing. The corrective fix — **modality dropout during pretraining** — is small, well-known in the multimodal-training literature, and now locked into the v1 design (`docs/design_decisions.md` D9). Follow-up EXP-002 confirms it cleanly restores text-only competence.

The actual sharpened Fernyhough question becomes: given that PILM (with modality dropout) and a text-only baseline both have text-only competence, does PILM's text-only competence *exceed* the baseline's? On synthetic data the answer is no — the lexical signal is too sparse for prosody pretraining to install useful transferable priors. On natural data this remains the Phase 5 question.

---

## 2. Background and motivation

### 2.1 The Fernyhough/Fodor hypothesis

Vygotsky proposed inner speech is internalized social speech. Fernyhough refined this with a two-form distinction: **expanded inner speech**, which retains phonological and prosodic properties of external dialogue, and **condensed inner speech**, which compresses semantically. Fodor's *Implicit Prosody Hypothesis* (1998, 2002) gives this an empirical handle: silent readers project prosody onto text and this measurably affects parsing, attachment ambiguity, and focus interpretation.

The PILM project's central claim, reformulating these in computational terms:

> A language model pretrained on parallel sub-word prosodic and segmental tiers should retain prosodic inductive biases at inference time, including when the prosody channel is unavailable. This would mirror the implicit-prosody phenomenon documented in human silent reading.

This is the falsifying gate for Phase 5. The test, as originally written, was:

> Train PILM with prosody. At inference, zero the prosody channel. If text-only PILM beats a text-only baseline trained without prosody, the hypothesis is supported.

EXP-001 is the synthetic mini-version of this test: small toy data, small model, fast iteration to validate the harness before committing to natural-speech compute.

### 2.2 Why a synthetic test first

The constrained-first principle (`docs/design_decisions.md` D13) holds that we should not spend GPU budget on natural-speech pretraining before we have validated the architecture and the experimental harness. EXP-001 was designed to:

1. Verify the encoder's per-position-concatenation architecture works (Condition A should hit ceiling).
2. Verify the prosody-mask ablation is cleanly implemented at the input level (zeroing should produce a different output without changing parameters).
3. Verify the experimental harness (two models, three conditions, bootstrap CIs) runs end-to-end.

Whether the synthetic data could *prove* the Fernyhough hypothesis was secondary. Synthetic data, by design, has limited correlational structure between prosody and text — the kind of structure on which the Fernyhough effect would have to ride.

EXP-001 succeeded on (1)–(3). What it also produced was a sharper understanding of what the killer experiment is actually testing.

---

## 3. Setup

### 3.1 Data — the synthetic generative model

We generated 12,000 utterances (10k train / 1k dev / 1k test) using a toy generative model whose mechanics are fully transparent. Pseudocode:

```
For each utterance:
    1. Sample a pragmatic label L ∈ {STATEMENT, QUESTION, SURPRISED_QUESTION, FOCUS}
       with weights (3, 3, 2, 2). Empirical balance ≈ 30/30/20/20.

    2. Sample a number of words n_w ∈ [3, 8] uniformly.

    3. If L ∈ {QUESTION, SURPRISED_QUESTION}, with probability 0.5 prepend a
       question-typical filler word from a pool of four fixed shapes
       ({D-IY-D, K-AH-N, W-AH-T, B-OW-R}). This is the only purely-lexical
       signal in the v1 toy world.

    4. For each remaining word, sample 1–4 syllables. Each syllable is
       (C)V(C) drawn from:
           VOWELS = {AA, AE, IY, OW, UW, EH, AH}
           CONSONANTS = {P, T, K, B, D, G, M, N, S, Z, L, R, W, Y, F}

    5. If L = FOCUS, sample a focus_word index k ∈ [0, n_w).

    6. Apply prosody as a deterministic-ish function of L (with small noise):
           - declination: log_f0_z(syllable_i) = 0.5 - i × 1.0 / N
           - lexical stress lands on the first syllable of every word (H*)
           - if L = FOCUS, the stressed syllable of word k gets L+H*
             with peak log_f0_z = 1.0 (vs ~0.3 for ordinary stress)
           - if L = QUESTION, the terminal stressed syllable gets log_f0_z = 0.7
             and the final phone gets boundary B4_H (high terminal)
           - if L = SURPRISED_QUESTION, terminal log_f0_z = 1.5
             and boundary = B4_HH
           - if L = STATEMENT, terminal boundary = B4_L (low terminal)
           - all continuous values get Gaussian noise σ=0.08
           - duration multipliers: stressed × 1.2, focused × 1.4, pre-boundary × 1.1
```

Each phone in each utterance is emitted with: the phone identity, vowel/consonant flag, syllable index, word index, stress flag, accent label (NONE/H\*/L\*/L+H\*), boundary label (NONE/B1/B4_L/B4_H/B4_HH), continuous log-F0-z (null on consonants), continuous energy-z, and continuous duration-ratio.

### 3.2 Why this generative process is informative

The data has a **fully-determining prosody signal**: given accent, boundary, and continuous prosody, the label is recoverable with certainty. Specifically:

- The terminal boundary token alone disambiguates STATEMENT (B4_L) vs. QUESTION (B4_H) vs. SURPRISED_QUESTION (B4_HH).
- The presence of L+H\* (only emitted in FOCUS utterances) and its position fully specify FOCUS.

In contrast, the **text-only signal is sparse**. The question-typical filler appears in only ~50% of QUESTION/SURPRISED_QUESTION utterances and tells you nothing about FOCUS or STATEMENT vs. SURPRISED_QUESTION. So a text-only model has signal only for "is this a question class?" with ~50% recall.

This asymmetry between channels is what makes the test interesting. It is also what makes the failure mode predictable in retrospect.

### 3.3 Architecture

The model is a small encoder-decoder transformer trained as encoder-only for v1 (decoder is in the checkpoint but not active). The per-position input embedding is the concatenation of four slices:

$$
e_t = W_{\text{in}} \begin{bmatrix} \phi(p_t) \\ \alpha(a_t) \\ \beta(b_t) \\ W_{\text{cont}}\, c_t \end{bmatrix}
$$

where:
- $\phi : \{0, \ldots, 21\} \to \mathbb{R}^{64}$ is the phone embedding lookup
- $\alpha : \{0, \ldots, 3\} \to \mathbb{R}^{16}$ is the accent embedding lookup
- $\beta : \{0, \ldots, 4\} \to \mathbb{R}^{16}$ is the boundary embedding lookup
- $W_{\text{cont}} \in \mathbb{R}^{16 \times 4}$ projects the continuous channel
- $c_t \in \mathbb{R}^4$ is `[log_f0_z, voiceless_flag, energy_z, dur_rel]`
- $W_{\text{in}} \in \mathbb{R}^{128 \times 112}$ projects the concatenated 112-dim input to the model dimension

A learned [CLS] token is prepended, learned absolute positional embeddings are added, and the resulting sequence is passed through 4 pre-norm Transformer encoder layers (4 attention heads, FFN width 4×128, dropout 0.1, GELU). The [CLS] hidden state at the final layer is mapped to 4 logits via a linear head $W_{\text{out}} \in \mathbb{R}^{4 \times 128}$.

Total parameters: **842,852**. Of these:
- Phone embedding: 22 × 64 = 1,408
- Accent embedding: 4 × 16 = 64
- Boundary embedding: 5 × 16 = 80
- Continuous projection: (4 × 16) + 16 = 80
- Input projection: (112 × 128) + 128 = 14,464
- Position embedding: 256 × 128 = 32,768
- 4 transformer layers: ~789K
- LayerNorm + classification head: ~700

Note that the prosody-related parameters (accent embed + boundary embed + continuous projection) total **224 parameters out of 842,852** — roughly 0.03%. This is important for fairness analysis: the architecture is essentially the same regardless of whether the prosody channels are used.

### 3.4 The prosody mask (Condition B)

The killer-experiment ablation is implemented at the input embedding level:

```python
def forward(..., with_prosody: bool = True):
    if not with_prosody:
        accent_ids = torch.zeros_like(accent_ids)        # → "NONE"
        boundary_ids = torch.zeros_like(boundary_ids)    # → "NONE"
        continuous = torch.zeros_like(continuous)
    e_phone   = phone_embed(phone_ids)
    e_accent  = accent_embed(accent_ids)
    e_bdry    = boundary_embed(boundary_ids)
    e_cont    = cont_proj(continuous)
    x = cat([e_phone, e_accent, e_bdry, e_cont], dim=-1)
    ...
```

The vocabularies are constructed so that token id 0 corresponds to `NONE` for both accent and boundary categories. When `with_prosody=False`, every position effectively has the unaccented, no-boundary, no-continuous-prosody embedding, regardless of its true value. The phone channel is unaffected.

This is a clean ablation: same parameters, same forward pass, only the input slices differ.

### 3.5 Training

Two encoders were trained, identical in architecture and parameter count, differing only in training data:

- **PILM-prosody**: trained with `with_prosody=True` for every batch.
- **Text-only baseline**: trained with `with_prosody=False` for every batch.

In both cases the underlying batch tensors are identical; the baseline simply ignores the prosody slice via the model's input-level mask.

Hyperparameters: AdamW (lr=3e-4, weight_decay=0.01), 10 epochs, batch size 64, gradient clipping at 1.0, MPS device (Apple Silicon GPU). Each model trained in roughly 50 seconds.

### 3.6 Evaluation

Three test-set conditions:

- **A**: PILM-prosody, `with_prosody=True` at inference. Upper bound — the model has all channels.
- **B**: PILM-prosody, `with_prosody=False` at inference. The killer-experiment condition. Trained with prosody, evaluated text-only.
- **Baseline**: text-only baseline, `with_prosody=False` at inference. Floor — model never saw prosody.

Bootstrap 95% CIs computed with 1,000 resamples of the test set. Per-label accuracy and confusion matrices computed for each condition.

---

## 4. Results

### 4.1 Headline numbers

Test set, n = 1000:

| Condition | Description | Overall accuracy | 95% CI |
|---|---|---:|---:|
| A | PILM, all channels | **1.0000** | 1.000 – 1.000 |
| B | PILM, prosody zeroed at inference | 0.2840 | 0.258 – 0.313 |
| Baseline | Text-only baseline | 0.4290 | 0.399 – 0.461 |

**B − Baseline = −14.5 pp.** The CIs do not overlap: B is reliably below the baseline.

### 4.2 Per-label decomposition

| Condition | STATEMENT (n=273) | QUESTION (n=284) | SURPRISED_QUESTION (n=213) | FOCUS (n=230) |
|---|---:|---:|---:|---:|
| A | 1.000 | 1.000 | 1.000 | 1.000 |
| B | **0.000** | **1.000** | **0.000** | **0.000** |
| Baseline | 0.989 | 0.560 | 0.000 | 0.000 |

The decomposition is the diagnostic. PILM-B's accuracy on QUESTION is 100% and on every other label is exactly 0%. Inspecting the confusion matrix confirms it: PILM-B predicts QUESTION for every test example. This is the signature of complete representational collapse, not a graceful fallback to the lexical signal.

The baseline, by contrast, behaves as expected for a model that has only the lexical filler to work with:
- STATEMENT 99% — the absence of a question filler is a near-perfect cue.
- QUESTION 56% — picks up roughly the half of QUESTION utterances that have a filler.
- SURPRISED_QUESTION 0% — these always (~50%) start with the same filler as QUESTION, so the baseline classifies them all as QUESTION.
- FOCUS 0% — no purely-lexical signal exists, so the baseline classifies these as STATEMENT (no filler).

The baseline's behavior is what a sane text-only classifier *should* do given this data. PILM-B's behavior is degenerate.

---

## 5. Why PILM-B failed — the math

This is a textbook case of what the multimodal-learning literature calls **shortcut learning** or **modality collapse**: when one modality is sufficient to solve the task, gradient pressure on the other modality vanishes.

Formally, the training objective is

$$
\mathcal{L}(\theta) = \mathbb{E}_{(x_{\text{text}}, x_{\text{pros}}, y)} \big[ \ell\big(f_\theta(x_{\text{text}}, x_{\text{pros}}), y\big) \big]
$$

where $f_\theta$ is the model with all channels active. The data is constructed so that there exists a function $g : x_{\text{pros}} \to y$ with $\ell(g, y) = 0$ — the prosody channel deterministically encodes the label.

Once the model finds parameters $\theta^*$ such that $f_{\theta^*}(\cdot, x_{\text{pros}}) \approx g(x_{\text{pros}})$, the gradient of the loss with respect to *any* parameter that is upstream of only the text branch goes to zero. In particular, the phone embedding $\phi$, the parts of $W_{\text{in}}$ that read from the phone slice, and any downstream attention or FFN parameters that primarily attend to text-derived features, all stop receiving gradient signal.

What the model has at $\theta^*$ on the text pathway is whatever it had at initialization, slightly perturbed by early-training gradients before the prosody shortcut was found. At inference with the prosody slice zeroed, the model receives only its (essentially untrained) text representation. The output is whatever the final classifier head produces from a near-uniform mid-network activation — empirically, in our run, it is the QUESTION class for every input.

A second observation makes this stark: the prosody embeddings (224 parameters total) are doing all the work. The model has effectively learned to be a tiny lookup from accent/boundary/continuous → label, ignoring the bulk of its 842K parameters. The phone embedding and the transformer body are barely engaged.

This is consistent with what we observe in training: PILM-prosody hits 99.6% train accuracy by epoch 3 and 100% from epoch 7 onward, with training loss dropping below 0.001. A text-only learner would never converge that fast on this task.

---

## 6. The fix — modality dropout

The standard fix in the multimodal-learning literature is **modality dropout**: during training, randomly zero one or more modalities for a fraction of examples. This forces the model to develop competence on each modality alone while preserving the ability to integrate them when both are available.

In implementation, the simplest version is per-example: for each training example, with probability $p_{\text{drop}}$, replace its prosody slice with zeros before the forward pass:

```python
drop_mask = torch.rand(B) < p_drop                # (B,) bool
accent_ids[drop_mask] = 0                         # NONE
boundary_ids[drop_mask] = 0                       # NONE
continuous[drop_mask] = 0.0
```

The model architecture is unchanged. The model's `with_prosody=True` forward path is used for every training step; the dropout is realized purely by the input-side masking.

This is the change we have folded into `docs/design_decisions.md` D9, with $p_{\text{drop}} = 0.2$ as the default. EXP-002 sweeps $p_{\text{drop}} \in \{0.0, 0.2, 0.5, 1.0\}$ on the same data and shows that as little as $p = 0.2$ fully restores text-only competence (PILM-B 0.423 vs floor 0.430 — within the 95% CI of the floor) without any cost to the upper bound (PILM-A still 100%).

The fact that even $p = 0.2$ suffices is itself informative. The model only needs to *occasionally* be forced to use the text pathway for that pathway to develop and be available at inference. Higher $p$ values give text-only competence equal to or marginally better than the baseline, which is what we would expect from the textbook-level treatment of modality dropout.

---

## 7. What this changes for Phase 5

The Phase 5 killer experiment, as originally articulated, was:

> Train PILM with prosody. Test text-only. If PILM-text-only beats a text-only baseline, the Fernyhough prediction is supported.

EXP-001 shows this question is malformed without modality dropout, because PILM-text-only is degenerate when training never exercised the text pathway.

The corrected Phase 5 question:

> Train PILM with prosody and modality dropout p ∈ (0, 1). Train a text-only baseline (matched compute). Both models have text-only competence. **Does PILM's text-only inference benefit from having seen prosody during training, beyond what the text-only baseline can extract from text alone?**

This is the actual Fernyhough/Fodor question, sharpened: not "does prosody pretraining produce text-only behavior" (yes, trivially, with dropout), but "does prosody pretraining produce *better* text-only behavior than text-only training would have."

The mechanism by which it could is exactly the implicit-prosody mechanism that Fodor documented in humans: lexical material, paired during pretraining with prosodic structure, becomes encoded with implicit-prosody-shaped representations. At inference with text alone, those representations are still active and shape the model's predictions in ways the text-only baseline cannot match.

For this to be observable, the data must have rich correlational structure linking lexical content and prosody. Synthetic data does not. EXP-002 confirms this: with weak lexical signal, PILM-B with $p = 0.2$ matches the floor exactly. EXP-004 with stronger lexical signal also fails to produce a positive Fernyhough result (B reaches 53–55% vs. floor 56%; the floor is the highest).

The genuine test waits for natural speech where the correlational structure is rich (LibriTTS clean-100, Phase 4).

---

## 8. Assumptions, caveats, and what could go wrong

### 8.1 The synthetic data's prosody is too determining

Real human prosody is noisy, speaker-variable, and only probabilistically tied to pragmatic intent. Our toy generator's prosody signal is noisy *only* in the sense of small Gaussian additive noise — the categorical accent and boundary labels are deterministic functions of the pragmatic label. This makes the shortcut to a prosody-only solution faster and starker than it would be in natural data. It's possible the natural-data setting will be less prone to the shortcut, but the safer bet is to apply modality dropout regardless.

### 8.2 Per-example dropout vs. per-position dropout

We chose per-example dropout (whole utterance prosody zeroed) for v1. Per-position dropout (each phone independently masked) is also common and may be preferable for transformers, which can learn to route attention around dropped tokens. This is left for v2 if needed.

### 8.3 The 0.0% on FOCUS for the baseline

The text-only baseline gets 0% on FOCUS. This is correct given the data: FOCUS has no lexical-only signal. The baseline's behavior on FOCUS utterances is to classify them as STATEMENT (no filler word at start). EXP-004 with focus-marker words raises this to ~28% but not to par with QUESTION-class detection.

### 8.4 The 0.0% on SURPRISED_QUESTION across all conditions when prosody is unavailable

This is intentional. SURPRISED_QUESTION shares the question-filler distribution with QUESTION; the only thing that distinguishes them is the F0 peak height at the terminal. By construction, no text-only model can disambiguate these. This makes SURPRISED_QUESTION the prosody-only contrast in our synthetic harness.

### 8.5 What if the natural-data Fernyhough test also fails?

This is the published-as-negative-result scenario (`docs/design_decisions.md` D14). It would mean: prosody pretraining does not install useful inductive biases that transfer to text-only inference. PILM is then "merely" a multimodal model — useful, but not the strong claim. A negative result would still be publishable as a benchmark / failure-mode contribution to the literature on modality transfer in speech LMs.

### 8.6 What if our auto-ToBI labeler is too noisy?

The Phase 2 auto-ToBI labeler has its own failure modes. If labels are noisy, the prosody channel during pretraining is partly garbage, and the kind of useful priors we are trying to install may not be installable. This is the chokepoint risk in the constrained-first plan.

---

## 9. Connection to broader literature

The modality-dropout fix is well-established. Some relevant references:

- **Modality dropout in audio-visual speech recognition** (e.g., Hu et al., 2016 onward) routinely use modality-specific dropout to prevent the model from collapsing onto whichever modality is easier in training.
- **Multimodal contrastive learning** (CLIP, ALIGN) implicitly enforces both modalities via the contrastive objective; modality dropout is less needed because the loss explicitly requires alignment.
- **In speech LM literature**, pGSLM (Kharitonov et al., 2022) trains on F0 and duration channels alongside semantic tokens; their ablations are informative about which channels carry which structure but they do not do an explicit modality-dropout-style analysis.

For PILM, the relevant precedent and adjacent literature is summarized in `docs/literature_review.md` sections 8 (self-supervised speech representations) and 9 (computational prosody for TTS).

---

## 10. Summary table — what was learned

| Question | Answer |
|---|---|
| Does the encoder architecture work? | Yes (Cond A = 100%). |
| Does the prosody-mask ablation function correctly? | Yes (Cond A vs Cond B logits differ; ablation is at the input level). |
| Does vanilla multimodal training produce text-only competence as a side effect? | **No** (Cond B = 28%, lower than text-only baseline 43%). |
| Does modality dropout fix this? | **Yes** (EXP-002: at p=0.2, B = 42.3% vs floor 43.0%, within CI). |
| Does the Fernyhough effect appear on synthetic data? | **No** (B never exceeds floor across dropout sweeps and lexical-signal regimes). |
| Does this falsify the Fernyhough hypothesis? | No — synthetic data is too sparse in correlational structure to support the hypothesis. The real test waits for natural data in Phase 5. |

---

## 11. Further reading inside this repo

- `docs/theory_notes.md` — the project thesis and the original killer-experiment design.
- `docs/design_decisions.md` — locked decisions (D9 now includes modality dropout).
- `docs/phases.md` — full project roadmap with Publications & Milestones.
- `docs/experiments.md` — the experiments log (short-form summaries).
- `data/synthetic/killer_test_results.json` — raw results JSON.
- `scripts/run_synthetic_killer_test.py` — current harness (now sweeps p_drop).
- `scripts/gen_synthetic_prosody.py` — generative model for the synthetic data, with `--lexical-signal {weak,strong}`.
- `models/pilm_toy.py` — encoder.
- `models/synthetic_dataset.py` — JSONL dataset and collator.
