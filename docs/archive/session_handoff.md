# PILM — Session Handoff

_Read this first when starting a new session. ~2 minutes._

_Last touched: 2026-04-26 — Phase 1.5 closed earlier; same day, follow-up batch added EXP-007b (5-test validation of question-prediction probe) + EXP-009 (anger per-dim diagnostic) + AMI scoping. See `docs/experiments.md` and `docs/diary.md` for details._

---

## What PILM is in one sentence

A computational test of the Fernyhough/Fodor inner-speech-as-prosody claim: pretrain a model with parallel sub-word prosodic + segmental tiers, then check whether prosodic inductive biases persist at text-only inference, beyond what a same-compute text-only model would have learned alone.

For full theory: `docs/theory_notes.md`. For locked decisions: `docs/design_decisions.md` (D1–D18). For phased plan: `docs/phases.md`. For literature: `docs/literature_review.md`.

---

## Where we are right now

**Phase 0 (foundations)** ✅ done. Framing, decisions, lit review, scaffolding all in place.

**Phase 1 (synthetic toy world)** ✅ done. Architecture validated; modality-collapse failure mode diagnosed and fixed via dropout p=0.2 (locked in D9). EXP-001/002/004 documented; long-form `docs/writeups/exp001_modality_collapse.md`.

**Phase 1.5 (parametric prosody validation on MELD)** ✅ done (2026-04-26). v2 (MFA-aligned) parametric vectors extracted for all three splits (9444 train + 2490 test + 1035 dev). Supervised classification battery comparing text vs parametric vs combined; bi-LSTM with mean/attention pooling; frame-level F0 ablation; rule-based ToBI labels — all documented as EXP-005 to EXP-008 in `docs/experiments.md`. Headline: text dominates on MELD; best prosody-only (bi-LSTM + attention + frame-F0, 35 dim) is macro-F1 0.175 emotion / 0.427 sentiment vs text's 0.318 / 0.601. Combined adds +0.017 emotion / +0.005 sentiment over text alone, regardless of which prosody stream. Per-class diagnostic confirms prosody owns anger (+0.140 F1); text owns surprise (+0.200) / joy.

**Phase 1.5+ follow-up batch** ✅ done (same day). Three additions:
- **EXP-007b** — 5-test validation of EXP-007's question-prediction-from-prosody result. All five passed. Strongest: position ablation shows last-syllable AUC = 0.65 vs middle-syllable 0.55 vs first-syllable 0.57 — boundary tone localised exactly where AM theory predicts. Bootstrap CIs (n=200) show top 5 last-syl coefficients all sign-stable. The "matches English yn-Q phonetics" claim is now statistically real, not rhetorical. Code: `scripts/validate_exp007.py`. Results: `data/meld/validate_exp007.json`.
- **EXP-009** — anger per-dim diagnostic. Three views (ANOVA F-stat, univariate AUC, drop-one ablation) converge: anger raises the entire F0 register (`f0_nucleus_st` alone gets AUC 0.67 on anger-vs-rest), with `rms_max_z` and `f0_min_st` as load-bearing complementary dims. Folk theory says anger is loud-and-fast; data say anger is high-pitched. Code: `scripts/anger_diagnostic.py`.
- **AMI scoping** — annotations downloaded (139 meetings, 117k DAs, 1.15M words) at `data/ami/ami_annotations/`. Native NXT XML format, so the same parser will work for AMI and Switchboard NXT. Cross-corpus text baseline (`scripts/predict_da_from_text_ami.py`, `scripts/predict_question_from_text.py`) shows MELD `?` is a perfect oracle (text-with-punct AUC = 1.0) while AMI's text-with-punct = 0.93; words-only is similar across corpora (~0.86–0.89). **Important reframe:** PILM's thesis lands on emotion/affect, not question detection — English question lexicon is too predictive in any corpus for prosody to dominate. Long-form: `docs/writeups/ami_scoping.md`.

**Phase 2 (Switchboard NXT pipeline)** 🅿 pinned. Waiting for UW LDC institutional access to LDC97S62 (Switchboard-1 audio, ~4 GB) and LDC2009T26 (NXT annotations, ~50 MB). Parametric extractor + rule-based ToBI mapper ready to run on NXT day-one. AMI is now a credible Phase 2 hedge / preview corpus.

---

## What changed in the last sessions (two redirections to know about)

### Redirection 1 (earlier this day): NXT replaces LibriTTS

- **Switchboard NXT replaces LibriTTS as the v1 primary corpus.** Reason: NXT has gold ToBI on ~63 conversations of real conversational dialogue, plus dialog acts and focus/contrast (kontrast) annotations. LibriTTS is read-speech monologue and doesn't exercise pragmatic inference.
- **Pretraining is now strictly unsupervised** (D18). Pragmatic labels are probing-only.
- **Label space refactored** (D16): 2-axis `speech_act × affect` + optional `focus_word_idx`.
- **Annotation tool planned** (D17): Streamlit, TextGrid + JSON, active-learning hook.

### Redirection 2 (this session): the parametric prosody pivot

The bigger structural change. Detailed in `docs/writeups/parametric_prosody_pivot.md`.

- **PILM no longer trains against AM/ToBI categorical labels.** Reason: ToBI inter-annotator agreement is ~80% on accent presence and ~60% on accent type. Using ToBI as ground truth caps any model at the human-agreement ceiling. ToBI was designed in 1992 as a transcription standard for human linguists, not as an ML target.
- **Replaced by**: an 18-dim continuous parametric vector per syllable (D19). Pitch geometry (PoLaR-style) + Tilt-style event geometry + energy + duration + boundary features. All speaker-normalized (D6).
- **ToBI labels survive only as probe targets** (D5). After pretraining, we ask whether a linear probe can recover ToBI from the learned representation — that's a sanity check, not training signal.
- **The supervised auto-ToBI labeler (formerly Phase 3.2)** is dropped (D10). Replaced by a deterministic Parselmouth-driven extractor + AuToBI as a probe-target generator. Phase 3.2 now packages this extractor as `pilm-prosody-frontend`.
- **A pGSLM-style frame-level F0 baseline ablation** is added to Phase 5 (D20). Tells us whether the 18 dims capture the linguistic prosody available in F0 or whether microprosodic detail matters.
- **Phase 1.5 is new**: validate the parametric extractor on MELD with AuToBI as a probe target before committing the design to NXT pretraining.

**Long-arc data plan (D15, slightly retargeted):** MELD parametric validation (Phase 1.5) → Switchboard NXT (Phase 2) → real-world third (NPR/podcasts, hand-annotated via the tool).

---

## What to do next, in order

**While LDC access is pending** (do this work in parallel):

1. **Build `scripts/nxt_xml_reader.py` against the local AMI annotations** (`data/ami/ami_annotations/`). Same NXT format as Switchboard, so the parser is dual-purpose. Validate by reproducing AMI's DA counts / segment counts. **This is the highest-leverage next task.**
2. **Download 3 AMI meeting audios** (~500 MB total, ~40 MB/channel × 4 channels × 3 meetings). Adapt `scripts/extract_parametric_prosody_mfa.py` to AMI's per-channel WAV + word XML inputs. Run prosody pipeline on those meetings, fill the missing AMI-prosody-AUC cell for el.inf prediction.
3. **(Optional) Browser-download the 7 Cloudflare-blocked / login-walled Tier 1–3 papers** from `litt/README.md`. UW VPN is enough; just open each link and click Download. ~7 min.
4. **(Optional) Read Tier 1: Breen 2014 + Alderson-Day & Fernyhough 2015.** Theoretical foundation. Plus **`docs/writeups/parametric_prosody_pivot.md`** for the architectural reframing context.
5. **(Optional) Phase 1.5 paper writeup**, leaning on EXP-007b + EXP-009 as the structural validation evidence. Lead with position ablation as the cleanest finding.
6. **(Optional) Scaffold the Streamlit annotation app shell** with AuToBI as the pre-annotation backend.

**Once LDC access lands:**

1. Download LDC97S62 + LDC2009T26 to `data/switchboard/{audio,nxt}/` (gitignored).
2. Run sph→wav conversion.
3. Run NXT XML parser → extract one conversation's worth of (terminals, accents, breaks, kontrast, dialog acts, syllables, phones).
4. Run Parselmouth feature extraction with per-speaker (per-channel) F0/energy baselines.
5. Spot-check end-to-end on 5 conversations.
6. Then Phase 3 (annotation tool + auto-ToBI labeler) and Phase 4 (unsupervised pretraining).

---

## How to start a new session cleanly

When you `cd` into `/Users/felipe.casadei/vscode/vsclean/PILM/` and start a new Claude Code session:

1. The memory entries (auto-loaded) point at this file.
2. Read this doc.
3. Then read `docs/TODO.md` for the persistent action list.
4. The `docs/experiments.md` log captures all empirical findings to date.
5. `litt/` has the priority-read papers; the manifest in `litt/README.md` is sorted by tier.

---

## Project geography

```
PILM/
├── README.md                   # one-paragraph project intro
├── pilm_roadmap.md             # high-level overview pointing into docs/
├── pyproject.toml              # phase-aware dependency extras
├── docs/
│   ├── theory_notes.md         # Fernyhough/Fodor/AM thesis
│   ├── design_decisions.md     # locked architectural choices (D1–D18)
│   ├── phases.md               # phased roadmap with gates
│   ├── literature_review.md    # 14 sections, ~90 citations
│   ├── experiments.md          # log: EXP-001, EXP-002, EXP-004
│   ├── writeups/
│   │   └── exp001_modality_collapse.md   # long-form
│   ├── uw_pitch.md             # one-pager, held until Phase 5 lands
│   ├── session_handoff.md      # this file
│   └── TODO.md                 # persistent action list
├── models/
│   ├── pilm_toy.py             # Phase 1 encoder (842K params, validated)
│   ├── synthetic_dataset.py    # Phase 1 dataset
│   └── prosody_frontend/       # PAT-era stub (will be replaced)
├── scripts/
│   ├── gen_synthetic_prosody.py      # Phase 1 toy data generator
│   ├── run_synthetic_killer_test.py  # Phase 1 harness with dropout sweep
│   ├── setup_env.sh, download_litt.sh, etc.
│   └── (Phase 2 scaffolding goes here)
├── data/
│   ├── synthetic/, synthetic_v2/    # Phase 1 generated data + results
│   ├── switchboard/                  # Phase 2 destination (not yet populated)
│   └── (others)
├── litt/                       # downloaded paper PDFs (gitignored)
├── fst/                        # archived (WFST formally dropped)
└── .venv/                      # local virtualenv
```

---

## Critical quick references

- **Activate env:** `source .venv/bin/activate` (Python 3.11, torch 2.11, transformers 5.6, etc.)
- **Run Phase 1 smoke test:** `.venv/bin/python -m models.pilm_toy` (forward-pass smoke test on the toy encoder).
- **Run the synthetic harness:** `.venv/bin/python -m scripts.run_synthetic_killer_test` (sweeps dropout, ~3 min on M-series MPS).
- **Generate fresh synthetic data:** `.venv/bin/python scripts/gen_synthetic_prosody.py --preview 3 --lexical-signal strong`.
- **Last commit:** `f7cd4b1` (local only — push blocked by Keynote-LFS issue in parent vscode repo).
