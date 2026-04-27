# Prosody-Internalized Language Models (PILM)

> **Research in progress.** This repository accompanies an in-development
> position paper. APIs, dim layouts, and conclusions are subject to change.
> Empirical results are reproduced in `docs/findings.md`; the proposal in
> `docs/proposal/` is a living document.

PILM tests a single, falsifiable claim: **installing a parallel sub-word
prosodic tier at training time produces inductive biases that survive at
inference, even on text-only inputs.** If true, this is a computational
test of the developmental observation that humans internalize prosodic
structure as a layer of meaning rather than a peripheral acoustic cue
(Spelke, Prieto, Snedeker, and others).

## Theoretical background

The thesis sits between two existing literatures:

- **Spoken-LM work** (pGSLM, AudioPaLM, Spirit-LM) preserves prosody
  *at the unit level*, not as a structural representation. None of these
  test whether prosodic information persists in text-only inference.
- **Text-only LMs** are presumed to recover punctuation-mediated prosody
  implicitly, but no controlled comparison isolates this against an
  explicit prosodic tier installed during training.

PILM proposes a discovery experiment with five matched encoders
(text-only, random-prosody control, and three modes of prosodic
perception ranging from bottom-up acoustic features to learned
HuBERT-style units). At inference time, all five take **text only**.

See `docs/proposal/pilm_proposal.pdf` for the current proposal draft and
`docs/PILM.md` for the project overview.

## Status

| Phase | Status | Notes |
|---|---|---|
| 0 — foundations | ✅ done | Environment, baseline encoder, synthetic data generator. |
| 1 — synthetic toy world | ✅ done | Architecture validated. Modality-collapse failure mode diagnosed; fixed via dropout p=0.2 (D9). |
| 1.5 — naturalistic prosody pilot | ✅ done | MELD + AMI cross-corpus probe (EXP-007/010). 22-dim parametric vector validated; boundary-tone localisation confirmed via MFA-aligned per-DA AMI extraction. |
| 1.5+ — v3 spec | 🔄 in progress | 26-dim vector (+4 voice-quality, +4 phrase-digest). Re-extraction running on MELD train/test/dev + AMI per-DA. |
| 2 — Switchboard NXT | 🅿 pinned | Awaiting UW LDC institutional access. |

## Repository layout

```
PILM/
├── docs/
│   ├── PILM.md                 — project overview
│   ├── plan.md                 — phased roadmap
│   ├── decisions.md            — locked architectural choices (D1–D19)
│   ├── experiments.md          — empirical findings log
│   ├── findings.md             — current results dashboard
│   ├── diary.md                — running session notes
│   ├── uw_pitch.md             — one-page compute pitch
│   ├── proposal/               — position paper (LaTeX + PDF)
│   ├── writeups/               — long-form analyses
│   └── archive/                — superseded plans + lit review
├── models/
│   ├── pilm_toy.py             — Phase 1 encoder
│   ├── synthetic_dataset.py    — Phase 1 dataset
│   └── prosody_frontend/       — front-end stub (will be replaced)
├── scripts/                    — extractors, probes, orchestrators
├── fst/                        — archived (WFST formally dropped)
└── pyproject.toml
```

The `data/` and `litt/` trees are gitignored; this repo is code +
documentation only.

## Setup

```sh
./scripts/setup_env.sh
source .venv/bin/activate
```

`pyproject.toml` declares phase-aware dependency extras. Phases 0–1 need
only the core dependencies; later phases pull in acoustic / ML /
training extras.

Phase 1 encoder smoke test:

```sh
.venv/bin/python -m models.pilm_toy
```

Phase 1 dropout-sweep harness (~3 min on M-series MPS):

```sh
.venv/bin/python -m scripts.run_synthetic_killer_test
```

## Key scripts

| Script | What it does |
|---|---|
| `extract_parametric_prosody_mfa.py` | MELD: 22-dim parametric prosody from MFA-aligned syllables. |
| `extract_parametric_prosody_ami_v2.py` | AMI: same extractor, driven from per-DA manifest. |
| `add_phrase_digest.py` | Post-processor adding 4 phrase-level dims (→ 26 total). |
| `voice_quality_features.py` | Creak fraction, H1−H2, CPP, spectral tilt (Parselmouth). |
| `bilstm_question_probe.py` | Probe for question-vs-statement from prosody alone. |
| `bilstm_emotion.py` | MELD emotion probe. |
| `compare_prosody_text.py` | Head-to-head prosody / text / both for cross-corpus probes. |
| `run_v3_extraction.sh` | Full v3 orchestrator (MELD + AMI). |

## Reproducing results

The empirical claims in `docs/findings.md` and `docs/experiments.md`
are reproducible from raw corpora once the data dependencies (MELD,
AMI Meeting Corpus) are obtained from their respective sources. Per-run
seeds, splits, and hyperparameters are recorded inline in each script.

The 26-dim parametric prosody spec (D19, "v3") is documented in
`docs/proposal/pilm_proposal.pdf` §3.

## Author

Felipe Imbelissieri-Casadei
University of Washington Computational Linguistics
faic@uw.edu

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

If this work informs yours before publication, please cite as:

```bibtex
@misc{casadei2026pilm,
  author = {Imbelissieri-Casadei, Felipe},
  title  = {Prosody-Internalized Language Models: Toward a Prosody-Aware
            Foundation for Language Modelling},
  year   = {2026},
  note   = {Position paper, in preparation.
            \url{https://github.com/beethovensaves/PILM}}
}
```
