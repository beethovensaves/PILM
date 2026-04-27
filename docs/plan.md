# PILM — Plan

_Last updated: 2026-04-26._

The single TODO. Replaces the old `TODO.md`, `phases.md`, and the "what to do
next" section of `session_handoff.md`. If a task isn't here, it's not on
the path.

## Now (this session / next session)

1. **Finish AMI v1 cross-corpus probe (EXP-010).**
   - [ ] AMI IH audio mirror to Mac mini completes (background, ~90 min remaining).
   - [ ] Pull 30-meeting subset to laptop (`scripts/pull_ami_subset.sh`, ~5 GB, ~10 min).
   - [ ] Slice subset to per-segment WAVs (`scripts/prepare_ami_for_mfa.py --meeting all-local`).
   - [ ] Run AMI v1 extractor (`scripts/extract_parametric_prosody_ami_v1.py`) → `data/ami/parametric_prosody_ami_v1.jsonl`.
   - [ ] Build el.inf-prediction probe analogous to EXP-007 (yn-only AUC, position ablation, bootstrap CIs).
   - [ ] Same probe on MELD using the v1 outputs (`data/meld/parametric_prosody_*_v1.jsonl`, already done).
   - [ ] Document as **EXP-010** in `findings.md`.

2. **Verdict on EXP-010 → Phase 4 architecture commit.**
   - If AMI prosody-only AUC for el.inf is meaningfully above MELD's 0.65 → corpus floor confirmed; the "PILM thesis works on natural conversation" framing solidifies.
   - If AMI ≈ MELD → method floor; need richer model (Phase 4) to extract more signal.
   - Either way: lock D21–D23 architecture and proceed to Phase 4 build.

## Next (Phase 4 — build the model)

3. **`models/pilm_v1.py`** — encoder-only transformer with the new architecture:
   - Input: phone token (256d embed) + 18-d parametric projection (64d) + 6-d position features (D23).
   - Per-layer prosody-modulated attention bias (D21): `score(i,j) += α_h · MLP(prosody_j)`.
   - Output heads: phone CE + parametric MSE.
   - Modality dropout p=0.2 (D9).
   - ~50M parameters target. 6 layers, 8 heads, 384d hidden.

4. **`scripts/pretrain_pilm.py`** — pretraining loop on the union of
   MELD parametric + AMI parametric (~14k segments). Single A100 / M-series
   for v1; ~24 hours.

5. **Phase 4 probing** — frozen encoder, linear probe heads on:
   - el.inf detection (AMI)
   - emotion / sentiment (MELD)
   - speaker-held-out splits for GroupKFold robustness (per EXP-007b lesson)

## Pinned (waiting on external)

6. **UW LDC institutional access** for LDC97S62 (Switchboard-1 audio) and
   LDC2009T26 (NXT annotations). Once cleared:
   - Download to `data/switchboard/{audio,nxt}/` (~4.5 GB).
   - Use existing `scripts/nxt_xml_reader.py` (will be written for AMI;
     same NXT format → drop-in) to parse annotations.
   - Run AMI-style audio slicing + prosody extraction.
   - This unblocks **Phase 5 (the killer experiment)**.

## Phase 5 (killer experiment, conditional on Phase 4 + LDC)

7. Compare PILM-with-prosody-trained-and-modality-dropout vs same-compute
   text-only baseline. Both probed text-only at inference on:
   - Switchboard NXT dialog acts
   - NXT kontrast (focus / contrast)
   - MELD emotion
   - AMI el.inf
   - Statement-vs-question minimal pairs

   Negative result is publishable. (D14 commitment.)

## Background, when bandwidth allows

8. **Streamlit annotation tool (D17)** — UI scaffolding can begin without data.
9. **Switchboard NXT XML parser** — schema in Calhoun et al. (2010); reference
   notebooks in `emeinhardt/switchboard-lm`. Already ~50% done in
   `prepare_ami_for_mfa.py` (same NXT format).
10. **Tier 1 paper read** — Breen 2014 + Alderson-Day & Fernyhough 2015 (both
    in `litt/`).
11. **Browser-download** the 7 paywalled papers from `litt/README.md` (UW VPN).

## Deferred to later phases

- Cross-attention architecture revisit (Phase 4+ if D7 underperforms).
- Decoder activation + discrete prosody auxiliary output head (Phase 7).
- Multilingual extension (Phase 8).
- Disentanglement losses (Phase 4+ conditional on probing results).
- Reading-time psycholinguistics test (Phase 9.10).
- Curated pragmatic minimal-pair corpus as community resource (Phase 9.8).

## Done (not stale, kept as cross-session anchor)

- Phase 0 framing, Phase 1 synthetic harness, Phase 1.5 MELD parametric
  validation, Phase 1.5+ follow-up batch (EXP-007b validations + EXP-009
  anger diagnostic + AMI scoping).
- Cross-corpus text-only baselines (`scripts/predict_da_from_text_ami.py`,
  `scripts/predict_question_from_text.py`).
- v1 parametric extractor rerun on MELD all splits (apples-to-apples for
  EXP-010).
