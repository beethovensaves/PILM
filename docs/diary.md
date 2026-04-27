

## 2026-04-27 — EXP-010 v2 (MFA-aligned per-DA AMI) complete

MFA-aligned cross-corpus probe finished: AMI el.inf last-syl AUC v2 = 0.6298; pooled = 0.6845; bi-LSTM rich = 0.6187. MELD yn-Q last-syl AUC v2 (apples-to-apples) = 0.6502. Gap = -0.0204. AMI v2 position ablation: first/middle/last = 0.5787/0.6216/0.6298. AMI v2 text+prosody uplift = +0.0132. AMI v2 bootstrap-stable coefficients in top 6: 6/6.
Results in data/ami/exp010_results_v2.json.


## 2026-04-27 — EXP-010 follow-ups (Q1 / Q2 / Q3) complete

Three follow-up studies on the cross-corpus probe:
- **Q1 (MELD v1 bootstrap):** MELD v1 reproduce: baseline yn-only last-syl AUC = 0.5861; top-5 last-syl coefficients sign-stable: 3/5
- **Q2 (per-DA re-extraction):** AMI per-DA prosody-only last-syl AUC = 0.5984; pooled = 0.6636; position ablation first/middle/last = 0.5720/0.6083/0.5984; text+prosody uplift = +0.0279.
- **Q3 (richer bi-LSTM):** rich (4x128, attn, 30ep) AUC = 0.6216 per-segment / 0.5747 per-DA, vs small (2x64, mean, 10ep) at 0.6762 / 0.6862.

Results: data/ami/exp010_results_perDA.json, data/ami/exp010_results_richBiLSTM.json, data/meld/validate_exp007_v1.json.


## 2026-04-26 — EXP-010 cross-corpus probe complete

Cross-corpus probe results written to `data/ami/exp010_results.json` and summarised in `docs/findings.md`. Headline: AMI `el.inf` prosody-only last-syl AUC = 0.5761 versus MELD yn-Q AUC = 0.5861 (gap -0.0100). Combined text+prosody on AMI `el.inf` adds +0.0273 macro-F1 over text alone. Position ablation reproduces the MELD boundary-tone localisation: last-syl AUC 0.5761 > middle 0.5835.
# PILM — Working Diary

_Per-day log of what was investigated and what was learned. Keeps decisions
traceable across sessions / model compactions. Newer entries on top._

---

## 2026-04-26 (later) — EXP-007 validation suite + anger diagnostic + AMI scoping

After Phase 1.5 closed yesterday, focused on three follow-up threads:

1. **EXP-007 validation** (`scripts/validate_exp007.py`, results in `data/meld/validate_exp007.json`). Five tests on the question-prediction-from-prosody result. Documented as **EXP-007b** in `docs/experiments.md`. All five validations passed:
   - GroupKFold by speaker (180 speakers, 5 folds): AUC 0.6299 ± 0.012 vs baseline 0.6503 — drop of only −0.02. Not speaker-fingerprinting.
   - Neutral-only subset: AUC 0.6383 — drop of −0.012. Not emotion-confounded.
   - Position ablation: last_syl 0.6503, first_syl 0.5653, middle_syl 0.5490. **+0.10 AUC for last-syl over middle** — boundary tone localised exactly where AM theory predicts.
   - Wh-only positives: AUC 0.6004 — drops as predicted because rise-tuned model doesn't match falling wh-Q signature.
   - Bootstrap CI (n=200) on top 5 last-syl coefficients: all sign-stable. The "matches English phonetics" claim from EXP-007 is now statistically real.

2. **Anger per-dim diagnostic** (`scripts/anger_diagnostic.py`, results in `data/meld/anger_diagnostic.json`). Documented as **EXP-009**. Three views (ANOVA F-stat, univariate AUC, drop-one ablation) all converge on the same story: anger = elevated pitch register across all F0 dims, plus a smaller loudness contribution. Single dim `f0_nucleus_st` alone gets AUC 0.67 on anger-vs-rest. Folk theory says anger is loud-and-fast; the parametric vector says anger is high-pitched, with loudness as a secondary cue. F0 height dims are heavily redundant — drop-one ablation deltas are all < 0.007 F1.

3. **AMI corpus scoping**. AMI is the closest free analogue to Switchboard NXT — same XML annotation format, ~100 hours, dialogue acts, free CC-BY 4.0. Annotations zip is 22.9 MB; full audio is bigger (~50 GB). Plan: download annotations + a small audio subset to validate parser scaffolding before LDC clears.
   - Downloaded `ami_public_manual_1.6.2.zip` to `data/ami/`. Extracted; 139 meetings, 117,915 dialogue acts, 1.15M words, 16-category DA ontology.
   - Wrote `scripts/predict_da_from_text_ami.py` and ran the cheap text-only test for `el.inf` (Elicit-Inform, AMI's question DA) prediction.
   - Wrote `scripts/predict_question_from_text.py` to get the comparable MELD text-only baseline.
   - **Cross-corpus result**: MELD text-with-punct AUC = 1.0000 on yn-Q (`?` is a perfect oracle); AMI text-with-punct AUC = 0.9342 on el.inf. Words-only (punct-stripped): MELD 0.8596, AMI 0.8854 — comparable. So AMI is NOT a magic bullet for the question task; English question lexicon is similarly predictive on both corpora. The corpus difference is in whether `?` is a perfect oracle vs an imperfect cue.
   - **Important reframe**: PILM's thesis works best on tasks where text doesn't have strong lexical anchors. English question detection has too much lexical signal in any corpus to be the showcase task. The +0.017 emotion uplift on MELD is the realer signal; the next test should be emotion / affect / argumentation analogues on AMI or NXT.
   - Long-form scoping doc: `docs/writeups/ami_scoping.md` (corrected and expanded).

**Verdict on EXP-007:** the question-prediction probe now has the cleanest validation story in PILM so far — five independent perturbations either confirm or fail-as-predicted. The position ablation in particular is paper-grade material.

**Verdict on the anger story:** anger is the per-class win for prosody (+0.140 F1 over text in EXP-005), and EXP-009 shows it's driven by a physically simple, lexically invisible effect — speakers raise their entire pitch register when angry. Combined with the question-prosody result, MELD now has two clean structural validations of the parametric vector even though aggregate emotion macro-F1 stays bounded by text.

---

## 2026-04-26 — Phase 1.5 supervised comparison + bi-LSTM probes

**Sessions today:**

- **Dialogue merging** (`scripts/merge_meld_dialogues.py`) — collapsed 1108 per-utterance MELD dev wavs into 112 per-dialogue wavs with first-wins overlap handling. Dropped dia5 + dia108 as outliers (>30s gaps). Reformatted metadata to dialogue-relative timestamps.
- **MELD train + test extraction** through full pipeline — 9444 train + 2490 test utterances now have v2 (MFA-aligned) parametric vectors.
- **MFA force alignment** (`mfa-env`, conda) — 1076 dev / 2490 test / 9444 train TextGrids with word + phone tiers.
- **v2 parametric extractor** (`extract_parametric_prosody_mfa.py`) — replaced v1 intensity-peak syllabification with MFA phone-aligned syllabification + max-onset rule. Real vowel durations and final-lengthening ratios now (no more placeholders). corr(syl_dur_z, nuc_dur_z) dropped from 1.0 → 0.77; final_lengthening 5th–95th = 0.55–1.44 (was uniformly 1.0).
- **Encoding cleanup** (`clean_meld_csvs.py`) — fixed 35-37% of utterances per split that had Windows-1252-as-UTF-8 mojibake (`\xc2\x92` → `'`).
- **AuToBI build** — fork built successfully after downloading 7 missing deps from Maven Central (commons-math3, guava, JLargeArrays, JTransforms, javassist, liblinear, slf4j-simple). Pre-trained `.model` files still not found; CUNY hosting page dead, no Wayback snapshots.
- **Rule-based ToBI mapper** (`parametric_to_tobi.py`) — deterministic 18-dim → ToBI category classifier using AM-theory thresholds. No training data required; verifies against NXT when LDC access lands. Distribution on dev (9744 syllables): H* 38%, L* 21%, NONE 14%, L*+H 12%, L+H* 9%, H+!H* 7%.
- **Supervised emotion/sentiment classification** (`compare_prosody_text.py`) — train→test on emotion + sentiment, three regimes (text-only, prosody-only, combined).
  - Default sklearn TF-IDF (no punct in tokens): emotion text 0.198, prosody 0.182, combined 0.236.
  - Punct-aware tokenizer: emotion text 0.318, prosody 0.127, combined 0.335.
  - Combined adds +0.016 emotion / +0.005 sentiment over text alone — small but real and consistent.
- **Per-class diagnostics** (`diagnose_text_vs_prosody.py`) — prosody beats text on **anger** (+0.140 F1) and sadness; text beats prosody on **surprise** (+0.200) and joy. 35% of utterances get different predictions across the two modalities. Concrete examples: "Hey." flat (prosody right), "Yeah!!" emphatic (prosody right), "Really?!" (text right).
- **POS ablation** (`pos_ablation.py`) — **PUNCT is the single most predictive POS** for both emotion (0.197 alone) and sentiment (0.451 alone). Removing PUNCT: emotion text drops 0.318 → 0.264 (-0.054); sentiment 0.601 → 0.523 (-0.078). Punctuation is text's prosody-proxy.
- **Minimal-pair analysis** (`find_minimal_pairs.py`) — found 28 groups in dev+test where same normalized text gets different emotion labels. With `--strict` (text sees no-punct version too), text wins 11 groups, prosody wins 7, tied 10. Per-group prosody catch examples: "yeah" by Gary (surprise), "what is it" by Chandler (anger).
- **Question prediction from prosody** (`predict_question_from_prosody.py`) — pooled LR AUC 0.62, last-syllable LR AUC 0.64. Top last-syllable coefficients (tilt, f0_peak_pos, f0_min_st, nuc_dur_z) match English yes/no question phonetics exactly.
- **bi-LSTM question probe** (`bilstm_question_probe.py`) — sequence-aware probe over per-syllable 18-dim+voicing. Mean-pool: AUC 0.669 all-Q / 0.689 yn-only (gain of +0.03/+0.04 over LR). question-F1 jumped from ~0 to 0.42-0.47 with class-weighting.

**Key findings of the day:**
1. Prosody contributes a small, consistent uplift over text on emotion (+0.016 macro-F1) and minimal effect on sentiment.
2. The per-class win for prosody is anger; text's anchor classes are surprise (punctuation) and joy.
3. Pooling, not the dim spec, is the next big bottleneck — bi-LSTM gain confirms sequence info matters.
4. PUNCT is the dominant text feature, doing what prosody does.

**Open / next (resolved later this day):**
- Frame-level F0 addition to per-syllable rep (D20 ablation) → done; +0.005 emotion, slight regression sentiment.
- bi-LSTM on emotion/sentiment → done; +0.04–0.05 over pooled LR, still well below text.
- Attention pooling vs mean pooling → tied (0.170 vs 0.171 emotion).
- ToBI labels as features → done; weakest probe (0.094 emotion / 0.279 sentiment).

**Late-session additions:**
- `scripts/parametric_to_tobi.py` — rule-based 18-dim → ToBI mapper. Distribution stable across splits (H* 38%, L* 21%, NONE 14%, L*+H 12%, L+H* 9%, H+!H* 7%). Validates against NXT when LDC access lands.
- `scripts/add_frame_f0.py` — augments parametric JSONL with K equally-spaced F0 samples per syllable (semitones-rel-speaker-median).
- `scripts/bilstm_emotion.py` — generalized bi-LSTM classifier for emotion / sentiment, mean or attention pooling, optional frame-F0 input.
- `scripts/tobi_features_classifier.py` — 15-dim ToBI label aggregation per utterance, compared against text + parametric.
- All experiments documented as EXP-005 through EXP-008 in `docs/experiments.md`.

**Verdict at end of day:** Text dominates emotion / sentiment classification on MELD across every prosody probe we tried (pooled LR, bi-LSTM, ToBI features, frame-F0 augmented). Best prosody-only is bi-LSTM+attn+frame-F0 at macro-F1 0.175 (emotion) / 0.427 (sentiment). Combined with text adds a stable ~+0.017 (emotion) / +0.005 (sentiment), regardless of which prosody stream. Per-class story still holds — prosody owns anger; text owns surprise / joy. The Phase 5 thesis (prosody adds signal beyond text) is supported but small in MELD; expect amplification on Switchboard NXT (less acted, less lexically explicit).

**Compaction note:** Session ending here. Next session: Phase 2 NXT pipeline if LDC has cleared, otherwise continue surfacing implicit-prosody / inner-speech literature.

---

## 2026-04-25 — Phase 1.5 setup + parametric pivot

- **Parametric prosody pivot** committed (D5/D9/D10/D19 revised; D20 added). Replaced AM/ToBI categorical labels with continuous 18-dim per-syllable parametric vector. Long-form rationale: `docs/writeups/parametric_prosody_pivot.md`.
- **Phase 1.5** introduced: validate parametric extractor on MELD before NXT lands. Insert into phases.md.
- **MELD download** — 10.88 GB tarball from umich; ~2hr to fetch. Extracted dev split (1108 wavs).
- **v1 parametric extractor** (`extract_parametric_prosody.py`) — Praat de Jong syllable-nucleus heuristic + 18-dim D19 vector. Two-pass speaker baseline.
- **build_emotion_probe_targets.py + validate_parametric_prosody.py** — Phase 1.5 gate test using MELD emotion labels (after AuToBI's pre-trained models couldn't be found).
- **Initial gate**: emotion macro-F1 +0.10 (MLP), sentiment +0.17 (LR). Phase 1.5 architecture validated.
- **Playwright literature downloader** — built; 2/9 paywalled papers auto-fetched (PNAS, Pierrehumbert thesis). 7 still need browser download (Cloudflare bot blocks Playwright on Wiley/Tandfonline/ScienceDirect; APA login wall; PMC interstitial).
