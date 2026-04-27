# PILM — Persistent TODO

_Canonical action list. Updated as work progresses. Survives session ends and conversation compaction._

_Last touched: 2026-04-25 (parametric prosody pivot — Phase 1.5 sprint plan; D5/D9/D10/D19/D20 revised)_

---

## 🅿 Pinned (waiting on external dependency)

- [ ] **Confirm UW LDC institutional access** for both LDC97S62 (Switchboard-1 audio) and LDC2009T26 (NXT annotations). User is currently confirming. Until this clears, Phase 2 data work is gated.

## ▶ In progress / next sprint

- [ ] **Phase 1.5 — parametric prosody pipeline validation on MELD dev** (unblocked once MELD finishes downloading; see `docs/phases.md` Phase 1.5). Reframed from earlier "AuToBI smoke-test" — AuToBI is now the *probe target* not the label source. Concretely:
  - [ ] After `data/meld/MELD.Raw.tar.gz` finishes: `tar -xzf MELD.Raw.tar.gz "MELD.Raw/dev_splits_complete" "MELD.Raw/dev_sent_emo.csv"` then delete the tarball. Net disk ~1.1GB.
  - [ ] **`scripts/extract_parametric_prosody.py`** — Parselmouth-driven extractor for the 18-dim per-syllable parametric vector (D19). Per-speaker baselining per D6.
  - [ ] **`scripts/run_autobi_on_meld.py`** — runs AuToBI on the same syllables; outputs are *probe target*, never training input.
  - [ ] **`scripts/validate_parametric_prosody.py`** — train a linear/MLP probe over parametric vectors → predict AuToBI accent presence/absence. Gate: F1 ≥ 0.65.
  - [ ] Hand-spot-check 10 MELD clips against the parametric extractor's outputs (F0 contour, peak alignment, boundary signals) for sanity.
- [ ] **Phase 2.2 — NXT parser scaffolding (can begin without LDC data).** Concretely:
  - [ ] `scripts/nxt_xml_reader.py` — parses NXT multi-layer XML into per-conversation Python objects (terminals, accents, breaks, kontrast, dialog acts, syllables, phones). NB: ToBI accents/breaks become *probe targets*, not training labels (D5).
  - [ ] `scripts/sph_to_wav.py` — wraps `sph2pipe`/`sox` for batch conversion (gated on data, but the script can be written).
  - [ ] **Reuse** `scripts/extract_parametric_prosody.py` from Phase 1.5 — same 18-dim per-syllable vector spec, just on stereo Switchboard audio with per-channel-as-per-speaker baselines.
  - [ ] `models/switchboard_dataset.py` — emits per-phone JSONL with phone-tier segmental + replicated syllable parametric vector (per D7), parallel ToBI probe-target stream.
- [ ] **Phase 2.3 — single-conversation sanity loop** (gated on LDC data).
- [ ] **Phase 3.1 — Streamlit annotation app.** Can scaffold UI without data; AuToBI wired as the pre-annotation backend for active-learning loop. Integration with NXT once data arrives.
- [ ] **Phase 3.2 — package `pilm-prosody-frontend` (pip / Hugging Face release).** Replaces the dropped supervised auto-ToBI labeler (D10). Same community-infrastructure role, simpler implementation.

## Open user actions

- [ ] Browser-download the **7** remaining Cloudflare-blocked / login-walled papers (UW VPN; landing URLs in `litt/README.md`). PNAS 2025 + Pierrehumbert 1980 thesis are now auto-downloaded via `scripts/download_paywalled.py`. **Tier 1 priority: Breen 2014 and Alderson-Day & Fernyhough 2015.**
- [ ] Read Tier 1 lit before reviewing Phase 4 model design.
- [ ] Investigate UW faculty fit for the pitch (pre-fill `docs/uw_pitch.md` "Why UW" section).
- [ ] Confirm or redirect Phase 9 bet priorities (currently leaning: 9.1 inner-speech computational model, 9.6 prosody for AI alignment, 9.10 reading-time psycholinguistics).
- [ ] Re-read the lit review and flag anything off (still open from earlier turn).
- [ ] Resolve the Keynote-LFS issue blocking remote pushes (1 commit `f7cd4b1` is local-only; see "blocked work" below).

## Deferred

- [ ] **Beat Cloudflare bot-management** in `scripts/download_paywalled.py`. Playwright (headed Chromium, UW VPN, AutomationControlled disabled) still gets gated by Cloudflare on Wiley, Tandfonline, ScienceDirect. Real browsers pass. Options: install `playwright-stealth`, or use `undetected-chromedriver`. Worth revisiting if we need to refresh the lit pull at scale (e.g. paper-writing time). For now, 7 papers remain manual.
- [ ] **Dimensionality reduction on D19 parametric vector.** Run PCA on parametric vectors across MELD dev; report effective rank. If effective rank < 12, prune redundant dims. Phase 4 cleanup task.
- [ ] **Phrase-level digest features.** If Phase 4 probing shows poor dialog-act recovery, add a 4-dim "phrase digest" at every break point (mean F0, range, declination slope, speech rate). Cheap post-hoc add.
- [ ] Decide compute partner (UW department vs Lambda Labs) — defer until Phase 5 result.
- [ ] **Cross-attention vs concatenation, revisited.** If concat underperforms in Phase 4 probing, run a second architecture experiment with phone-stream / syllable-stream cross-attention.

## Blocked work

- [ ] **Local commit `f7cd4b1` (PAT→PILM rename + Phase 0/1) cannot push** — parent vscode repo has prior commits with two Keynote files >100 MB, GitHub rejects them. Three resolution paths in the prior turn's notes:
  1. `git lfs migrate import --include="*.key"` then force push (recommended, history-rewriting, requires explicit user authorization).
  2. Filter-repo to drop the keynote_test paths.
  3. Cherry-pick PILM commit onto a clean branch and push that.

## Watching

- [ ] ProsodyLM follow-ups — recheck arXiv quarterly. We're critiquing this paper structurally; if they release an updated version that addresses the granularity issue, the framing may need to shift.
- [ ] **pGSLM successors / frame-level prosody work** — relevant to D20 ablation framing. SpiritLM v2 in particular if it adds explicit prosody handling.
- [ ] New speech-LM releases (Moshi successors, etc.) that may shift the competitive landscape.

## Done (kept for cross-session context)

### Phase 0 (foundations)

- [x] All framing docs written (theory, lit review, design decisions, phases, uw_pitch).
- [x] Project renamed PAT → PILM (directory + all in-content references; `fst/` archived; WFST formally dropped).
- [x] Environment scaffolding: `pyproject.toml`, `scripts/setup_env.sh`, `.gitignore`.
- [x] Literature corpus: 21/30 PDFs auto-downloaded into `litt/`. Remaining 9 require browser download.

### Phase 1 (synthetic toy world)

- [x] **Synthetic prosody generator** (`scripts/gen_synthetic_prosody.py`). Weak/strong lexical-signal regimes. Status: kept as Phase 1 reference / smoke test.
- [x] **Toy encoder** (`models/pilm_toy.py`). 842K params, per-position concatenation, `with_prosody` ablation. Architecture validated.
- [x] **Synthetic dataset / collator** (`models/synthetic_dataset.py`).
- [x] **Killer-experiment harness** (`scripts/run_synthetic_killer_test.py`) with dropout sweep + bootstrap CIs.
- [x] **EXP-001** (vanilla, no dropout): A=100%, B=28% — modality collapse diagnosed.
- [x] **EXP-002** (dropout sweep on weak data): p=0.2 fully fixes the collapse at no upper-bound cost. No Fernyhough effect on synthetic data (expected).
- [x] **EXP-004** (strong lexical signal): floor rises to ~56%; PILM-B still doesn't exceed floor. Real test waits for natural data.
- [x] **D9 updated** to lock prosody dropout p=0.2 into v1 pretraining.
- [x] **Long-form writeup** (`docs/writeups/exp001_modality_collapse.md`).

### Architecture / scientific decisions locked through 2026-04-25 (post-parametric-pivot)

- Encoder-decoder; v1 trains encoder + heads only. Decoder activates Phase 7.
- **Per-position concatenation: `[phone_embed ⊕ syllable_param_proj]`** with the 18-dim parametric vector replicated to all phones in a syllable (D4, D7).
- **Prosody representation: 18-dim per-syllable parametric vector (D19)**, all per-speaker normalized (D6).
- **ToBI labels are probe targets only, never training signal (D5).** AuToBI used as probe wrapper, not labeler.
- **Pretraining loss: masked phone prediction + masked parametric vector regression (D9).** Categorical accent/break-index losses dropped.
- **Deterministic Parselmouth-driven prosody extractor (D10).** Custom supervised auto-ToBI labeler dropped.
- **pGSLM frame-level F0 baseline ablation in Phase 5 (D20).**
- Modality dropout p=0.2 during pretraining (D9 / EXP-001 lesson).
- Dataset stack: MELD parametric validation (Phase 1.5) → Switchboard NXT (Phase 2) → real-world third (D15).
- 2-axis label space: speech_act × affect, plus optional focus_word_idx (D16).
- Annotation tool: Streamlit web UI, TextGrid + JSON output (D17).
- Pretraining is fully unsupervised; pragmatic + ToBI labels are probing-only (D18).
- WFST: dropped.

## Decisions explicitly deferred (from `design_decisions.md`)

- Cross-attention vs concatenation revisit (if Phase 4 probing underperforms).
- Decoder activation (Phase 7).
- Multilingual extension (Phase 8).
- Disentanglement losses (revisit if Phase 4 probing shows speaker info leaks into prosody channel).
- Curated pragmatic minimal-pair corpus as community resource (Phase 9.8).
- Inner-speech / silent-reading psycholinguistics test (Phase 9.10).
