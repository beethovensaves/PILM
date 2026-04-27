# Parametric Prosody Extractor — Implementation Walkthrough

_Written 2026-04-25 alongside `scripts/extract_parametric_prosody.py` (v1). Updated 2026-04-26 with the v2 (MFA-driven) variant in `scripts/extract_parametric_prosody_mfa.py`. Explains how the code works, what each of the 18 dimensions measures, and what corners are cut where._

**Two variants exist in the repo:**

| Variant | Script | Driver | When to use |
|---|---|---|---|
| **v1** | `extract_parametric_prosody.py` | de Jong & Wempe intensity-peak heuristic for syllables | When you don't have MFA-aligned TextGrids. Phone-aligned features are placeholders. |
| **v2** | `extract_parametric_prosody_mfa.py` | MFA word + phone TextGrids; max-onset syllabification | When MFA has run on the corpus. Real vowel durations and final-lengthening ratios. **Preferred.** |

The math for the 18 dimensions is identical between v1 and v2; only the syllable-detection and duration-computation paths differ. Sections 1–10 below describe v1; section 11 documents the v2 changes.

This is the implementation companion to two design docs:

- `docs/design_decisions.md` — D19 specifies the 18 dimensions; D6 specifies speaker normalization; D10 specifies the deterministic-extractor design.
- `docs/writeups/parametric_prosody_pivot.md` — long-form rationale for why we replaced ToBI categorical labels with this parametric vector.

If you read those two and then the script, you should not need this document. It exists to make the script easier to reason about by walking the code's structure and the math behind each dimension in plain language.

---

## 1. What the script does, in one paragraph

Given a directory of MELD utterance audio files plus a CSV of `(Dialogue_ID, Utterance_ID, Speaker)` rows, the script produces a JSONL where each line corresponds to one utterance and contains a list of detected syllables. Each syllable is annotated with the 18-dim parametric prosody vector (D19) and a voicing flag. All quantities are speaker-normalized — the same physical event sounds different in a low-pitched speaker than a high-pitched one, and our representation has to account for that explicitly.

Internally the script does two passes over the audio. First it walks every utterance to build per-speaker baselines (median F0, syllable-duration distribution, energy distribution). Second it walks them again with those baselines in hand and computes the per-syllable vectors.

---

## 2. Why two passes

We need the speaker's median F0 to express F0 in semitones-relative-to-speaker (D6). We could compute the median utterance-by-utterance, but that's lossy: a speaker who happens to ask one question (high-F0 utterance) followed by one statement (low-F0) would get baselines that drift wildly. We want the baseline to be a stable property of the speaker, computed across many utterances.

The same logic applies to the energy and syllable-duration distributions: we z-score everything against a *speaker* distribution, not against the local utterance.

So pass 1 walks all of a speaker's wavs, accumulates F0 / energy / duration samples, and computes summary statistics. Pass 2 reuses those statistics to convert every measurement to a speaker-normalized form before writing it out. The cost is two reads of the audio; the benefit is a vector whose components mean the same thing across speakers.

There's a subtle correctness issue here. We're using the dev split to compute its own baselines. That's fine in Phase 1.5 because we're validating an extractor, not training a model. In Phase 4 we'll be careful to compute baselines on each split independently (or, ideally, on the union of speaker-aligned utterances regardless of split membership).

---

## 3. How F0 extraction actually works

Parselmouth wraps Praat. We use Praat's autocorrelation-based F0 estimator (`To Pitch (cc)`):

```python
pitch = call(sound, "To Pitch (cc)", 0.005, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
```

In English: every 5 ms (200 Hz frame rate), find the F0 in Hz between 75 and 600 Hz that best explains the local autocorrelation of the waveform. Up to 15 candidates are considered per frame and a Viterbi-like path through them minimizes octave jumps. Frames where no good F0 is found are flagged as unvoiced (returned as 0).

Why those parameters:
- **75 Hz floor** — below any adult human's F0; cuts off rumble.
- **600 Hz ceiling** — above almost any adult's F0; lets us capture surprise/shouts without truncating.
- **5 ms time step** — matches our intensity time step so frame-by-frame combinations align cleanly.
- The other parameters are Praat defaults that have been validated in the literature.

The output is a `Pitch` object whose `.selected_array["frequency"]` is a NumPy array of F0-in-Hz, one entry per frame, with 0 marking unvoiced.

### Why semitones instead of Hz

Pitch perception is logarithmic: a doubling of Hz is one octave (12 semitones), and listeners hear "pitch height" as approximately log-frequency. If we encode F0 in raw Hz, an H\* in a 200 Hz speaker (peak ~280 Hz, rise of 80 Hz) and an H\* in a 100 Hz speaker (peak ~140 Hz, rise of 40 Hz) look very different in the feature vector — but they're the *same* prosodic event perceptually. In semitones-rel-speaker-median, both events look like a +5 ST rise. That's what we want.

The conversion is:

```python
semitones = 12 * log2(hz / speaker_median_hz)
```

(See `hz_to_semitones()` in the script.)

---

## 4. How syllable nuclei are detected

We need to know where the syllables are in order to compute per-syllable vectors. MELD doesn't ship with phone or syllable alignments, so we detect nuclei automatically.

The script implements a simplified version of de Jong & Wempe (2009), which is widely used in the field. The core idea: a syllable nucleus is a local intensity peak in a region with active voicing.

Steps:

1. Compute the intensity (loudness) contour at 5 ms resolution.
2. Find every local maximum.
3. Discard peaks more than 25 dB below the utterance's intensity maximum (those are noise floor, not speech).
4. Discard peaks where the F0 isn't voiced (those are aspiration, fricatives, or other non-syllabic intensity bumps).
5. Discard peaks closer than 50 ms to a kept peak (humans don't produce syllables that fast).
6. Require a 2 dB dip between consecutive peaks (otherwise the "two peaks" are one peak with a tiny dimple).

What survives is a list of nucleus timepoints. Around each nucleus we build a syllable span as nucleus ± 75 ms, but clamp the span boundaries to the midpoints between consecutive nuclei so adjacent syllables don't overlap.

This is approximate. It works well on clean read speech and reasonably on conversational dialogue. It will miss syllables in fast or noisy speech, and it doesn't know where the *vowel* is within the syllable — both limitations get fixed when we have phone alignment from NXT or MFA.

---

## 5. The 18 dimensions — what each measures, in code

I'll walk through them in vector order. The full code is in `parametric_vector()` in the script.

### 5.1 Pitch geometry — dims 0–6

These describe "what's the F0 doing inside this syllable?"

**Dim 0 — `f0_onset_st`.** F0 at the syllable's left edge, in semitones rel. speaker median. The script finds the closest *voiced* F0 sample to the syllable start time and converts it. NaN if no voiced sample exists in the syllable.

**Dim 1 — `f0_nucleus_st`.** F0 at the syllable's nucleus (the time of maximum intensity, found in §4). NaN handling identical to onset.

**Dim 2 — `f0_offset_st`.** F0 at the syllable's right edge.

These three together form an "F0 trajectory shape" — a piecewise-linear approximation of the pitch contour across the syllable. A flat statement syllable will have onset ≈ nucleus ≈ offset; a question-final syllable will have onset < nucleus < offset (rising); an L+H\* will have onset low, nucleus high, offset partway down.

**Dim 3 — `f0_max_st`.** Maximum F0 across all voiced frames in the syllable. Captures the highest the pitch reaches, regardless of timing.

**Dim 4 — `f0_min_st`.** Minimum F0 across voiced frames. Floor of the pitch within the syllable.

**Dim 5 — `f0_range_st`.** `f0_max_st − f0_min_st`. How much pitch movement the syllable contains. A syllable on a flat tone has range ≈ 0; a syllable with a sharp accent has range > 4 ST.

**Dim 6 — `f0_slope_st_per_ms`.** `(f0_offset_st − f0_onset_st) / duration_ms`. Average slope of the F0 across the syllable in semitones per millisecond. Positive on rises, negative on falls. The 'per ms' normalization means it doesn't artificially blow up for long syllables.

### 5.2 Tilt-style event geometry — dims 7–10

Pitch geometry alone doesn't tell you *when* the F0 peak occurred within the syllable. That timing distinguishes H\* (early peak) from L+H\* (late peak) in AM theory. So we add four dimensions that capture the *shape* of the F0 contour, in the spirit of Taylor 2000's Tilt model.

**Dim 7 — `f0_peak_position_norm`.** `(time of F0 peak − syllable start) / syllable duration`, ∈ [0, 1]. So a peak right at syllable onset is 0; a peak right at offset is 1. A peak in the middle is 0.5. This is what discriminates "peaks early" from "peaks late" without committing to a categorical H\*-vs-L+H\* decision.

**Dim 8 — `f0_rise_amplitude_st`.** `f0_max_st − f0_onset_st`. How much the F0 rose from the onset to the peak.

**Dim 9 — `f0_fall_amplitude_st`.** `f0_max_st − f0_offset_st`. How much the F0 fell from the peak to the offset.

**Dim 10 — `tilt`.** `(rise − fall) / (rise + fall)`, ∈ [−1, +1]. Symmetry of the F0 event:
- `tilt > 0` means the rise is bigger than the fall (peak comes near the end, like L+H\* or a question-rise).
- `tilt < 0` means the fall is bigger than the rise (peak comes early, like H\*).
- `tilt ≈ 0` means a symmetric peak.

This is a single number that captures something the categorical scheme spends multiple labels on. The information is preserved continuously and doesn't depend on a human's bucketing decision.

### 5.3 Energy — dims 11–12

F0 alone doesn't make a prominent syllable; loudness matters too. AuToBI's accent detector uses energy heavily, and we should too.

**Dim 11 — `rms_max_z`.** Peak intensity (in dB) within the syllable, z-scored against the speaker's energy distribution. A speaker who normally talks at 60 dB and produces a 70 dB syllable yields a positive z. A 50 dB whisper from the same speaker yields negative z.

**Dim 12 — `rms_mean_z`.** Mean intensity across the syllable, similarly z-scored.

We use Praat's intensity computation (`Sound.to_intensity(50, 0.005)`), which is a simple time-windowed RMS in the perceptually relevant 50 Hz–8 kHz band.

### 5.4 Duration — dims 13–14

Length matters for stress, focus, and final lengthening at phrase boundaries.

**Dim 13 — `syllable_duration_z`.** This syllable's duration (in seconds) z-scored against the speaker's syllable-duration distribution. Captures whether a syllable is "longer than usual for this speaker."

**Dim 14 — `nucleus_duration_z` (v1 approximation).** In principle this is the duration of *just the vowel*, z-scored. Vowel duration changes much more under stress and focus than syllable duration does, so it's a more sensitive feature.

In v1 we don't have phone alignment, so we approximate the vowel as 50% of the syllable (`duration × 0.5`) and compare to the speaker's average half-syllable duration. This is wrong in detail (vowel fraction varies syllable-by-syllable) but gives the model *a* dim that responds to syllable elongation in the right direction. Phase 2 with NXT phone alignments replaces this with the real vowel duration.

This dim has a `# v1 approx` marker in the code so future-Felipe knows where to upgrade.

### 5.5 Boundary — dims 15–17

These are the only three dims that look *across* syllables. They capture phrase structure that ToBI's break indices encoded categorically.

**Dim 15 — `pause_after_ms`.** Silence (in ms) between this syllable's end and the next syllable's start. Long pauses correlate with phrase boundaries (break-3) and utterance boundaries (break-4). On the last syllable of an utterance, this is set to 0 by convention (we don't know what comes next).

**Dim 16 — `final_lengthening_ratio` (v1 placeholder).** In principle: this syllable's duration divided by the mean of the preceding word's syllable durations. Final-lengthening — phrase-final syllables get longer — is a robust phrasing cue. But computing this requires word boundaries (which need forced alignment).

In v1 we set this to 1.0 (the placeholder value that means "no information"). The downstream consumer should mask this dim during Phase 1.5 validation. Phase 2 fills it in properly with MFA / NXT word alignment.

**Dim 17 — `f0_reset_st`.** F0 reset across the boundary: `next_syllable.f0_onset_st − this_syllable.f0_offset_st`. Large positive resets ("the speaker started the next phrase higher") signal phrase boundaries. NaN-safe: if either side is unvoiced, this is 0.

### 5.6 The voicing flag (companion, not in the 18)

`voiced_fraction ∈ [0, 1]` — fraction of frames within the syllable where F0 was reliably extracted. Reported alongside the vector but *outside* the 18 dims. Why outside?

Because dims 0–10 (everything F0-derived) are NaN when `voiced_fraction = 0`. We need to tell the encoder when to mask those dims. The voicing flag is that signal. The encoder treats it as an attention/masking gate, not as another regression target.

Another way to think about it: the 18 dims are the data; the voicing flag is the meta-data ("how much should you trust the F0-derived dims for this syllable?").

---

## 6. NaN handling and downstream masking

### What NaN even is

`NaN` ("Not a Number") is the IEEE 754 floating-point sentinel for "no valid value at this position." It exists because some computations don't have a meaningful real-valued answer (0/0, log of a non-positive, square root of negative) and the alternative — picking a sentinel like -999 — would silently get treated as a real number by downstream code. NaN is loud: `NaN != NaN` (it's not equal to itself) and any arithmetic with it propagates NaN. You check with `math.isnan(x)` or `np.isnan(arr)`.

Practically: NaN means "the model should not trust this slot."

### When NaN shows up in our extractor

Three causes:

1. **Unvoiced syllable** (a whispered word, a syllable that's mostly /s/ or /f/). F0 isn't defined, so dims 0–10 (everything F0-derived) are NaN. The companion `voiced_fraction` flag tells the consumer this happened.
2. **F0 not detected at the requested timepoint** even within a partly-voiced syllable. The closest-voiced-sample helper (`_closest_voiced`) returns NaN if no voiced sample exists nearby.
3. **Speaker baseline missing.** Defensive `+ 1e-9` in denominators prevents divide-by-zero, so baseline issues don't actually produce NaNs — they'd produce very large z-scores. Consumer should clip.

Last-syllable `f0_reset_st` is set to 0 (not NaN) because "no next syllable" has a meaningful "no reset" interpretation.

### Serialization and downstream masking

NaN doesn't exist in JSON, so we serialize it as `null`. The consumer (`scripts/validate_parametric_prosody.py` and Phase 4 pretraining) reads `null` and replaces it with a masked value. Two patterns work:

- **Mean imputation + binary mask**: replace NaN with the column mean, append a separate binary mask feature ("was this slot NaN?"). Keeps dim count stable, lets the model learn that masked slots have lower information.
- **Zero-fill + voicing flag**: set NaNs to 0 and rely on the `voiced_fraction` flag carried alongside. Simpler; works for our case because the F0-derived dims are the only ones that go NaN in normal operation, and they all share one cause.

We use the second pattern in the encoder. The voicing flag becomes an additional input feature (so the encoder sees 19 dims at the input — 18 + voicing) but only 18 are regressed against in the masked-prediction loss (D9).

---

## 6.5. How the 18 dims map to ToBI markings

Important framing: **we do not hard-code this mapping**. A small linear/MLP probe *learns* it during validation (Phase 1.5 gate). But the 18 dims are designed so each ToBI category corresponds to a *region* in 18-dim space — i.e., a probe should be able to recover it.

The expected signatures, derived from AM theory (Pierrehumbert 1980; Beckman & Hirschberg 2005):

| ToBI category | Expected parametric signature |
|---|---|
| **No accent** | low `f0_range_st`, low `rms_max_z`, near-zero `f0_rise_amplitude_st` |
| **H\*** (high-target peak, early/mid alignment) | high `f0_max_st`, `f0_peak_position_norm` ≈ 0.3–0.5, `tilt < 0` (peak earlier than midpoint), high `rms_max_z` |
| **L\*** (low target) | low `f0_min_st`, `f0_nucleus_st` near `f0_min_st`, `f0_rise_amplitude_st` ≈ 0 |
| **L+H\*** (rising peak, late alignment) | high `f0_max_st`, `f0_peak_position_norm` > 0.6, large `f0_rise_amplitude_st`, `tilt > 0` |
| **L\*+H** (low target with delayed rise) | low `f0_nucleus_st`, peak near offset, `f0_peak_position_norm` ≈ 0.8–1.0, large positive `f0_slope` |
| **H+!H\*** (downstepped) | high `f0_max_st` early, then drop to mid-height — best captured by *neighboring* syllables' `f0_reset_st` rather than the current syllable alone |
| **Break 0/1** (within-word / no boundary) | `pause_after_ms` ≈ 0, near-zero `f0_reset_st`, `final_lengthening_ratio` ≈ 1 |
| **Break 3** (intermediate phrase) | moderate `pause_after_ms` (~80–200 ms), moderate `f0_reset_st`, some lengthening |
| **Break 4** (intonational phrase) | long `pause_after_ms` (>200 ms), large `f0_reset_st`, strong final lengthening |
| **Boundary tone H%** (final rise) | last-syllable `f0_offset_st` high, positive `f0_slope_st_per_ms` |
| **Boundary tone L%** (final fall) | last-syllable `f0_offset_st` low, negative `f0_slope_st_per_ms` |

The Phase 1.5 gate is precisely the test that this mapping holds: if a small linear probe can recover (at least) accent presence from the 18 dims, the parametric vector encodes what AM theory says it should.

### Why this matters

We replaced ToBI categorical labels with these 18 continuous dims (D5/D19) explicitly because (a) ToBI labels carry inter-annotator noise (~80% agreement on accent presence, ~60% on accent type), (b) the underlying acoustic decisions are continuous and quantizing them throws away information. The probe is the sanity check: *did our parametric vector preserve enough information that the categorical decisions a human would have made are recoverable from it?* If yes, we get the categorical interpretability when we need it (probing) without paying the human-annotation-noise tax during training.

If the probe fails (F1 < 0.65 on accent presence), the parametric vector is missing something. Most likely additions: spectral tilt (4 dims for spectral balance), per-utterance declination context.

---

## 7. Output schema

Each line of `data/meld/parametric_prosody_dev.jsonl` is one utterance. Example (formatted for readability — actual file is single-line):

```json
{
  "utterance_id": "dia0_utt0",
  "speaker_id": "Joey",
  "audio_path": "data/meld/MELD.Raw/dev_splits_complete/dia0_utt0.mp4",
  "n_syllables": 6,
  "syllables": [
    {
      "nucleus_t_ms": 142.5,
      "t_start_ms": 67.5,
      "t_end_ms": 217.5,
      "vec": [
        2.1,    1.8,   1.4,   2.4,   1.2,   1.2,    -0.0046,
        0.42,   0.6,   1.0,   -0.25,
        0.84,   0.31,
        0.12,   0.05,
        45.0,   1.0,   -0.8
      ],
      "voiced_fraction": 0.95
    },
    ...
  ]
}
```

Vector indices are stable; if a dim is NaN it appears as `null`. `nucleus_t_ms`, `t_start_ms`, `t_end_ms` are convenience timestamps for cross-referencing back to the audio (and to AuToBI's syllable spans when we run the probe-target script).

---

## 8. Running it

```bash
# Smoke test on 5 utterances (fast):
.venv/bin/python scripts/extract_parametric_prosody.py \
    --in-dir data/meld/MELD.Raw/dev_splits_complete \
    --metadata data/meld/MELD.Raw/dev_sent_emo.csv \
    --out data/meld/parametric_prosody_smoke.jsonl \
    --limit 5

# Full dev split:
.venv/bin/python scripts/extract_parametric_prosody.py \
    --in-dir data/meld/MELD.Raw/dev_splits_complete \
    --metadata data/meld/MELD.Raw/dev_sent_emo.csv \
    --out data/meld/parametric_prosody_dev.jsonl
```

Pass 1 (baseline computation) is the slow part — it touches every audio file. Pass 2 reuses the cached audio loads conceptually but in this v1 implementation re-reads each file, so end-to-end runtime is roughly 2× the audio duration on a single core. ~1,000 MELD dev utterances at average ~3 s each ≈ 50 min wall-clock. Acceptable for a one-time pass.

---

## 9. Approximations / what we owe future-us

In priority order for future improvement:

1. **Syllable boundary detection from forced alignment.** v1 uses an intensity-peak heuristic. Phase 2 should integrate MFA (Montreal Forced Aligner) or Whisper-based alignment to get phone-level boundaries, then derive syllable boundaries via CMUdict (D4).
2. **`nucleus_duration_z` from real vowel boundaries.** Currently approximated as half-syllable. Replace once phone alignment lands.
3. **`final_lengthening_ratio` from word boundaries.** Currently a 1.0 placeholder. Replace once forced alignment + word-level grouping land.
4. **Pass 2 reuses pass 1's audio loads.** v1 re-reads files. Optimization: cache the `Sound`/`Pitch`/`Intensity` objects in pass 1 and reuse in pass 2. Not critical for Phase 1.5 (~50 min runtime is fine).
5. **MELD audio is mp4; Parselmouth reads only wav/aiff/flac.** The script handles this by lazily converting mp4 → wav via `ffmpeg` into `<in-dir>/_wav_cache/` (or wherever `--wav-cache` points). First run pays the conversion cost; subsequent runs reuse the cached wavs. Phase 2 NXT audio is `.sph` and will need `sph2pipe`/`sox` conversion (`scripts/sph_to_wav.py` per the TODO).

---

## 10. How we know it's working (Phase 1.5 gate)

Once the JSONL is written, `scripts/validate_parametric_prosody.py` (next on the TODO) does the actual validation:

1. Run `scripts/run_autobi_on_meld.py` — produces an AuToBI-derived TextGrid per utterance with categorical accent / break decisions over AuToBI's pseudo-syllables.
2. Align AuToBI's syllables to our parametric extractor's syllables (nearest-nucleus matching).
3. Train a small linear probe (sklearn LogisticRegression) on the 18-dim vector → predict AuToBI's accent presence/absence.
4. Report 5-fold cross-validated F1.
5. **Gate: F1 ≥ 0.65.** That tells us our parametric vector encodes ~most of what AuToBI uses to decide accent presence. We don't expect F1 = 1.0 because some of AuToBI's signal is microprosodic (frame-level F0 detail not in our 18 dims) and some is from features we don't include (spectral tilt, etc.). 0.65 is a "the parametric channel is doing something useful" threshold.

If we don't hit 0.65, we revisit the dimension set — most likely add spectral tilt (4 dims) and re-run.

---

## 11. Where this fits in the bigger picture

After Phase 1.5 validates the extractor on MELD, the same script (with input-format adapters) runs on Switchboard NXT in Phase 2. Then in Phase 4, the extracted parametric vectors become the prosodic input channel for PILM-small pretraining (per revised D7, D9). The extractor itself ships as `pilm-prosody-frontend` in Phase 3.2 — no model weights, just code + speaker-baseline computation logic.

The script you're reading is therefore foundational: every later phase's input data flows through it.

---

## 12. v2: MFA-driven extractor

Added 2026-04-26 in `scripts/extract_parametric_prosody_mfa.py`. Replaces the v1 syllable-detection heuristic with phone-aligned syllabification from Montreal Forced Aligner (MFA) TextGrids. Same 18-dim D19 spec, same speaker-baseline machinery, same NaN handling — only the syllable derivation and three of the duration-related dims change.

### 12.1 What MFA gives us

For each utterance wav, MFA produces a TextGrid with two interval tiers:

- **`words`** — word-level intervals (with `<unk>` for out-of-vocab).
- **`phones`** — ARPABET phones with stress digit suffix (`OW1`, `M`, `AY1`, `G`, `AA1`, `D`, `spn`, ...). `spn` = "spoken noise" / OOV phones; we treat these as non-speech.

Run via:

```bash
mfa align <wav_dir> english_us_arpa english_us_arpa <textgrid_dir> --num_jobs 8 --clean
```

Pre-requisite: each `<basename>.wav` has a `<basename>.lab` next to it with the transcript (we generate these via `scripts/prepare_mfa_transcripts.py`).

On MELD dev: 1076/1108 wavs aligned successfully (97%) in ~57 sec.

### 12.2 Max-onset syllabification

Given a sequence of phones inside a word, we split into syllables using a simplified max-onset principle:

1. Each vowel phone (ARPABET set: `AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW`, with stress digits stripped) anchors one syllable.
2. Intervocalic consonant clusters split with bias toward the next syllable's onset:
   - 1 consonant: 0+1 (all to next onset)
   - 2 consonants: 1+1 (one to coda, one to onset)
   - 3 consonants: 1+2
   - n consonants: floor(n/2) to coda, rest to onset
3. Coda of last vowel includes any trailing consonants up to word end.
4. `spn` and empty phones are skipped before splitting.

This is approximate. A phonotactic-aware splitter (sonority sequencing) would do "factor"=fac-tor while avoiding "extra"=ex-tra. The floor-half rule gets common English cases right; phonotactic refinement is a future task if probing shows the boundaries are off.

### 12.3 Three dims now use real values (was placeholder in v1)

| Dim | v1 | v2 |
|---|---|---|
| 14 `nucleus_duration_z` | `(syl_dur × 0.5 − speaker_mean × 0.5) / (speaker_std × 0.5)` — pegged to syllable_duration_z | `(vowel_dur − speaker_vowel_mean) / speaker_vowel_std` — real vowel timing from phone tier |
| 16 `final_lengthening_ratio` | uniformly `1.0` (placeholder) | `this_syl_dur / mean(syl_durs in same word)` — real intra-word lengthening signal |
| Syllable spans (used everywhere) | nucleus ± 75 ms (intensity-peak heuristic) | first-phone-onset → last-phone-offset of the syllable's phone group |

**Empirical impact on the dev split (1035 utterances, 9744 syllables):**

- `corr(syl_dur_z, nuc_dur_z)`: was **1.000** in v1 (placeholder). Now **0.771** in v2 — correlated (longer syllables tend to have longer vowels) but not redundant. The previously-placeholder dim now carries independent signal.
- `final_lengthening_ratio` distribution: was uniformly 1.0 in v1. Now 5th=0.55, 50th=1.00, 95th=1.44 in v2 — matches the phonetic literature on English final lengthening (~1.3–1.5× at phrase ends).
- Total syllables detected: ~6,700 (v1) → 9,744 (v2). MFA finds more syllables and they're phonetically real, not intensity-peak artifacts.
- `voiced_fraction` mean: 0.81 (v1) → 0.74 (v2). Slight drop because v2 syllables include onset/coda consonants (some unvoiced, e.g. /s/, /t/), whereas v1's intensity-peak windows centered on vowels. This is *more accurate*, not worse — the voicing flag handles the masking correctly.

### 12.4 Output format additions

Each syllable record in the v2 JSONL includes two new fields useful for spot-checking:

```json
{
  "phones": ["G", "AA1", "D"],
  "nucleus_phone": "AA1",
  "word": "god",
  "vec": [...18 dims...],
  ...
}
```

Plus a top-level `textgrid_path` per utterance.

### 12.5 What MFA can't do (yet)

About 3% of MELD dev utterances fail alignment — mostly very short single-word clips ("Ow!"), audio/text mismatches, or non-ASCII characters in the source text (Windows-1252 right-quotes from MELD's CSV). Cleaning the transcripts more aggressively would recover most. For now we treat unaligned utterances as drop-list inputs to downstream tasks.

The `alignment_analysis.csv` MFA emits in the output dir is per-utterance quality info: `overall_log_likelihood`, `phone_duration_deviation`, `snr`. These can be used to filter low-confidence alignments out of training data — see `scripts/filter_low_confidence_alignments.py`.

### 12.6 v2 invocation

```bash
.venv/bin/python scripts/extract_parametric_prosody_mfa.py \
    --in-dir data/meld/MELD.Raw/dev_splits_complete \
    --textgrids-dir data/meld/dev_textgrids_mfa \
    --metadata data/meld/MELD.Raw/dev_sent_emo.csv \
    --out data/meld/parametric_prosody_dev_mfa.jsonl
```

Runtime on MELD dev (with wav cache populated, MFA already run): ~3 min for both passes.

### 12.7 v2 result on the supervised comparison

`scripts/compare_prosody_text.py` runs three probes (text-only, prosody-only, combined) on the parametric vectors. Headline (5-fold CV on dev):

| Task | Probe | macro-F1 |
|---|---|---|
| Emotion (7-way) | majority | 0.086 |
| | text (TF-IDF + LR) | 0.198 |
| | **prosody (D19 pooled + LR)** | **0.182** |
| | **combined** | **0.236** |
| Sentiment (3-way) | majority | 0.200 |
| | text | 0.436 |
| | prosody | 0.385 |
| | combined | 0.437 |

Two takeaways:
1. **Prosody alone is comparable to text** (0.182 vs 0.198 emotion; 0.385 vs 0.436 sentiment). The 18-dim parametric vector encodes ~80–90% of the discriminative signal that TF-IDF text features capture.
2. **Prosody adds to text on emotion (+0.038), but not on sentiment (~0).** This matches the intuition: emotion has prosodic differentiators that text alone misses ("I'm fine" said sadly vs neutrally), while sentiment is dominated by lexical content ("I love it" / "I hate it").

This is the small-scale precursor to the Phase 5 killer experiment.
