# AMI Meeting Corpus — scoping for PILM Phase 2 / Phase 1.5+

_Written 2026-04-27. Companion to `docs/session_handoff.md` and `docs/phases.md`._

## Why we're looking at AMI

Phase 2 (Switchboard NXT) is pinned on UW LDC institutional access. While that's
pending, the central open question from Phase 1.5 is whether the +0.017 emotion
ceiling on MELD is corpus-driven (acted speech, transcriber-added punctuation
acts as text's prosody-proxy) or method-driven (supervised classification on top
of pre-extracted features can't extract more given text). NXT will eventually
answer this; AMI lets us partially answer it now.

## What AMI is

- **AMI Meeting Corpus** — ~100 hours of multi-party meeting speech.
- **License: CC-BY 4.0**, free to download.
- **Native NXT XML format** — same standoff-annotation system as Switchboard
  NXT. Our parser scaffold for one will work for both.
- **139 meetings**, **1,147,783 words**, **117,915 dialogue act annotations**.
- 4-speaker meetings predominantly; occasionally 3 or 5.
- Two scenarios: **Scenario meetings** (role-play product-design tasks; majority
  of corpus) and **Non-Scenario meetings** (genuine free-form discussion).

## Annotation layers available

```
abstractive/        decision/         disfluency/        focus/
argumentation/      dialogueActs/     extractive/        handGesture/
configuration/      ontologies/       headGesture/       movement/
namedEntities/      participantRoles/                    segments/
participantSummaries/                  topics/           words/
youUsages/
```

Layers most relevant to PILM:

- **`words/`** — every word has `starttime`/`endtime`. Punctuation is its own
  word with `punc="true"`. Times are channel-aligned, so we can extract Praat
  features per-word, per-syllable. Punct distribution: **101,149 periods,
  58,522 commas, 9,890 question marks, ~1 each of `!` and `:`**. So question
  marks ARE present in AMI text, but they are sparse (0.86% of all tokens) —
  contrast MELD where `?` and `!` are aggressively used by the transcribers
  and act as effectively perfect oracles for question detection.
- **`segments/`** — transcriber-marked utterance/turn boundaries. Equivalent to
  MELD's per-utterance rows.
- **`dialogueActs/`** — DAs link to a span of words. Each DA points to one of
  16 categories in `ontologies/da-types.xml`.
- **`focus/`** — actually visual focus-of-attention (gaze direction). **NOT**
  the linguistic focus/contrast (kontrast) we'd want for focus prosody. AMI
  does not appear to have a kontrast-equivalent annotation.

## Dialogue act ontology (the gold)

```
minor:   bck (Backchannel),  stl (Stall),  fra (Fragment)
task:    inf (Inform),       sug (Suggest), ass (Assess)
elicit:  el.inf (Elicit-Inform — i.e., a question seeking info)
         el.sug (Elicit-Offer-Or-Suggestion)
         el.ass (Elicit-Assessment)
         el.und (Elicit-Comment-Understanding)
other:   off (Offer), und (Comment-About-Understanding),
         be.pos (Be-Positive), be.neg (Be-Negative), oth (Other)
unlab:   Unlab
```

The four `el.*` categories are explicit *question* tags. AMI question labels
are derived from DA tagging by trained annotators, independent of orthographic
punctuation. AMI does have some `?` marks (9,890 across 1.15M words ≈ 0.9%)
but unlike MELD they are sparse and not dispositive for the DA labels.

**EXP-007-analogue text-only baseline on AMI** (`scripts/predict_da_from_text_ami.py`,
results in `data/ami/text_only_elinf_baseline.json`, 5-fold CV over 70,918
DAs, positives = 4,050 el.inf, negatives = 66,868 statement-class DAs):

| Probe | AUC | macro-F1 |
|---|---:|---:|
| text-with-punct | 0.9342 | 0.7988 |
| **words-only (punct stripped)** | **0.8854** | 0.6842 |
| punct-only (control) | 0.8853 | 0.8218 |

Comparison anchor — same text-only baseline run on MELD yn-Q detection
(`scripts/predict_question_from_text.py`):

| Probe | MELD yn-Q AUC | MELD all-Q AUC |
|---|---:|---:|
| text-with-punct | 1.0000 | 0.9999 |
| words-only | 0.8596 | 0.8897 |
| punct-only | 1.0000 | 1.0000 |

**Conclusions from this comparison:**

1. **MELD's question task is solved by punct alone (AUC = 1.0).** The `?` is
   effectively a perfect oracle — the entire 18% positive class is marked.
   So EXP-007's prosody AUC of 0.65 was never measured against a text-uses-
   punct baseline; it was measured against zero-text. Prosody is doing
   real work *given that text was hidden*, but a real-world classifier with
   text + punct + prosody would gain nothing over text + punct alone for
   questions.
2. **Words-only AUC is comparable across corpora**: MELD 0.86, AMI 0.89.
   English question lexicon ("did", "do", "what", "how", "is") is
   similarly predictive on both. The corpus difference is in *whether
   punctuation acts as a cheat-code*, not in the underlying lexical
   signal.
3. **AMI's text-with-punct is NOT a perfect oracle for el.inf** (0.93 vs
   1.0 on MELD). That's where the corpus-difference matters: there's
   genuine text-side ambiguity for AMI, so prosody could plausibly add
   incremental value where it could not on MELD.
4. **The PILM thesis target is wrong if framed around question detection.**
   English questions are overwhelmingly lexically anchored regardless of
   corpus; the +0.017 prosody-emotion uplift on MELD will not be
   amplified for questions on AMI. The emotion / affect / argumentation
   tasks are where prosody's marginal value should land.

## Audio availability

- Per-meeting individual headset (IH) channels, ~40 MB per channel as 16 kHz
  WAV. A meeting averages 4 channels → ~160 MB / meeting.
- `Mix-Headset.wav` is the down-mix (~40 MB / meeting), useful when channel
  separation isn't critical.
- Full corpus audio ≈ 50 GB — too big for current 29 GB free disk.
- **Scoped plan: download 3 meetings of IH audio (~500 MB) for pipeline
  validation; defer full-corpus download until disk space is available or LDC
  clears.**

## What AMI can do for PILM today

### Cheap, no-audio test (do today)

Run text-only `el.inf` (question vs not) prediction on AMI annotations alone.
If MELD's text dominance was punct-driven, AMI's text-only AUC on the question
task should be substantially lower than MELD's (~0.85 with punct, dropped to
~0.78 without per EXP-006 minus PUNCT). Compare:

- MELD text-only `?`-prediction AUC: not separately computed yet, but text-only
  emotion macro-F1 with PUNCT removed dropped 0.318 → 0.264 (-17%).
- AMI text-only `el.inf` AUC: TBD by `scripts/predict_da_from_text_ami.py`.

This is a free signal we can get without any prosody pipeline.

### Pipeline validation (this week)

1. Download 3 representative meetings + their IH audio (~500 MB).
2. Adapt `scripts/extract_parametric_prosody_mfa.py` to operate on AMI's
   per-channel audio + word XML (instead of MELD wav + TextGrid). Same 18-dim
   D19 vector spec; same per-speaker baselining (per-channel).
3. Re-run EXP-007 analogue: predict `el.inf` from the 18-dim parametric vector
   on those meetings. Compare AUC to MELD's 0.65–0.69 yn-only.
4. Re-run EXP-005 analogue: text vs prosody vs combined on `el.inf` prediction.
   Decisive on the corpus-vs-method question.

### Phase 2 fallback / hedge

If LDC access doesn't clear within the next month, AMI is a credible
substitute for the killer-experiment corpus. Trade-offs vs Switchboard NXT:

| Property | Switchboard NXT | AMI |
|---|---|---|
| ToBI gold labels | yes (~63 conversations) | no |
| Dialogue acts | yes (SWBD-DAMSL) | yes (AMI DA, 16 categories) |
| Linguistic focus / kontrast | yes | no (only visual gaze) |
| Conversation type | 2-party phone | 4-party meetings |
| Hours | ~50 | ~100 |
| Free? | no (LDC) | **yes** |
| Native NXT format | yes | yes |

Without ToBI, AMI cannot serve as a probe-target generator for our parametric
vector validation (D5). But the question / dialogue-act prediction story
generalises directly.

## Concrete action items

- [x] Download AMI annotations (22.9 MB). Extracted to `data/ami/ami_annotations/`.
- [ ] Write `scripts/nxt_xml_reader.py` — generic NXT-format parser. Validate
      on AMI; same parser will read Switchboard NXT when LDC lands.
- [ ] Write `scripts/predict_da_from_text_ami.py` — text-only el.inf
      prediction baseline. Decisive on the punct-cheating hypothesis.
- [ ] (Stretch) Download 3 meetings' audio + run parametric extraction.
- [ ] (Stretch) Cross-corpus EXP-007: predict `el.inf` from prosody on AMI;
      compare AUC to MELD yn-only.

## Naming convention for AMI data

```
data/ami/
  ami_public_manual_1.6.2.zip      # downloaded annotations zip (preserved)
  ami_annotations/                 # extracted
    words/                         # per-meeting per-speaker word XMLs
    dialogueActs/                  # DAs referencing word spans
    segments/                      # transcriber utterance boundaries
    ontologies/da-types.xml        # the 16-DA ontology
    ...
  audio/                           # to be populated, 3 meetings to start
    ES2002a/
      Headset-0.wav, Headset-1.wav, ...
  parametric_prosody_ami.jsonl     # when pipeline lands
```
