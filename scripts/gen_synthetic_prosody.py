"""
Synthetic prosody data generator for PILM Phase 1.

PHASE 1 ARCHIVE — kept for reproducibility and as a smoke test.
The synthetic harness validated the architecture and diagnosed the
modality-collapse failure mode in EXP-001 (see docs/experiments.md and
docs/writeups/exp001_modality_collapse.md). It is superseded for the
core scientific work by the Phase 2+ Switchboard NXT pipeline (D15)
but kept in-repo because it remains a useful unit test of the encoder.

Produces utterances with paired:
- Phone sequence (toy ARPAbet alphabet, with vowel/consonant flags)
- Syllable and word indices per phone
- AM/ToBI accent + boundary categories per phone
- Continuous prosody features per phone: log_f0_z, energy_z, dur_rel
- Utterance-level pragmatic label

Pragmatic labels and how prosody encodes them:
- STATEMENT          : declination + terminal L%
- QUESTION           : declination + terminal H% rise on final stressed syllable
- FOCUS_K            : L+H* on word K (contrastive focus); otherwise default declination
- SURPRISED_QUESTION : QUESTION with extra-high F0 peak at terminal (~+1.5 z vs ~+0.7)

A mild lexical-pragmatic correlation is baked in: a small set of "question-typical"
filler words appears more often in QUESTION / SURPRISED_QUESTION utterances. This
gives a text-only baseline a plausible signal to learn against, so the killer
experiment in Phase 1 has a meaningful margin to measure.

Output: JSONL, one utterance per line. Plus a sidecar vocab.json with the
discrete vocabularies (phones, accents, boundaries, labels).

Usage:
    python scripts/gen_synthetic_prosody.py --n 10000 --out data/synthetic/train.jsonl
    python scripts/gen_synthetic_prosody.py --preview 3
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Toy linguistic resources
# ---------------------------------------------------------------------------

VOWELS = ["AA", "AE", "IY", "OW", "UW", "EH", "AH"]
CONSONANTS = ["P", "T", "K", "B", "D", "G", "M", "N", "S", "Z", "L", "R", "W", "Y", "F"]
PHONES = VOWELS + CONSONANTS
PHONE_TO_ID = {p: i for i, p in enumerate(PHONES)}
IS_VOWEL = {p: (p in VOWELS) for p in PHONES}

# AM/ToBI categories, v1 reduced.
ACCENTS = ["NONE", "H*", "L*", "L+H*"]
ACCENT_TO_ID = {a: i for i, a in enumerate(ACCENTS)}

BOUNDARIES = ["NONE", "B1", "B4_L", "B4_H", "B4_HH"]  # final tone is encoded in B4_*
BOUNDARY_TO_ID = {b: i for i, b in enumerate(BOUNDARIES)}

LABELS = ["STATEMENT", "QUESTION", "SURPRISED_QUESTION", "FOCUS"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

# A small content-word vocabulary for the toy lexicon. Words are tuples of
# syllables; syllables are (onset_consonants, nucleus_vowel, coda_consonants).
# We synthesize words on the fly rather than enumerate them — see make_word().

# Question-typical filler words placed at sentence start when the label is
# QUESTION / SURPRISED_QUESTION. The text-only signal: their phonetic shape
# is fixed and the model can learn it. All phones drawn from the toy alphabet.
QUESTION_WORDS_PHONES = [
    ["D", "IY", "D"],   # "did"-shaped
    ["K", "AH", "N"],   # "can"-shaped
    ["W", "AH", "T"],   # "what"-shaped
    ["B", "OW", "R"],   # invented; gives a four-shape pool to spread mass over
]

# Statement-typical filler words appended at the end of STATEMENT utterances
# under the "strong" lexical-signal regime (EXP-004+).
STATEMENT_END_WORDS_PHONES = [
    ["Y", "EH"],          # "yeah"-shaped
    ["R", "IY", "T"],     # "right"-shaped
    ["S", "OW"],          # "so"-shaped
    ["F", "AE", "N"],     # invented; fourth shape to spread probability mass
]

# Focus-marker words inserted just before the focused word under "strong"
# lexical-signal regime. In real English these would be words like "not",
# "but", "rather", "instead".
FOCUS_MARKER_WORDS_PHONES = [
    ["N", "AA", "T"],     # "not"
    ["B", "AH", "T"],     # "but"
    ["L", "EH", "S"],     # "less"
    ["AE", "S"],          # "as"
]

# Sanity check at import time so any future phone-set drift fails loudly.
for _wlist in (QUESTION_WORDS_PHONES, STATEMENT_END_WORDS_PHONES, FOCUS_MARKER_WORDS_PHONES):
    for _w in _wlist:
        for _p in _w:
            assert _p in PHONE_TO_ID, f"unknown phone in synthetic lexicon: {_p}"


# ---------------------------------------------------------------------------
# Word / sentence sampling
# ---------------------------------------------------------------------------

def make_syllable(rng: random.Random) -> list[str]:
    """Sample a (C)V(C) syllable as a list of phones."""
    syll: list[str] = []
    if rng.random() < 0.7:
        syll.append(rng.choice(CONSONANTS))
    syll.append(rng.choice(VOWELS))
    if rng.random() < 0.4:
        syll.append(rng.choice(CONSONANTS))
    return syll


def make_word(rng: random.Random, n_syllables: Optional[int] = None) -> list[list[str]]:
    """Sample a word as a list of syllables (each a list of phones)."""
    if n_syllables is None:
        n_syllables = rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]
    return [make_syllable(rng) for _ in range(n_syllables)]


def question_word(rng: random.Random) -> list[list[str]]:
    """Return a question-typical filler word as a one-syllable word."""
    phones = rng.choice(QUESTION_WORDS_PHONES)
    return [phones]


def statement_end_word(rng: random.Random) -> list[list[str]]:
    """Return a statement-typical filler word as a one-syllable word."""
    return [rng.choice(STATEMENT_END_WORDS_PHONES)]


def focus_marker_word(rng: random.Random) -> list[list[str]]:
    """Return a focus-marker word as a one-syllable word."""
    return [rng.choice(FOCUS_MARKER_WORDS_PHONES)]


# ---------------------------------------------------------------------------
# Prosody assignment given the pragmatic label
# ---------------------------------------------------------------------------

@dataclass
class Phone:
    phone: str
    is_vowel: bool
    syllable_idx: int
    word_idx: int
    is_stressed: bool       # primary stress on the syllable
    accent: str             # AM/ToBI accent category for this position
    boundary: str           # boundary category that *follows* this position
    log_f0_z: Optional[float]   # None on voiceless / consonants without F0
    energy_z: float
    dur_rel: float


def _add_noise(rng: random.Random, x: float, sd: float = 0.1) -> float:
    return x + rng.gauss(0.0, sd)


def assign_prosody(
    rng: random.Random,
    words: list[list[list[str]]],
    label: str,
    focus_word: Optional[int],
) -> list[Phone]:
    """Materialize phones with accent / boundary / continuous features.

    The key generative rules (these are the patterns Phase 1 must learn):
        - Lexical stress lands on the first syllable of every content word.
        - L+H* (contrastive focus) lands on the stressed syllable of the
          focused word, if FOCUS label is active.
        - QUESTION raises F0 on the final stressed syllable; the boundary
          token after the last phone becomes B4_H instead of B4_L.
        - SURPRISED_QUESTION pushes the terminal F0 even higher (B4_HH)
          and the final-syllable peak to ~+1.5 z.
        - STATEMENT and FOCUS use B4_L terminal.
    """
    # Flatten with bookkeeping.
    n_syllables_total = sum(len(w) for w in words)
    flat: list[Phone] = []
    syllable_counter = 0

    # Pre-compute which syllable carries lexical stress per word (always 1st).
    stressed_syllables = set()
    for wi, w in enumerate(words):
        # Word-relative syllable index 0 is the lexically stressed syllable.
        # Convert to absolute syllable index in the utterance.
        absolute = sum(len(words[k]) for k in range(wi))
        stressed_syllables.add(absolute)

    # Identify the absolute syllable index of the "terminal stressed syllable"
    # (the rightmost stressed syllable in the utterance).
    terminal_stressed = max(stressed_syllables) if stressed_syllables else 0

    # Identify the focused word's stressed syllable (for FOCUS).
    focused_stressed = None
    if label == "FOCUS" and focus_word is not None:
        focused_stressed = sum(len(words[k]) for k in range(focus_word))

    # Walk through the utterance.
    for wi, word in enumerate(words):
        for si_in_word, syll in enumerate(word):
            absolute_syl = syllable_counter
            syllable_counter += 1

            is_stressed = absolute_syl in stressed_syllables
            is_terminal_stressed = absolute_syl == terminal_stressed
            is_focused = absolute_syl == focused_stressed

            # Determine the syllable's accent.
            if is_focused:
                syll_accent = "L+H*"
            elif is_terminal_stressed and label in ("QUESTION", "SURPRISED_QUESTION"):
                syll_accent = "H*"
            elif is_stressed:
                syll_accent = "H*"
            else:
                syll_accent = "NONE"

            # Determine the syllable's nucleus log_f0_z.
            # Default declination: linear drop from +0.5 to -0.5 across the utterance.
            decline_pos = absolute_syl / max(n_syllables_total - 1, 1)
            base_f0 = 0.5 - decline_pos * 1.0  # +0.5 → -0.5

            if is_focused:
                f0_peak = 1.0
            elif is_terminal_stressed and label == "SURPRISED_QUESTION":
                f0_peak = 1.5
            elif is_terminal_stressed and label == "QUESTION":
                f0_peak = 0.7
            elif is_stressed:
                f0_peak = base_f0 + 0.3
            else:
                f0_peak = base_f0

            energy_base = 0.5 if is_stressed else 0.0
            dur_factor = 1.0
            if is_stressed:
                dur_factor *= 1.2
            if is_focused:
                dur_factor *= 1.2

            # Pre-boundary lengthening for the last syllable of the word.
            is_word_final_syllable = (si_in_word == len(word) - 1)
            if is_word_final_syllable:
                dur_factor *= 1.1

            # Materialize phones in this syllable.
            for pi, phone in enumerate(syll):
                v = IS_VOWEL[phone]
                is_phone_word_final = (
                    is_word_final_syllable and pi == len(syll) - 1
                )
                is_phone_utt_final = (
                    is_phone_word_final
                    and wi == len(words) - 1
                )

                # Boundary: only assigned to the last phone of a word.
                if is_phone_utt_final:
                    if label == "SURPRISED_QUESTION":
                        bdry = "B4_HH"
                    elif label == "QUESTION":
                        bdry = "B4_H"
                    else:
                        bdry = "B4_L"
                elif is_phone_word_final:
                    bdry = "B1"
                else:
                    bdry = "NONE"

                # Accent on the syllable nucleus (vowel). Toy syllables are CV(C),
                # so each syllable has a single vowel that carries the accent.
                accent = syll_accent if v else "NONE"

                # log_f0_z: only on vowels (voiceless segments masked).
                if v:
                    log_f0 = _add_noise(rng, f0_peak, sd=0.08)
                else:
                    log_f0 = None

                # Energy and duration noisy.
                energy = _add_noise(rng, energy_base, sd=0.08)
                dur = _add_noise(rng, dur_factor, sd=0.05)

                flat.append(Phone(
                    phone=phone,
                    is_vowel=v,
                    syllable_idx=absolute_syl,
                    word_idx=wi,
                    is_stressed=is_stressed,
                    accent=accent,
                    boundary=bdry,
                    log_f0_z=None if log_f0 is None else round(log_f0, 4),
                    energy_z=round(energy, 4),
                    dur_rel=round(dur, 4),
                ))
    return flat


# ---------------------------------------------------------------------------
# Utterance generation
# ---------------------------------------------------------------------------

def sample_label(rng: random.Random) -> tuple[str, Optional[int]]:
    """Sample a pragmatic label and (for FOCUS) a focused-word index placeholder."""
    label = rng.choices(LABELS, weights=[3, 3, 2, 2])[0]
    return label, None  # focus index assigned after we know the word count


@dataclass
class LexicalSignalConfig:
    """Probabilities controlling how often each lexical-pragmatic correlate fires.

    Two named regimes used in the experiments:

        weak (EXP-001 / EXP-002):
            question_filler_prob = 0.5
            statement_end_prob   = 0.0
            focus_marker_prob    = 0.0

        strong (EXP-004+):
            question_filler_prob = 0.8
            statement_end_prob   = 0.5
            focus_marker_prob    = 0.6

    The "strong" regime gives a text-only model meaningful signal for STATEMENT,
    QUESTION, and FOCUS. SURPRISED_QUESTION still requires prosody to disambiguate
    from QUESTION (intentionally — that's the prosody-only contrast).
    """
    question_filler_prob: float = 0.8
    statement_end_prob: float = 0.5
    focus_marker_prob: float = 0.6


WEAK_LEXICAL_SIGNAL = LexicalSignalConfig(
    question_filler_prob=0.5,
    statement_end_prob=0.0,
    focus_marker_prob=0.0,
)
STRONG_LEXICAL_SIGNAL = LexicalSignalConfig()  # uses field defaults


def sample_utterance(
    rng: random.Random,
    n_words_range: tuple[int, int] = (3, 8),
    lex: Optional[LexicalSignalConfig] = None,
) -> dict:
    """Generate one utterance dictionary."""
    lex = lex or STRONG_LEXICAL_SIGNAL
    label, _ = sample_label(rng)

    n_words = rng.randint(*n_words_range)
    words: list[list[list[str]]] = []

    # Maybe prepend a question-typical filler for QUESTION / SURPRISED_QUESTION.
    if label in ("QUESTION", "SURPRISED_QUESTION") and rng.random() < lex.question_filler_prob:
        words.append(question_word(rng))
        n_words -= 1

    for _ in range(max(n_words, 1)):
        words.append(make_word(rng))

    # Maybe append a statement-typical filler at the end of STATEMENT utterances.
    if label == "STATEMENT" and rng.random() < lex.statement_end_prob:
        words.append(statement_end_word(rng))

    # Assign focus word for FOCUS label.
    focus_word: Optional[int] = None
    if label == "FOCUS":
        focus_word = rng.randrange(0, len(words))
        # Maybe insert a focus-marker word just before the focused word.
        if rng.random() < lex.focus_marker_prob:
            marker = focus_marker_word(rng)
            words.insert(focus_word, marker)
            focus_word += 1

    phones = assign_prosody(rng, words, label, focus_word)

    return {
        "label": label,
        "focus_word": focus_word,
        "n_words": len(words),
        "n_syllables": sum(len(w) for w in words),
        "phones": [asdict(p) for p in phones],
    }


# ---------------------------------------------------------------------------
# Vocabulary export
# ---------------------------------------------------------------------------

def vocab_dict() -> dict:
    return {
        "phones": PHONE_TO_ID,
        "accents": ACCENT_TO_ID,
        "boundaries": BOUNDARY_TO_ID,
        "labels": LABEL_TO_ID,
        "vowels": VOWELS,
        "consonants": CONSONANTS,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10000, help="Number of utterances to generate.")
    p.add_argument("--out", type=Path, help="Output JSONL path. Required unless --preview is set.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--preview", type=int, default=0, help="If >0, print this many readable previews and exit.")
    p.add_argument(
        "--lexical-signal", choices=["weak", "strong"], default="strong",
        help="weak (EXP-001/002): only question filler at start. "
             "strong (EXP-004+): adds statement-end and focus-marker words.",
    )
    return p.parse_args()


def preview(rng: random.Random, n: int, lex: LexicalSignalConfig) -> None:
    for _ in range(n):
        utt = sample_utterance(rng, lex=lex)
        phones = utt["phones"]
        print(f"=== label={utt['label']}  focus_word={utt['focus_word']}  n_words={utt['n_words']} ===")
        # Render as a phone-aligned table.
        header = f"{'idx':>3} {'phone':>5} {'V':>1} {'syl':>3} {'wd':>2} {'stress':>6} {'accent':>5} {'bdry':>5} {'logF0z':>7} {'energy':>6} {'dur':>5}"
        print(header)
        for i, p in enumerate(phones):
            f0 = "    -- " if p["log_f0_z"] is None else f"{p['log_f0_z']:>7.3f}"
            print(
                f"{i:>3} {p['phone']:>5} {'V' if p['is_vowel'] else 'C':>1} "
                f"{p['syllable_idx']:>3} {p['word_idx']:>2} "
                f"{('*' if p['is_stressed'] else ' '):>6} "
                f"{p['accent']:>5} {p['boundary']:>5} "
                f"{f0} {p['energy_z']:>6.3f} {p['dur_rel']:>5.3f}"
            )
        print()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    lex = STRONG_LEXICAL_SIGNAL if args.lexical_signal == "strong" else WEAK_LEXICAL_SIGNAL

    if args.preview > 0:
        preview(rng, args.preview, lex)
        return

    if args.out is None:
        raise SystemExit("--out is required unless --preview is set.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vocab_path = args.out.with_name("vocab.json")

    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            utt = sample_utterance(rng, lex=lex)
            f.write(json.dumps(utt) + "\n")

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_dict(), f, indent=2)

    print(f"Wrote {args.n} utterances to {args.out} (lexical-signal={args.lexical_signal})")
    print(f"Wrote vocab to {vocab_path}")


if __name__ == "__main__":
    main()
