"""
AMI text-only baseline: predict whether a dialogue act is `el.inf` (Elicit-Inform —
the canonical question DA) from its text alone.

This is the decisive cheap test for the corpus-vs-method question raised by
Phase 1.5 / EXP-006. MELD's text-only emotion ceiling was substantially driven
by transcriber-added punctuation (PUNCT alone macro-F1 = 0.197 emotion / 0.451
sentiment). AMI strips orthographic punctuation from word tokens (zero `?`
marks across 1.15M words; commas/periods are present but not `?`/`!`),
identifying questions purely through DA labelling instead.

If MELD's text dominance was punct-driven, AMI text-only AUC on `el.inf` vs
non-elicit should drop substantially compared to a punct-equipped corpus.
That tells us the +0.017 emotion ceiling on MELD was a corpus floor, and
prosody's marginal contribution should grow on AMI / NXT-style corpora.

Pipeline:
    1. Walk `data/ami/ami_annotations/dialogueActs/` for per-meeting/speaker DA XMLs.
    2. For each <dact>: extract the DA-type id (via the role="da-aspect" pointer)
       and the span of words it covers (via <nite:child href="...words.xml#id(X)..id(Y)">).
    3. Resolve the DA-type id against `ontologies/da-types.xml` — get name (`el.inf` etc).
    4. Resolve the word span by reading the per-speaker words XML and grabbing the
       text between the start/end ids (inclusive).
    5. Build a binary task: positives = DA name == "el.inf"; negatives = "inf" / "ass" / "sug" /
       "off" / "und" / "be.pos" / "be.neg" (statements, not other questions).
       Drop unlab/fra/stl/bck and other el.* — keep the cleanest contrast.
    6. Train TF-IDF + LR (same probe as EXP-005 text), compute AUC + macro-F1.
    7. Sanity: also report PUNCT-only ablation (token_pattern allows commas, periods,
       and ellipsis), so we can see if non-question-mark punctuation still leaks.

Reports:
    - Counts: total DAs, positives, negatives, dropped.
    - TF-IDF + LR train→test AUC and macro-F1 (5-fold CV; AMI has no canonical
      train/test split for DAs).
    - For comparison, also runs a "punct-only" ablation where text is replaced
      by just the punctuation tokens.

Usage:
    .venv/bin/python scripts/predict_da_from_text_ami.py \\
        --ami-root data/ami/ami_annotations \\
        --out data/ami/text_only_elinf_baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

NITE_NS = "http://nite.sourceforge.net/"
NS = {"nite": NITE_NS}

# Positives = el.inf (Elicit-Inform, the canonical question DA).
# Negatives = clear-statement DAs only, to avoid "yes-question vs other-question"
# confusion. Dropped categories listed below.
POS_DA = {"el.inf"}
NEG_DA = {"inf", "ass", "sug", "off", "und", "be.pos", "be.neg"}
# Dropped: bck, stl, fra (minor), oth (catch-all), unlab; el.sug/el.ass/el.und
# (other elicits) — we want clean question-vs-statement contrast.


def _ref_id(href: str) -> str:
    """Extract the id from `da-types.xml#id(ami_da_4)` or
    `Speaker.words.xml#id(Foo.words123)..id(Foo.words130)`."""
    m = re.search(r"#id\(([^)]+)\)", href)
    return m.group(1) if m else ""


def _span_ids(href: str) -> tuple[str, str]:
    """Extract (start_id, end_id) from `Foo.words.xml#id(X)..id(Y)`. If the
    span is a single id, returns (id, id)."""
    m = re.match(r"([^#]+)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", href)
    if not m:
        return ("", "")
    file_, start, end = m.group(1), m.group(2), m.group(3)
    return (start, end if end else start)


def load_da_ontology(da_types_path: Path) -> dict[str, str]:
    """ami_da_4 → 'inf', ami_da_5 → 'el.inf', etc."""
    out: dict[str, str] = {}
    tree = ET.parse(da_types_path)
    for node in tree.iter():
        if "name" in node.attrib and node.attrib.get(f"{{{NITE_NS}}}id", "").startswith("ami_da_"):
            nid = node.attrib[f"{{{NITE_NS}}}id"]
            out[nid] = node.attrib["name"]
    return out


def load_words(words_path: Path) -> list[tuple[str, str, bool]]:
    """Returns ordered list of (word_id, surface_text, is_punct) for one
    speaker file."""
    tree = ET.parse(words_path)
    out = []
    for w in tree.getroot().findall("{%s}w" % NITE_NS) + tree.getroot().findall("w"):
        wid = w.attrib.get(f"{{{NITE_NS}}}id", "")
        text = (w.text or "").strip()
        is_punct = w.attrib.get("punc", "false") == "true"
        out.append((wid, text, is_punct))
    return out


def collect_da_text(ami_root: Path) -> list[dict]:
    """Walk dialogueActs/, resolve each DA to (da_name, text, n_words, n_punct,
    speaker, meeting). Returns list of dicts."""
    da_types_path = ami_root / "ontologies" / "da-types.xml"
    da_id_to_name = load_da_ontology(da_types_path)

    da_files = sorted((ami_root / "dialogueActs").glob("*.dialog-act.xml"))
    print(f"  found {len(da_files)} DA files")

    out: list[dict] = []
    seen_word_files: dict[Path, list[tuple[str, str, bool]]] = {}

    for da_file in da_files:
        # Filename pattern: MEETING.SPEAKER.dialog-act.xml
        stem = da_file.name.removesuffix(".dialog-act.xml")
        parts = stem.split(".")
        if len(parts) != 2:
            continue  # e.g. MEETING.adjacency-pairs.xml
        meeting, speaker = parts
        words_file = ami_root / "words" / f"{meeting}.{speaker}.words.xml"
        if not words_file.exists():
            continue
        if words_file not in seen_word_files:
            seen_word_files[words_file] = load_words(words_file)
        words = seen_word_files[words_file]
        wid_to_idx = {wid: i for i, (wid, _, _) in enumerate(words)}

        try:
            tree = ET.parse(da_file)
        except ET.ParseError:
            continue
        for dact in tree.getroot().findall("{%s}dact" % NITE_NS) + tree.getroot().findall("dact"):
            # da-type id via pointer
            da_id = ""
            for ptr in dact.findall("{%s}pointer" % NITE_NS):
                if ptr.attrib.get("role") == "da-aspect":
                    da_id = _ref_id(ptr.attrib.get("href", ""))
                    break
            da_name = da_id_to_name.get(da_id, "?")
            # word span via child
            child = dact.find("{%s}child" % NITE_NS)
            if child is None:
                continue
            href = child.attrib.get("href", "")
            start_id, end_id = _span_ids(href)
            if start_id not in wid_to_idx:
                continue
            si = wid_to_idx[start_id]
            ei = wid_to_idx.get(end_id, si)
            span = words[si: ei + 1]
            words_only = [t for (_, t, p) in span if not p]
            puncts_only = [t for (_, t, p) in span if p]
            text = " ".join(words_only)
            out.append({
                "meeting": meeting,
                "speaker": speaker,
                "da_name": da_name,
                "text": text,
                "punct_string": " ".join(puncts_only),
                "n_words": len(words_only),
                "n_punct": len(puncts_only),
            })
    return out


def build_binary_task(records: list[dict]) -> tuple[list[str], list[str], list[str], np.ndarray, dict]:
    """Returns (texts_full, texts_words_only, punct_strings, labels, stats).

    texts_full       : words + their adjacent punctuation tokens, joined.
    texts_words_only : just the surface words (no punct markers in the string).
                       Equivalent to running TF-IDF after stripping `?`/`.`/`,`.
    punct_strings    : punct tokens only (control).
    label = 1 for el.inf, 0 for clear-statement DAs.
    """
    texts_full, texts_words, puncts, labels = [], [], [], []
    counts = {"pos": 0, "neg": 0, "dropped": 0}
    da_counts: dict[str, int] = {}
    for r in records:
        da_counts[r["da_name"]] = da_counts.get(r["da_name"], 0) + 1
        if r["da_name"] in POS_DA:
            label = 1
            counts["pos"] += 1
        elif r["da_name"] in NEG_DA:
            label = 0
            counts["neg"] += 1
        else:
            counts["dropped"] += 1
            continue
        if r["n_words"] == 0:
            counts["dropped"] += 1
            counts[("pos" if label else "neg")] -= 1
            continue
        # full text: append punctuation onto the words string
        full = r["text"]
        if r["punct_string"]:
            full = full + " " + r["punct_string"]
        texts_full.append(full)
        texts_words.append(r["text"])  # words only, no punct
        puncts.append(r["punct_string"] if r["punct_string"] else "EMPTY")
        labels.append(label)
    return texts_full, texts_words, puncts, np.array(labels, dtype=np.int64), {"counts": counts, "da_counts": da_counts}


def make_text_probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b|[!?]+|\.{2,}",
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_punct_only_probe() -> Pipeline:
    """TF-IDF over punct strings only. If punct still leaks, we'd see it here."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, lowercase=False,
                                   token_pattern=r"[^\s]+")),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def cv_metrics(probe, X, y, n_splits: int = 5, seed: int = 42) -> dict:
    aucs, f1s = [], []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, te_idx in cv.split(X, y):
        Xtr = [X[i] for i in tr_idx]
        Xte = [X[i] for i in te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        probe.fit(Xtr, ytr)
        proba = probe.predict_proba(Xte)[:, 1]
        pred = probe.predict(Xte)
        aucs.append(roc_auc_score(yte, proba))
        f1s.append(f1_score(yte, pred, average="macro", zero_division=0))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "auc_per_fold": [float(a) for a in aucs]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-root", type=Path,
                        default=Path("data/ami/ami_annotations"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] AMI annotations from {args.ami_root}")
    records = collect_da_text(args.ami_root)
    print(f"  resolved {len(records)} dialogue acts")

    texts_full, texts_words, puncts, y, stats = build_binary_task(records)
    print(f"  task: positives (el.inf) = {stats['counts']['pos']}, "
          f"negatives (statement DAs) = {stats['counts']['neg']}, "
          f"dropped = {stats['counts']['dropped']}")
    print(f"  positive rate: {100*y.mean():.2f}%")
    print()
    print("[da-name distribution across all DAs in AMI]")
    for name, cnt in sorted(stats["da_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<10s}  {cnt:>6d}")

    print()
    print("=" * 72)
    print("Text-with-punct probe — words AND punct tokens (TF-IDF + LR, 5-fold CV)")
    print("=" * 72)
    text_full_metrics = cv_metrics(make_text_probe(), texts_full, y)
    print(f"  AUC mean ± std: {text_full_metrics['auc_mean']:.4f} ± {text_full_metrics['auc_std']:.4f}")
    print(f"  per-fold AUC:   {[round(a, 3) for a in text_full_metrics['auc_per_fold']]}")
    print(f"  macro-F1 mean:  {text_full_metrics['macro_f1_mean']:.4f}")

    print()
    print("=" * 72)
    print("Words-only probe — punct stripped (the corpus-floor for prosody)")
    print("=" * 72)
    text_words_metrics = cv_metrics(make_text_probe(), texts_words, y)
    print(f"  AUC mean ± std: {text_words_metrics['auc_mean']:.4f} ± {text_words_metrics['auc_std']:.4f}")
    print(f"  per-fold AUC:   {[round(a, 3) for a in text_words_metrics['auc_per_fold']]}")
    print(f"  macro-F1 mean:  {text_words_metrics['macro_f1_mean']:.4f}")

    print()
    print("=" * 72)
    print("Punct-only probe — control: how much of the signal is in `?` etc.")
    print("=" * 72)
    punct_metrics = cv_metrics(make_punct_only_probe(), puncts, y)
    print(f"  AUC mean ± std: {punct_metrics['auc_mean']:.4f} ± {punct_metrics['auc_std']:.4f}")
    print(f"  per-fold AUC:   {[round(a, 3) for a in punct_metrics['auc_per_fold']]}")
    print(f"  macro-F1 mean:  {punct_metrics['macro_f1_mean']:.4f}")

    print()
    print("=" * 72)
    print("Interpretation")
    print("=" * 72)
    print(f"  text+punct AUC: {text_full_metrics['auc_mean']:.4f}")
    print(f"  words-only AUC: {text_words_metrics['auc_mean']:.4f}   "
          f"(Δ = {text_full_metrics['auc_mean'] - text_words_metrics['auc_mean']:+.4f})")
    print(f"  punct-only AUC: {punct_metrics['auc_mean']:.4f}")
    headroom = 1 - text_words_metrics['auc_mean']
    print(f"  → corpus-floor for prosody (above words-only): {headroom:.4f} AUC of headroom")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({
                "task": "el.inf vs clear-statement DAs",
                "stats": stats,
                "n_total_records": len(records),
                "n_used": int(len(y)),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
                "text_with_punct": text_full_metrics,
                "words_only_no_punct": text_words_metrics,
                "punct_only": punct_metrics,
            }, f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
