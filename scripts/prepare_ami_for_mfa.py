"""
Prepare AMI per-segment WAVs + .lab transcripts for MFA alignment.

For each (meeting, speaker) we:

  1. Look up the channel index from `corpusResources/meetings.xml`
     (A=0, B=1, C=2, D=3 in the canonical config; verified per-meeting).
  2. Read `segments/<meeting>.<speaker>.segments.xml` for utterance boundaries.
  3. Read `words/<meeting>.<speaker>.words.xml` to resolve segment-to-word spans.
  4. Slice the per-channel WAV (Headset-{channel}.wav) at each segment's
     `transcriber_start` / `transcriber_end` time → write per-segment WAV.
  5. Write a matching .lab file containing the segment's words (lowercase,
     punctuation-stripped, MFA-ready).

Output layout:
    <out>/<meeting>/
        <meeting>_<speaker>_seg<segment_index>.wav
        <meeting>_<speaker>_seg<segment_index>.lab

Plus a manifest `<out>/<meeting>/manifest.jsonl`, one line per segment, with
utterance_id, speaker letter, global_name, role, channel, t_start_ms, t_end_ms,
n_words, text, das_contained.

Skips segments where:
  - text is empty after cleaning (all-punct or laughter-only)
  - duration < 100 ms
  - source WAV missing

Usage:
    .venv/bin/python scripts/prepare_ami_for_mfa.py \\
        --meeting ES2002a \\
        --audio-root data/ami/audio \\
        --ami-root   data/ami/ami_annotations \\
        --out        data/ami/mfa_input
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

NITE_NS = "http://nite.sourceforge.net/"


# ---------------------------------------------------------------------------
# Speaker → channel / global_name lookup
# ---------------------------------------------------------------------------

def load_meeting_speakers(meetings_xml: Path) -> dict[str, dict[str, dict]]:
    """observation_id → {speaker_letter: {'channel', 'global_name', 'role'}}."""
    tree = ET.parse(meetings_xml)
    out: dict[str, dict[str, dict]] = {}
    for m in tree.getroot().findall("meeting"):
        obs = m.attrib.get("observation", "")
        if not obs:
            continue
        out[obs] = {}
        for s in m.findall("speaker"):
            letter = s.attrib.get("nxt_agent", "")
            if not letter:
                continue
            out[obs][letter] = {
                "channel": int(s.attrib.get("channel", "-1")),
                "global_name": s.attrib.get("global_name", ""),
                "role": s.attrib.get("role", ""),
            }
    return out


# ---------------------------------------------------------------------------
# Words / segments / DA loaders
# ---------------------------------------------------------------------------

def load_words_for(speaker_words_xml: Path) -> dict[str, tuple[str, float, float, bool]]:
    """word_id → (surface_text, t_start_s, t_end_s, is_punct)."""
    tree = ET.parse(speaker_words_xml)
    out = {}
    for w in tree.getroot():
        if not w.tag.endswith("w"):
            continue
        wid = w.attrib.get(f"{{{NITE_NS}}}id", "")
        text = (w.text or "").strip()
        is_punct = w.attrib.get("punc", "false") == "true"
        try:
            t_start = float(w.attrib.get("starttime", "nan"))
            t_end = float(w.attrib.get("endtime", "nan"))
        except ValueError:
            continue
        out[wid] = (text, t_start, t_end, is_punct)
    return out


def parse_span_href(href: str) -> tuple[str, str]:
    """Return (start_word_id, end_word_id) from a `Foo.words.xml#id(X)..id(Y)` href."""
    m = re.match(r"[^#]*#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", href)
    if not m:
        return ("", "")
    return (m.group(1), m.group(2) or m.group(1))


def load_segments_for(speaker_segments_xml: Path) -> list[dict]:
    """Each segment: {'segment_id', 't_start_s', 't_end_s', 'word_ids': [...]}."""
    tree = ET.parse(speaker_segments_xml)
    out = []
    for seg in tree.getroot():
        if not seg.tag.endswith("segment"):
            continue
        seg_id = seg.attrib.get(f"{{{NITE_NS}}}id", "")
        try:
            t_start = float(seg.attrib.get("transcriber_start", "nan"))
            t_end = float(seg.attrib.get("transcriber_end", "nan"))
        except ValueError:
            continue
        word_ids: list[str] = []
        for child in seg:
            if child.tag.endswith("child"):
                href = child.attrib.get("href", "")
                start_wid, end_wid = parse_span_href(href)
                if not start_wid:
                    continue
                # We only collect endpoints; the consumer expands these against
                # the speaker's word list (they're in document order).
                word_ids.append(start_wid)
                if end_wid != start_wid:
                    word_ids.append(end_wid)
        out.append({"segment_id": seg_id, "t_start_s": t_start, "t_end_s": t_end,
                    "word_endpoint_ids": word_ids})
    return out


def expand_word_span(word_order: list[str], endpoints: list[str]) -> list[str]:
    """Given a list of (start_id, end_id) endpoint pairs and the speaker's
    word IDs in document order, expand to all word IDs spanned (inclusive)."""
    if not endpoints:
        return []
    # endpoints come in start/end pairs (or singletons). Walk them.
    spans = []
    i = 0
    while i < len(endpoints):
        start = endpoints[i]
        if i + 1 < len(endpoints):
            end = endpoints[i + 1]
            i += 2
        else:
            end = start
            i += 1
        spans.append((start, end))
    idx_of = {w: i for i, w in enumerate(word_order)}
    out: list[str] = []
    for start, end in spans:
        si = idx_of.get(start)
        ei = idx_of.get(end)
        if si is None or ei is None:
            continue
        if si > ei:
            si, ei = ei, si
        out.extend(word_order[si:ei + 1])
    return out


def load_das_for(speaker_da_xml: Path, da_id_to_name: dict[str, str]) -> list[dict]:
    """Each DA: {'da_id', 'da_name', 'word_endpoint_ids'}."""
    if not speaker_da_xml.exists():
        return []
    tree = ET.parse(speaker_da_xml)
    out = []
    for dact in tree.getroot():
        if not dact.tag.endswith("dact"):
            continue
        # da-aspect via pointer
        da_aspect_id = ""
        for ptr in dact.findall("{%s}pointer" % NITE_NS):
            if ptr.attrib.get("role") == "da-aspect":
                m = re.search(r"#id\(([^)]+)\)", ptr.attrib.get("href", ""))
                if m:
                    da_aspect_id = m.group(1)
                break
        # word endpoints via child
        word_endpoints: list[str] = []
        for child in dact.findall("{%s}child" % NITE_NS):
            href = child.attrib.get("href", "")
            s, e = parse_span_href(href)
            if s:
                word_endpoints.extend([s, e])
        out.append({
            "da_id": dact.attrib.get(f"{{{NITE_NS}}}id", ""),
            "da_name": da_id_to_name.get(da_aspect_id, "?"),
            "word_endpoint_ids": word_endpoints,
        })
    return out


def load_da_ontology(da_types_xml: Path) -> dict[str, str]:
    tree = ET.parse(da_types_xml)
    out = {}
    for n in tree.iter():
        if "name" in n.attrib and n.attrib.get(f"{{{NITE_NS}}}id", "").startswith("ami_da_"):
            out[n.attrib[f"{{{NITE_NS}}}id"]] = n.attrib["name"]
    return out


# ---------------------------------------------------------------------------
# Text cleaning for MFA
# ---------------------------------------------------------------------------

def clean_for_mfa(text: str) -> str:
    """Lowercase, strip punctuation runs, collapse whitespace. MFA expects
    plain words; its dictionary lookup handles apostrophes."""
    text = text.lower()
    # Drop bracketed annotations
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    # Collapse all non-alphanumeric (keep apostrophes)
    text = re.sub(r"[^\w'\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Audio slicing
# ---------------------------------------------------------------------------

def slice_audio(in_wav: Path, out_wav: Path, t_start_s: float, t_end_s: float) -> bool:
    """Write a slice of the input wav at [t_start_s, t_end_s]. Returns True on
    success."""
    try:
        info = sf.info(str(in_wav))
        sr = info.samplerate
        s = max(0, int(t_start_s * sr))
        e = min(info.frames, int(t_end_s * sr))
        if e <= s:
            return False
        data, _ = sf.read(str(in_wav), start=s, stop=e, dtype="int16")
        sf.write(str(out_wav), data, sr, subtype="PCM_16")
        return True
    except Exception as exc:
        print(f"  [warn] slice failed for {out_wav.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main per-meeting routine
# ---------------------------------------------------------------------------

def process_meeting(
    meeting: str,
    audio_root: Path,
    ami_root: Path,
    out_root: Path,
    meeting_speakers: dict,
    da_id_to_name: dict[str, str],
    min_duration_s: float = 0.1,
) -> dict:
    speakers = meeting_speakers.get(meeting, {})
    if not speakers:
        return {"meeting": meeting, "status": "no-speakers-found"}

    out_dir = out_root / meeting
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    manifest = []

    n_segments_total = 0
    n_segments_kept = 0

    for letter, info in sorted(speakers.items()):
        channel = info["channel"]
        wav_path = audio_root / meeting / f"Headset-{channel}.wav"
        if not wav_path.exists():
            print(f"  [skip] {meeting} speaker {letter}: missing {wav_path}")
            continue

        seg_xml = ami_root / "segments" / f"{meeting}.{letter}.segments.xml"
        word_xml = ami_root / "words" / f"{meeting}.{letter}.words.xml"
        da_xml = ami_root / "dialogueActs" / f"{meeting}.{letter}.dialog-act.xml"
        if not seg_xml.exists() or not word_xml.exists():
            print(f"  [skip] {meeting} speaker {letter}: missing segments or words XML")
            continue

        words = load_words_for(word_xml)
        # word IDs in document order
        word_order = list(words.keys())
        segments = load_segments_for(seg_xml)
        das = load_das_for(da_xml, da_id_to_name)

        # Index DAs by their (start_word_idx, end_word_idx) so we can match against segments
        idx_of_word = {w: i for i, w in enumerate(word_order)}
        da_ranges = []
        for d in das:
            wids = expand_word_span(word_order, d["word_endpoint_ids"])
            if not wids:
                continue
            si = idx_of_word.get(wids[0], -1)
            ei = idx_of_word.get(wids[-1], -1)
            if si < 0 or ei < 0:
                continue
            da_ranges.append({"da_id": d["da_id"], "da_name": d["da_name"],
                              "word_si": si, "word_ei": ei})

        for sidx, seg in enumerate(segments):
            n_segments_total += 1
            seg_word_ids = expand_word_span(word_order, seg["word_endpoint_ids"])
            seg_word_ids = [w for w in seg_word_ids if w in words]
            non_punct_words = [(words[w][0], words[w][1], words[w][2])
                                for w in seg_word_ids if not words[w][3]]
            text_raw = " ".join(t for t, _, _ in non_punct_words)
            text = clean_for_mfa(text_raw)
            if not text:
                continue
            # Use the actual word time bounds rather than transcriber_start/end —
            # they are slightly tighter and better aligned to actual speech.
            t_start = min(t for _, t, _ in non_punct_words)
            t_end = max(t2 for _, _, t2 in non_punct_words)
            duration = t_end - t_start
            if duration < min_duration_s:
                continue

            # Compute DAs whose word-range overlaps this segment
            seg_si = idx_of_word.get(seg_word_ids[0], -1)
            seg_ei = idx_of_word.get(seg_word_ids[-1], -1)
            das_in_seg = [d["da_name"] for d in da_ranges
                           if d["word_si"] <= seg_ei and d["word_ei"] >= seg_si]

            utt_id = f"{meeting}_{letter}_seg{sidx:04d}"
            wav_out = out_dir / f"{utt_id}.wav"
            lab_out = out_dir / f"{utt_id}.lab"
            if not slice_audio(wav_path, wav_out, t_start, t_end):
                continue
            lab_out.write_text(text + "\n")
            manifest.append({
                "utterance_id": utt_id,
                "meeting": meeting,
                "speaker_letter": letter,
                "global_name": info["global_name"],
                "role": info["role"],
                "channel": channel,
                "segment_id": seg["segment_id"],
                "t_start_s": float(t_start),
                "t_end_s": float(t_end),
                "duration_s": float(duration),
                "n_words": len(non_punct_words),
                "text": text,
                "text_raw": text_raw,
                "das_in_seg": das_in_seg,
                "wav": str(wav_out),
                "lab": str(lab_out),
            })
            n_segments_kept += 1

    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    return {"meeting": meeting, "status": "ok", "n_segments_total": n_segments_total,
            "n_segments_kept": n_segments_kept, "manifest": str(manifest_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", required=True,
                        help="Meeting ID (e.g., ES2002a) or 'all-local' to process every meeting "
                             "with audio under --audio-root.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/ami/audio"))
    parser.add_argument("--ami-root", type=Path, default=Path("data/ami/ami_annotations"))
    parser.add_argument("--out", type=Path, default=Path("data/ami/mfa_input"))
    args = parser.parse_args()

    meetings_xml = args.ami_root / "corpusResources" / "meetings.xml"
    da_types_xml = args.ami_root / "ontologies" / "da-types.xml"
    print(f"[load] meetings_xml = {meetings_xml}")
    meeting_speakers = load_meeting_speakers(meetings_xml)
    print(f"  {len(meeting_speakers)} meetings with speaker mapping")
    da_id_to_name = load_da_ontology(da_types_xml)

    if args.meeting == "all-local":
        targets = sorted(p.name for p in args.audio_root.iterdir()
                          if p.is_dir() and (p / "Headset-0.wav").exists())
        print(f"[scan] {len(targets)} meetings with local IH audio: {targets[:5]}{'...' if len(targets) > 5 else ''}")
    else:
        targets = [args.meeting]

    for m in targets:
        print(f"\n[process] {m}")
        result = process_meeting(m, args.audio_root, args.ami_root, args.out,
                                  meeting_speakers, da_id_to_name)
        print(f"  → {result}")


if __name__ == "__main__":
    main()
