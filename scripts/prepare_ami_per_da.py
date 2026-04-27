"""
Prepare AMI per-DA WAVs (one slice per dialogue act, not per segment).

This is a counterpart to `prepare_ami_for_mfa.py` that slices on DA boundaries
rather than transcriber segment boundaries. Motivation: segments in AMI often
contain multiple DAs in chained turns ("Yeah. Right. Did you finish it?"),
so the "last syllable" of a segment is not the last syllable of any one DA.
The position-ablation finding from the cross-corpus probe (last-syl AUC ≈
middle-syl AUC ≈ first-syl AUC on AMI) may be an annotation-granularity
artifact; per-DA extraction tests this directly.

Output layout:
    <out>/<meeting>/
        <meeting>_<speaker>_da<da_index>.wav     (one wav per DA, audio sliced to that DA's word span)
        <meeting>_<speaker>_da<da_index>.lab     (DA's word string, for MFA)
    <out>/<meeting>/manifest.jsonl               (one line per DA)

The manifest schema mirrors the per-segment manifest, but each row corresponds
to a single DA (so there's one DA name in `das_in_seg`, kept as a list for
schema compatibility downstream).

Usage:
    .venv/bin/python scripts/prepare_ami_per_da.py \\
        --meeting all-local \\
        --audio-root data/ami/audio \\
        --ami-root   data/ami/ami_annotations \\
        --out        data/ami/per_da_input
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse the loaders + slicer from the per-segment script
from scripts.prepare_ami_for_mfa import (
    NITE_NS,
    clean_for_mfa,
    expand_word_span,
    load_da_ontology,
    load_das_for,
    load_meeting_speakers,
    load_segments_for,  # noqa: F401  (kept for reference/parity)
    load_words_for,
    slice_audio,
)


def process_meeting(
    meeting: str,
    audio_root: Path,
    ami_root: Path,
    out_root: Path,
    meeting_speakers: dict,
    da_id_to_name: dict[str, str],
    min_duration_s: float = 0.1,
) -> dict:
    """For each (meeting, speaker, DA), slice the per-channel audio to that
    DA's word span and write a per-DA wav + lab file."""
    speakers = meeting_speakers.get(meeting, {})
    if not speakers:
        return {"meeting": meeting, "status": "no-speakers-found"}

    out_dir = out_root / meeting
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    manifest = []

    n_total = 0
    n_kept = 0

    for letter, info in sorted(speakers.items()):
        channel = info["channel"]
        wav_path = audio_root / meeting / f"Headset-{channel}.wav"
        if not wav_path.exists():
            print(f"  [skip] {meeting} speaker {letter}: missing {wav_path}")
            continue

        word_xml = ami_root / "words" / f"{meeting}.{letter}.words.xml"
        da_xml = ami_root / "dialogueActs" / f"{meeting}.{letter}.dialog-act.xml"
        if not word_xml.exists() or not da_xml.exists():
            print(f"  [skip] {meeting} speaker {letter}: missing words or DA XML")
            continue

        words = load_words_for(word_xml)
        word_order = list(words.keys())
        das = load_das_for(da_xml, da_id_to_name)

        for da_idx, d in enumerate(das):
            n_total += 1
            wids = expand_word_span(word_order, d["word_endpoint_ids"])
            wids = [w for w in wids if w in words]
            non_punct_words = [(words[w][0], words[w][1], words[w][2])
                                for w in wids if not words[w][3]]
            text_raw = " ".join(t for t, _, _ in non_punct_words)
            text = clean_for_mfa(text_raw)
            if not text:
                continue
            t_start = min(t for _, t, _ in non_punct_words)
            t_end = max(t2 for _, _, t2 in non_punct_words)
            duration = t_end - t_start
            if duration < min_duration_s:
                continue

            utt_id = f"{meeting}_{letter}_da{da_idx:04d}"
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
                "segment_id": d["da_id"],  # use DA id as segment_id in this layout
                "t_start_s": float(t_start),
                "t_end_s": float(t_end),
                "duration_s": float(duration),
                "n_words": len(non_punct_words),
                "text": text,
                "text_raw": text_raw,
                "das_in_seg": [d["da_name"]],  # singleton list for schema parity
                "wav": str(wav_out),
                "lab": str(lab_out),
            })
            n_kept += 1

    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    return {"meeting": meeting, "status": "ok", "n_total": n_total,
             "n_kept": n_kept, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", required=True,
                        help="Meeting ID or 'all-local' for every meeting with audio under --audio-root.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/ami/audio"))
    parser.add_argument("--ami-root", type=Path, default=Path("data/ami/ami_annotations"))
    parser.add_argument("--out", type=Path, default=Path("data/ami/per_da_input"))
    args = parser.parse_args()

    meeting_speakers = load_meeting_speakers(args.ami_root / "corpusResources" / "meetings.xml")
    da_id_to_name = load_da_ontology(args.ami_root / "ontologies" / "da-types.xml")
    print(f"[load] {len(meeting_speakers)} meetings with speaker mapping")

    if args.meeting == "all-local":
        targets = sorted(p.name for p in args.audio_root.iterdir()
                          if p.is_dir() and (p / "Headset-0.wav").exists())
        print(f"[scan] {len(targets)} meetings with local IH audio")
    else:
        targets = [args.meeting]

    for m in targets:
        print(f"\n[process] {m}")
        result = process_meeting(m, args.audio_root, args.ami_root, args.out,
                                  meeting_speakers, da_id_to_name)
        print(f"  → {result}")


if __name__ == "__main__":
    main()
