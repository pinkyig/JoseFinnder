#!/usr/bin/env python3
"""
split_and_segment.py

Reads the metadata JSON (default: downloads_metadata.json), selects songs
by CSV numeric range (1-based, same semantics as batch_downloader), runs
Demucs two-stems ('vocals') to split vocals vs accompaniment, and segments
each stem into overlapping 12s WAV fragments (step 4s) saved under:
  resultados/1_stems_demucs/htdemucs/<song_name>/
  resultados/2_audio_fragments/<song_name>/vocals_seg_*.wav
  resultados/2_audio_fragments/<song_name>/accompaniment_seg_*.wav

Usage:
  python process_from_metadata.py --metadata downloads_metadata.json --range 1-5
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from typing import List, Dict, Any, Optional

import demucs.separate
import torch
from pydub import AudioSegment

# Directories (keep consistent with repo)
DIR_RESULTADOS = "resultados"
DIR_STEMS = os.path.join(DIR_RESULTADOS, "1_stems_demucs")
DIR_FRAGMENTS = os.path.join(DIR_RESULTADOS, "2_audio_fragments")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_filename(s: str, max_len: int = 200) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^0-9A-Za-z\-\._ ]+", "_", s)
    s = s.replace(" ", "_")
    return s[:max_len]


def parse_range(range_str: Optional[str], total: int) -> range:
    """Same semantics as batch_downloader: accepts 'start-end' (1-based inclusive) or single number.
    Returns a 0-based range object for CSV indices; caller may convert to 1-based index_in_csv."""
    if not range_str:
        return range(0, total)
    if "-" in range_str:
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 1
        end = int(parts[1]) if parts[1] else total
    else:
        start = int(range_str)
        end = start
    start_idx = max(0, start - 1)
    end_idx = min(total - 1, end - 1)
    return range(start_idx, end_idx + 1)


def load_metadata(metadata_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(metadata_file):
        return []
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except Exception:
        return []


def choose_latest_entry_for_index(entries: List[Dict[str, Any]], index_in_csv: int) -> Optional[Dict[str, Any]]:
    """From metadata entries choose the one with index_in_csv == index_in_csv and latest timestamp."""
    candidates = [e for e in entries if int(e.get("index_in_csv", 0) or 0) == index_in_csv]
    if not candidates:
        return None
    # choose by timestamp fields if present (timestamp_downloaded or timestamp)
    def ts(e):
        return float(e.get("timestamp_downloaded") or e.get("timestamp") or 0)
    candidates.sort(key=ts, reverse=True)
    return candidates[0]


def run_demucs_on_file(local_file: str, force: bool = False) -> Optional[str]:
    """Run demucs (htdemucs two-stems vocals). Returns vocals wav path or None."""
    song_name = os.path.splitext(os.path.basename(local_file))[0]
    out_dir = os.path.join(DIR_STEMS, "htdemucs", song_name)
    vocals_path = os.path.join(out_dir, "vocals.wav")

    if os.path.exists(vocals_path) and not force:
        print(f"  - Demucs output already exists for '{song_name}', skipping demucs.")
        return vocals_path

    os.makedirs(DIR_STEMS, exist_ok=True)
    print(f"  - Running Demucs (htdemucs, two-stems vocals) for '{song_name}' on device={DEVICE} ...")
    try:
        demucs.separate.main([
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-o", DIR_STEMS,
            "-d", DEVICE,
            local_file
        ])
    except Exception as e:
        print(f"    Demucs failed for {local_file}: {e}")
        traceback.print_exc()
        return None

    # locate vocals file
    if os.path.exists(vocals_path):
        return vocals_path

    # As fallback, find any wav under out_dir that contains 'voc' or 'vocal'
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            lf = f.lower()
            if lf.endswith(".wav") and ("vocal" in lf or "vocals" in lf):
                return os.path.join(out_dir, f)
        # else return first wav found
        wavs = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".wav")]
        if wavs:
            return wavs[0]
    return None


def locate_separated_files(song_name: str) -> Dict[str, Optional[str]]:
    """Look inside DIR_STEMS/htdemucs/<song> and return paths for vocals and accompaniment (if present)."""
    base = os.path.join(DIR_STEMS, "htdemucs", song_name)
    result = {"vocals": None, "accompaniment": None}
    if not os.path.isdir(base):
        return result
    wavs = [f for f in os.listdir(base) if f.lower().endswith(".wav")]
    wavs_l = [w.lower() for w in wavs]
    # try find vocals
    for i, w in enumerate(wavs_l):
        if "vocal" in w or "vox" in w or "voice" in w:
            result["vocals"] = os.path.join(base, wavs[i])
            break
    # try find accompaniment / no_vocals
    for i, w in enumerate(wavs_l):
        if result["vocals"] and os.path.join(base, wavs[i]) == result["vocals"]:
            continue
        # treat any other wav as accompaniment
        if w.endswith(".wav"):
            result["accompaniment"] = os.path.join(base, wavs[i])
            break
    return result


def segment_audio(input_path: str, out_dir: str, stem_label: str, step_ms: int = 4000, duration_ms: int = 12000) -> List[str]:
    """Segment input_path into overlapping fragments saved under out_dir/<stem_label>_seg_<i>.wav.
    Returns list of created fragment paths.
    """
    fragments = []
    try:
        audio = AudioSegment.from_file(input_path).set_channels(1)
    except Exception as e:
        print(f"    Error loading audio {input_path}: {e}")
        return fragments

    if len(audio) < duration_ms:
        # too short -> skip
        return fragments

    song_frag_dir = os.path.join(out_dir)
    os.makedirs(song_frag_dir, exist_ok=True)

    idx = 0
    # from 0 to len-duration inclusive
    for start in range(0, len(audio) - duration_ms + 1, step_ms):
        out_path = os.path.join(song_frag_dir, f"{stem_label}_seg_{start}.wav")
        if not os.path.exists(out_path):
            try:
                audio[start:start + duration_ms].export(out_path, format="wav")
            except Exception as e:
                print(f"      Failed exporting fragment {out_path}: {e}")
                continue
        fragments.append(out_path)
        idx += 1
    return fragments


def process_entry(local_file: str, force_demucs: bool = False):
    if not os.path.exists(local_file):
        print(f"  - File not found on disk: {local_file} -> skip")
        return

    song_name = safe_filename(os.path.splitext(os.path.basename(local_file))[0])
    print(f"\nProcessing song: {song_name}")
    os.makedirs(DIR_STEMS, exist_ok=True)
    os.makedirs(DIR_FRAGMENTS, exist_ok=True)

    vocals_wav = run_demucs_on_file(local_file, force=force_demucs)
    if not vocals_wav:
        print(f"  - Demucs did not produce a vocals stem for {song_name}, skipping segmentation.")
        return

    located = locate_separated_files(song_name)
    vocals = located.get("vocals")
    accomp = located.get("accompaniment")

    if not vocals:
        # fallback to the vocals_wav returned
        vocals = vocals_wav

    if vocals:
        frag_out_dir = os.path.join(DIR_FRAGMENTS, song_name)
        os.makedirs(frag_out_dir, exist_ok=True)
        print(f"  - Segmenting vocals -> {frag_out_dir} ...")
        vfrags = segment_audio(vocals, frag_out_dir, "vocals")
        print(f"    vocals fragments: {len(vfrags)}")
    else:
        print("  - No vocals stem found to segment.")

    if accomp:
        frag_out_dir = os.path.join(DIR_FRAGMENTS, song_name)
        os.makedirs(frag_out_dir, exist_ok=True)
        print(f"  - Segmenting accompaniment -> {frag_out_dir} ...")
        afrags = segment_audio(accomp, frag_out_dir, "accompaniment")
        print(f"    accompaniment fragments: {len(afrags)}")
    else:
        print("  - No accompaniment stem found to segment.")


def main():
    parser = argparse.ArgumentParser(description="Process songs from metadata: split (Demucs two-stems vocals) and segment stems.")
    parser.add_argument("--metadata", default="downloads_metadata.json", help="Metadata JSON produced by downloader")
    parser.add_argument("--range", default=None, help="CSV numeric range (1-based), e.g. 1-5 or 10")
    parser.add_argument("--force-demucs", action="store_true", help="Force rerunning demucs even if output exists")
    parser.add_argument("--limit", type=int, default=None, help="Max number of CSV indices to process from the selected range")
    args = parser.parse_args()

    entries = load_metadata(args.metadata)
    if not entries:
        print(f"No entries in metadata file: {args.metadata}")
        return

    # determine maximum CSV index covered by metadata
    max_idx = 0
    for e in entries:
        try:
            idx = int(e.get("index_in_csv") or 0)
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    if max_idx == 0:
        print("No valid 'index_in_csv' fields found in metadata.")
        return

    sel = parse_range(args.range, max_idx)
    target_indices = [i + 1 for i in sel]  # convert 0-based -> 1-based indexes present in metadata
    if args.limit is not None:
        target_indices = target_indices[: args.limit]

    print(f"Selected CSV indices to process: {target_indices}")

    for idx in target_indices:
        entry = choose_latest_entry_for_index(entries, idx)
        if not entry:
            print(f"\nIndex {idx}: no metadata entry found -> skipping")
            continue
        local_file = entry.get("local_file") or entry.get("local_path") or entry.get("file")
        if not local_file:
            print(f"\nIndex {idx}: metadata entry has no 'local_file' -> skipping")
            continue
        print(f"\n[{idx}] Found file: {local_file}")
        try:
            process_entry(local_file, force_demucs=args.force_demucs)
        except Exception as e:
            print(f"  ERROR processing index {idx}: {e}")
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()