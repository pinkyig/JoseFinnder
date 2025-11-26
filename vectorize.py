#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import chromadb

# Defaults / paths consistent with repo
METADATA_DEFAULT = "downloads_metadata.json"
DIR_FRAGMENTS = os.path.join("resultados", "2_audio_fragments")
DB_PATH = "music_database"
COLLECTION_NAME = "fragmentos_musicales"

def sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (md or {}).items():
        if v is None:
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[k] = str(v)
    return out

def safe_filename(s: str, max_len: int = 200) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^0-9A-Za-z\-\._ ]+", "_", s)
    s = s.replace(" ", "_")
    return s[:max_len]

def parse_range(range_str: Optional[str], total: int) -> range:
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

def load_metadata(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def choose_latest_entry_for_index(entries: List[Dict[str, Any]], index_in_csv: int) -> Optional[Dict[str, Any]]:
    candidates = [e for e in entries if int(e.get("index_in_csv") or 0) == index_in_csv]
    if not candidates:
        return None
    def ts(e):
        return float(e.get("timestamp_downloaded") or e.get("timestamp") or 0)
    candidates.sort(key=ts, reverse=True)
    return candidates[0]

def find_fragment_files_for_song(song_basename: str) -> List[str]:
    candidates = []
    direct = os.path.join(DIR_FRAGMENTS, song_basename)
    if os.path.isdir(direct):
        candidates.append(direct)
    safe = os.path.join(DIR_FRAGMENTS, safe_filename(song_basename))
    if safe != direct and os.path.isdir(safe):
        candidates.append(safe)
    if os.path.isdir(DIR_FRAGMENTS):
        for d in os.listdir(DIR_FRAGMENTS):
            if d.lower().startswith(song_basename.lower()) or song_basename.lower().startswith(d.lower()):
                p = os.path.join(DIR_FRAGMENTS, d)
                if os.path.isdir(p) and p not in candidates:
                    candidates.append(p)
    files = []
    for c in candidates:
        for f in os.listdir(c):
            if f.lower().endswith(".wav"):
                files.append(os.path.join(c, f))
    files.sort()
    return files

def generate_chroma_cens(ruta_audio: str, hop_length: int = 512, target_sr: int = 22050, downsample: int = 4, min_frames: int = 20) -> Optional[List[List[float]]]:
    try:
        try:
            y, sr = librosa.load(ruta_audio, sr=target_sr, mono=True)
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                return None
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cmd = [ffmpeg, "-y", "-i", ruta_audio, "-ar", str(target_sr), "-ac", "1", tmp]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return None
            y, sr = librosa.load(tmp, sr=target_sr, mono=True)
            try:
                os.remove(tmp)
            except Exception:
                pass

        if y is None or y.size == 0:
            return None
        if y.size < hop_length:
            return None
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.max(np.abs(y))) < 1e-6:
            return None

        try:
            chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, n_octaves=5)
        except Exception:
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            except Exception:
                return None

        chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(chroma, 0.0):
            return None

        chroma = np.asarray(chroma)
        if chroma.ndim != 2:
            return None
        if chroma.shape[0] == 12:
            chroma = chroma.T
        elif chroma.shape[1] == 12:
            chroma = chroma
        else:
            return None

        chroma_down = chroma[::downsample]
        if chroma_down.shape[0] < min_frames:
            return None

        return chroma_down.tolist()
    except Exception:
        traceback.print_exc()
        return None

def build_summary_embedding_from_chroma(chroma: List[List[float]], target_dim: int = 10) -> List[float]:
    arr = np.asarray(chroma, dtype=float)
    mean = np.mean(arr, axis=0)
    vec = mean.tolist()
    if len(vec) >= target_dim:
        return [float(v) for v in vec[:target_dim]]
    else:
        padded = vec + [0.0] * (target_dim - len(vec))
        return [float(v) for v in padded]

def add_fragments_to_chromadb(collection, fragments: List[Dict[str, Any]], skip_existing: bool = True):
    if not fragments:
        return 0, 0
    """
    ids = [f["id"] for f in fragments]
    embeddings = [f["vector_resumen"] for f in fragments]
    metadatas = [f["metadata"] for f in fragments]
    """
    ids = [f["id"] for f in fragments]
    embeddings = [f["vector_resumen"] for f in fragments]
    # ensure metadata values are primitives or strings (ChromaDB requirement)
    metadatas = [sanitize_metadata(f.get("metadata", {})) for f in fragments]
    try:
        print(f"    Attempting batch add of {len(ids)} items to ChromaDB...")
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print("    Batch add succeeded.")
        return len(ids), 0
    except Exception as e:
        print("    Batch add raised exception:")
        traceback.print_exc()
        print("    Falling back to per-item add (with checks).")
        added = 0
        skipped = 0
        for i, fid in enumerate(ids):
            try:
                if skip_existing:
                    try:
                        print(f"      Checking existence for id='{fid}' ...", end=" ")
                        got = collection.get(ids=[fid])
                        print("got:", {k: v for k, v in got.items() if k != "embeddings"})
                        if got and got.get("ids"):
                            print(" -> already exists, skipping.")
                            skipped += 1
                            continue
                    except Exception as eget:
                        print(" check raised exception:")
                        traceback.print_exc()
                        # continue to attempt to add
                print(f"      Adding single id='{fid}' ...", end=" ")
                collection.add(ids=[fid], embeddings=[embeddings[i]], metadatas=[metadatas[i]])
                print("ok")
                added += 1
            except Exception as eitem:
                print(f"      Failed to add id='{fid}':")
                traceback.print_exc()
                continue
        return added, skipped

def main():
    p = argparse.ArgumentParser(description="Vectorize fragment WAVs from metadata range and store in ChromaDB.")
    p.add_argument("--metadata", default=METADATA_DEFAULT)
    p.add_argument("--range", default=None, help="1-based inclusive range e.g. 1-5 or single index like 3")
    p.add_argument("--min-frames", type=int, default=20, help="Minimum chroma frames to accept a fragment")
    p.add_argument("--downsample", type=int, default=4, help="Downsample factor for chroma frames")
    p.add_argument("--limit", type=int, default=None, help="Max CSV indices to process from the selected range")
    p.add_argument("--db-path", default=DB_PATH, help="ChromaDB persistent path")
    p.add_argument("--collection", default=COLLECTION_NAME, help="ChromaDB collection name")
    args = p.parse_args()

    entries = load_metadata(args.metadata)
    if not entries:
        print("No metadata entries found.")
        return

    max_idx = 0
    for e in entries:
        try:
            v = int(e.get("index_in_csv") or 0)
            if v > max_idx:
                max_idx = v
        except Exception:
            continue
    if max_idx == 0:
        print("No valid index_in_csv fields found in metadata.")
        return

    sel = parse_range(args.range, max_idx)
    indices = [i + 1 for i in sel]
    if args.limit is not None:
        indices = indices[: args.limit]

    print(f"Target CSV indices: {indices}")

    print(f"Opening ChromaDB at: '{os.path.abspath(args.db_path)}', collection: '{args.collection}'")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
    try:
        print(f"Collection ready, current count: {collection.count()}")
    except Exception:
        print("Collection.count() raised exception:")
        traceback.print_exc()

    batch_fragments: List[Dict[str, Any]] = []
    total_saved = 0
    total_skipped = 0

    for idx in indices:
        entry = choose_latest_entry_for_index(entries, idx)
        if not entry:
            print(f"[{idx}] No metadata entry found — skipping")
            continue
        local_file = entry.get("local_file") or entry.get("local_path") or entry.get("file")
        track_uri = entry.get("track_uri") or entry.get("trackURI") or entry.get("Track URI")
        if not local_file:
            print(f"[{idx}] Entry missing local_file — skipping")
            continue

        song_basename = os.path.splitext(os.path.basename(local_file))[0]
        fragment_files = find_fragment_files_for_song(song_basename)
        if not fragment_files:
            print(f"[{idx}] No fragment files found for song '{song_basename}' (checked {DIR_FRAGMENTS})")
            continue

        print(f"[{idx}] Found {len(fragment_files)} fragments for '{song_basename}' — vectorizing and queuing to ChromaDB...")
        for frag in fragment_files:
            seg_name = os.path.basename(frag)
            vec = generate_chroma_cens(frag, downsample=args.downsample, min_frames=args.min_frames)
            if not vec:
                print(f"  Fragment {seg_name} -> skipped (no/short vector)")
                continue
            frag_id = f"{idx}_{safe_filename(song_basename)}_{os.path.splitext(seg_name)[0]}"
            vector_resumen = build_summary_embedding_from_chroma(vec, target_dim=10)
            metadata = {
                "index_in_csv": idx,
                "track_uri": track_uri,
                "song_basename": song_basename,
                "fragment": os.path.relpath(frag),
                "frames": len(vec),
                "vector_cens": json.dumps(vec, ensure_ascii=False),
                "source_metadata": json.dumps(entry, ensure_ascii=False)
            }
            batch_fragments.append({
                "id": frag_id,
                "vector_resumen": vector_resumen,
                "metadata": metadata
            })

        if len(batch_fragments) >= 200:
            added, skipped = add_fragments_to_chromadb(collection, batch_fragments, skip_existing=True)
            total_saved += added
            total_skipped += skipped
            print(f"  -> Flushed batch: added={added}, skipped={skipped}")
            # try printing collection.count() after flush
            try:
                print(f"  -> Collection count now: {collection.count()}")
            except Exception:
                print("  -> collection.count() raised exception:")
                traceback.print_exc()
            batch_fragments = []

    if batch_fragments:
        added, skipped = add_fragments_to_chromadb(collection, batch_fragments, skip_existing=True)
        total_saved += added
        total_skipped += skipped
        print(f"  -> Final flush: added={added}, skipped={skipped}")
        try:
            print(f"  -> Collection count now: {collection.count()}")
        except Exception:
            print("  -> collection.count() raised exception:")
            traceback.print_exc()

    print(f"Done. Total added: {total_saved}, total skipped(existing): {total_skipped}.")
    try:
        print(f"Final collection count: {collection.count()}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()