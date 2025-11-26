#!/usr/bin/env python3
"""
hum_query.py

Search the ChromaDB collection using a hummed MP3/WAV and return top-K results.

Output per result (as requested):
 1) id (only the numeric prefix) | emb_dist | dtw
 2) artist (from downloads_metadata.json) | song (from downloads_metadata.json)
 3) youtube link (from downloads_metadata.json) | timestamp (from downloads_metadata.json)

If downloads metadata for a candidate is not found, the script falls back to
existing parsing heuristics in the stored metadata.
"""
import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dtaidistance import dtw_ndim

from db_connector import get_music_collection
from songConverter import generar_vector_chroma_cens

# Path to downloads metadata file
DOWNLOADS_METADATA = "downloads_metadata.json"

ID_PREFIX_RE = re.compile(r"^(\d+)_")

def load_downloads_metadata(path: str) -> List[Dict[str, Any]]:
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

def build_summary_embedding_from_chroma(chroma: List[List[float]], target_dim: int = 10) -> List[float]:
    arr = np.asarray(chroma, dtype=float)
    mean = np.mean(arr, axis=0)
    vec = mean.tolist()
    if len(vec) >= target_dim:
        return [float(v) for v in vec[:target_dim]]
    else:
        padded = vec + [0.0] * (target_dim - len(vec))
        return [float(v) for v in padded]

def try_load_chroma_from_metadata(meta: Dict[str, Any]) -> Optional[np.ndarray]:
    candidates = ['vector_cens', 'contorno_json', 'contorno', 'contour']
    for key in candidates:
        if key in meta:
            val = meta[key]
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                except Exception:
                    continue
            else:
                parsed = val
            try:
                arr = np.asarray(parsed, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 12:
                    return arr
                if arr.ndim == 2 and arr.shape[0] == 12:
                    return arr.T
            except Exception:
                continue
    return None

def normalize_distance_for_dtw(dtw_dist: float, len_a: int, len_b: int) -> float:
    denom = (len_a + len_b)
    if denom <= 0:
        return float('inf')
    return float(dtw_dist) / denom

def _first_list_or_flat(v):
    if v is None:
        return []
    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
        return v[0]
    return v

def sanitize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).strip()

def parse_artist_and_title_from_meta(meta: Dict[str, Any]) -> Tuple[str, str]:
    artist = sanitize_text(meta.get("artist") or meta.get("uploader") or meta.get("author") or "")
    title = sanitize_text(meta.get("title") or meta.get("song_basename") or meta.get("cancion") or "")
    if (not artist or not title):
        src = meta.get("source_metadata") or meta.get("source") or meta.get("sourceMeta")
        if isinstance(src, str):
            try:
                src = json.loads(src)
            except Exception:
                src = None
        if isinstance(src, dict):
            title = title or sanitize_text(src.get("title") or src.get("fulltitle") or src.get("display_title") or src.get("name"))
            artist = artist or sanitize_text(src.get("artist") or src.get("uploader") or src.get("channel"))
            combined = title or ""
            if " - " in combined and not artist:
                parts = combined.split(" - ", 1)
                artist = sanitize_text(parts[0])
                title = sanitize_text(parts[1])
    if title and " - " in title and not artist:
        parts = title.split(" - ", 1)
        artist = sanitize_text(parts[0])
        title = sanitize_text(parts[1])
    return (artist or "Unknown Artist", title or "Unknown Song")

def timestamp_to_iso(ts: Any) -> str:
    try:
        t = float(ts)
        if t > 1e12:
            t = t / 1000.0
        return datetime.fromtimestamp(t).isoformat()
    except Exception:
        try:
            return str(ts)
        except Exception:
            return "(unknown time)"

def extract_link_from_dl(dl_meta: Dict[str, Any]) -> Optional[str]:
    if not dl_meta:
        return None

    # 1) Prefer explicit youtube sub-object fields (webpage_url, url, original_url)
    y = dl_meta.get("youtube")
    if isinstance(y, dict):
        for k in ("webpage_url", "url", "original_url"):
            if k in y and y[k]:
                return sanitize_text(y[k])
        # fallback to known video id keys and build a youtube watch URL
        for vidk in ("videoId", "video_id", "id"):
            if vidk in y and y[vidk]:
                return f"https://www.youtube.com/watch?v={sanitize_text(y[vidk])}"

    # 2) Then prefer any top-level webpage/url/original_url keys
    for k in ("webpage_url", "url", "original_url"):
        if k in dl_meta and dl_meta[k]:
            return sanitize_text(dl_meta[k])

    # 3) If a dedicated track_uri exists and it's a youtube http link, prefer that
    track_uri = dl_meta.get("track_uri")
    if track_uri and isinstance(track_uri, str):
        if track_uri.startswith("http://") or track_uri.startswith("https://"):
            return sanitize_text(track_uri)
        # sometimes track_uri contains a spotify:... value; defer spotify to last resort

    # 4) Check nested youtube-like keys inside top-level objects if present
    #    (some entries may have 'youtube' stored as stringified JSON; attempt to parse)
    maybe_src = dl_meta.get("source_metadata") or dl_meta.get("source")
    if isinstance(maybe_src, str):
        try:
            parsed = json.loads(maybe_src)
            if isinstance(parsed, dict):
                for k in ("webpage_url", "url", "original_url"):
                    if k in parsed and parsed[k]:
                        return sanitize_text(parsed[k])
        except Exception:
            pass

    # 5) As a last resort, fall back to spotify track URI if present
    sp = dl_meta.get("track_uri") or (dl_meta.get("spotify") and dl_meta.get("spotify").get("uri"))
    if sp:
        return sanitize_text(sp)

    return None

def extract_artist_song_from_dl(dl_meta: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if not dl_meta:
        return (None, None)
    # direct fields
    artist = _first_nonempty(dl_meta.get("artist"), dl_meta.get("uploader"))
    title = _first_nonempty(dl_meta.get("title"), dl_meta.get("fulltitle"))
    # check youtube object for title or uploader
    y = dl_meta.get("youtube")
    if isinstance(y, dict):
        title = title or _first_nonempty(y.get("title"), y.get("video_title"))
        artist = artist or _first_nonempty(y.get("uploader"), y.get("channel"))
        # sometimes the youtube title has "Artist - Song"
        if title and " - " in title and not artist:
            p = title.split(" - ", 1)
            artist = p[0].strip()
            title = p[1].strip()
    # spotify object: artists list or name fields
    sp = dl_meta.get("spotify")
    if isinstance(sp, dict):
        if not artist:
            # try to extract first artist name
            a = sp.get("artists")
            if isinstance(a, list) and len(a) > 0:
                if isinstance(a[0], dict) and a[0].get("name"):
                    artist = a[0].get("name")
                elif isinstance(a[0], str):
                    artist = a[0]
        if not title:
            title = _first_nonempty(sp.get("name"), sp.get("title"), sp.get("fulltitle"))
    # as final fallback try splitting a combined field in top-level title
    if not artist and title and " - " in title:
        a, t = title.split(" - ", 1)
        artist = a.strip()
        title = t.strip()
    return (artist, title)

def _first_nonempty(*args):
    for a in args:
        if a:
            return a
    return None

def make_ansi_hyperlink(url: str, text: str) -> str:
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"

def pretty_print_result_line_numeric(rank: int, numeric_id: Optional[int], emb_dist: Optional[float], dtw_score: Optional[float]):
    id_part = str(numeric_id) if numeric_id is not None else "(no-id)"
    parts = [f"{rank:2d}. id={id_part}"]
    if emb_dist is not None:
        parts.append(f"emb_dist={float(emb_dist):.4f}")
    if dtw_score is not None:
        parts.append(f"dtw={float(dtw_score):.4f}")
    print(" | ".join(parts))

def id_prefix_to_index(fid: str) -> Optional[int]:
    m = ID_PREFIX_RE.match(fid)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser(description="Search ChromaDB with a hummed MP3/WAV and return formatted results.")
    p.add_argument("--file", "-f", required=True, help="Path to query audio file (mp3/wav).")
    p.add_argument("--topk", "-k", type=int, default=10, help="Number of unique songs to return.")
    p.add_argument("--candidate-mult", type=int, default=4, help="Multiplier for how many DB candidates to fetch before dedupe (topk * mult).")
    p.add_argument("--rerank-dtw", action="store_true", help="Rerank top candidates using multidimensional DTW on chroma sequences (slower).")
    p.add_argument("--no-ansi", action="store_true", help="Disable ANSI clickable hyperlinks in output.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--unique", dest="unique", action="store_true", help="Enable deduplication by artist+song (default).")
    group.add_argument("--no-unique", dest="unique", action="store_false", help="Disable deduplication; return raw candidates.")
    p.set_defaults(unique=True)
    p.add_argument("--quiet", action="store_true", help="Minimize stdout.")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    downloads_entries = load_downloads_metadata(DOWNLOADS_METADATA)

    if not args.quiet: print("-> Extracting Chroma CENS from query audio...")
    q_chroma = generar_vector_chroma_cens(args.file)
    if q_chroma is None:
        print("Failed to extract chroma from query audio.")
        sys.exit(1)
    q_seq = np.array(q_chroma, dtype=float)
    if q_seq.ndim != 2 or q_seq.shape[1] != 12:
        print("Unexpected chroma shape from query; expected (T,12).")
        sys.exit(1)

    q_emb = build_summary_embedding_from_chroma(q_chroma, target_dim=10)

    col = get_music_collection()
    if not col:
        sys.exit(1)

    candidate_count = max(args.topk * args.candidate_mult, args.topk + 10)

    if not args.quiet: print("-> Querying ChromaDB by summary embedding...")
    include = ["metadatas", "distances"]
    try:
        result = col.query(query_embeddings=[q_emb], n_results=candidate_count, include=include)
    except Exception:
        try:
            result = col.query(queries=[q_emb], n_results=candidate_count, include=include)
        except Exception:
            print("ChromaDB query failed. Traceback:")
            traceback.print_exc()
            sys.exit(1)

    ids = _first_list_or_flat(result.get("ids"))
    metadatas = _first_list_or_flat(result.get("metadatas"))
    distances = _first_list_or_flat(result.get("distances"))
    if (not ids or len(ids) == 0) and result.get("documents"):
        ids = _first_list_or_flat(result.get("documents"))

    candidates = []
    L = max(len(ids or []), len(metadatas or []), len(distances or []))
    for i in range(L):
        cid = ids[i] if i < len(ids) else f"unknown_{i}"
        meta = metadatas[i] if i < len(metadatas) else {}
        emb_dist = distances[i] if i < len(distances) else None

        # find matching downloads entry by numeric id prefix
        dl_meta = None
        idx = id_prefix_to_index(cid)
        if idx is not None and downloads_entries:
            dl_meta = choose_latest_entry_for_index(downloads_entries, idx)

        merged_meta = dict(meta or {})
        if dl_meta:
            # inject prioritized display fields but avoid local file paths
            dl_url = extract_link_from_dl(dl_meta)
            dl_artist, dl_title = extract_artist_song_from_dl(dl_meta)
            if dl_artist:
                merged_meta["artist"] = dl_artist
            if dl_title:
                merged_meta["title"] = dl_title
            if dl_url:
                merged_meta["download_url"] = dl_url
            ts = _first_nonempty(dl_meta.get("timestamp_downloaded"), dl_meta.get("timestamp"))
            if ts is not None:
                merged_meta["download_timestamp"] = ts
            # store a sanitized copy of dl_meta (remove local_file/local_path)
            merged_meta["download_entry"] = {k: v for k, v in dl_meta.items() if k not in ("local_file", "local_path")}
        candidates.append({
            "id": cid,
            "numeric_id": idx,
            "metadata": merged_meta or {},
            "dl": dl_meta,
            "emb_dist": emb_dist
        })

    if not args.quiet:
        print(f"-> Got {len(candidates)} candidates from ChromaDB (pre-dedupe).")

    dtw_scores: Dict[str, Optional[float]] = {}
    if args.rerank_dtw:
        if not args.quiet: print("-> Attempting DTW re-ranking (this may be slow)...")
        for c in candidates:
            meta = c["metadata"] or {}
            target_seq = try_load_chroma_from_metadata(meta)
            if target_seq is None:
                try:
                    got = col.get(ids=[c["id"]], include=["metadatas"])
                    got_meta = _first_list_or_flat(got.get("metadatas")) if got.get("metadatas") else None
                    if isinstance(got_meta, list) and len(got_meta) > 0:
                        got_meta = got_meta[0]
                    if isinstance(got_meta, dict):
                        target_seq = try_load_chroma_from_metadata(got_meta)
                except Exception:
                    target_seq = None
            if target_seq is None:
                dtw_scores[c["id"]] = None
                continue
            try:
                d = dtw_ndim.distance(q_seq.astype(float), target_seq.astype(float))
                dtw_scores[c["id"]] = normalize_distance_for_dtw(d, len(q_seq), len(target_seq))
            except Exception:
                dtw_scores[c["id"]] = None

        def sort_key(item):
            dt = dtw_scores.get(item["id"])
            if dt is not None:
                return dt
            if item.get("emb_dist") is not None:
                return float(item.get("emb_dist"))
            return float("inf")

        candidates.sort(key=sort_key)
    else:
        candidates.sort(key=lambda x: float(x["emb_dist"]) if x.get("emb_dist") is not None else float("inf"))

    # Deduplicate if requested (prefer artist/song from downloads JSON when available)
    selected: List[Dict[str, Any]] = []
    if args.unique:
        seen_songs = set()
        for c in candidates:
            dl = c.get("dl")
            if dl:
                artist, song = extract_artist_song_from_dl(dl)
                artist = sanitize_text(artist)
                song = sanitize_text(song)
            else:
                artist, song = parse_artist_and_title_from_meta(c.get("metadata") or {})
                artist = sanitize_text(artist)
                song = sanitize_text(song)

            song_key = (artist + "||" + song).lower()
            if song_key in seen_songs:
                continue

            selected.append({**c, "artist": artist or "Unknown Artist", "song": song or "Unknown Song"})
            seen_songs.add(song_key)
            if len(selected) >= args.topk:
                break

        # If not enough unique, fill with remaining candidates (still respecting seen_songs)
        if len(selected) < args.topk:
            for c in candidates:
                dl = c.get("dl")
                if dl:
                    artist, song = extract_artist_song_from_dl(dl)
                    artist = sanitize_text(artist)
                    song = sanitize_text(song)
                else:
                    artist, song = parse_artist_and_title_from_meta(c.get("metadata") or {})
                    artist = sanitize_text(artist)
                    song = sanitize_text(song)

                song_key = (artist + "||" + song).lower()
                if song_key in seen_songs:
                    continue

                selected.append({**c, "artist": artist or "Unknown Artist", "song": song or "Unknown Song"})
                seen_songs.add(song_key)
                if len(selected) >= args.topk:
                    break
    else:
        # no unique: simply take topk candidates (preserve existing behavior)
        for c in candidates[: args.topk]:
            dl = c.get("dl")
            if dl:
                artist, song = extract_artist_song_from_dl(dl)
                artist = sanitize_text(artist) or "Unknown Artist"
                song = sanitize_text(song) or "Unknown Song"
            else:
                artist, song = parse_artist_and_title_from_meta(c.get("metadata") or {})
            selected.append({**c, "artist": artist, "song": song})

    # Print nicely in the exact format requested
    print("\n=== TOP RESULTS ===\n")
    for rank, s in enumerate(selected, start=1):
        numeric_id = s.get("numeric_id")
        embd = s.get("emb_dist")
        dscore = dtw_scores.get(s["id"]) if args.rerank_dtw else None

        # Prefer values from downloads JSON (dl) for artist/song/link/timestamp
        dl = s.get("dl")
        if dl and isinstance(dl, dict):
            artist, song = extract_artist_song_from_dl(dl)
            artist = artist or s.get("artist", "Unknown Artist")
            song = song or s.get("song", "Unknown Song")
            link = extract_link_from_dl(dl)
            ts = _first_nonempty(dl.get("timestamp_downloaded"), dl.get("timestamp"))
        else:
            artist = s.get("artist", "Unknown Artist")
            song = s.get("song", "Unknown Song")
            link = (s.get("metadata") or {}).get("download_url") or None
            ts = (s.get("metadata") or {}).get("download_timestamp")

        pretty_print_result_line_numeric(rank, numeric_id, embd, dscore)
        # artist | song line
        print(f"    {artist} | {song}")
         # link line (timestamp removed)
        link_text = link or "(no link)"
        if link and not args.no_ansi:
            try:
                link_print = make_ansi_hyperlink(link_text, link_text)
            except Exception:
                link_print = link_text
        else:
            link_print = link_text
        print(f"    {link_print}")
        print("")

    print("Hints:")
    print(" - Results prefer artist/song/link/timestamp from `downloads_metadata.json` (by numeric id prefix).")
    print(" - To disable deduplication (return raw candidates) use --no-unique.")
    print(" - To enable DTW re-ranking add --rerank-dtw (may be slow).")

if __name__ == "__main__":
    main()