import argparse
import csv
import json
import os
import re
import subprocess
import random
import time
from typing import List, Dict, Any

import yt_dlp
from spotify_scraper import SpotifyClient

CSV_DEFAULT = "top_10000_1950-now.csv"
OUT_DIR_DEFAULT = "music"
METADATA_FILE_DEFAULT = "downloads_metadata.json"

# Utilidades

def safe_filename(s: str, max_len: int = 200) -> str:
    s = s.strip()
    # reemplazar caracteres no válidos por underscore
    s = re.sub(r"[^0-9A-Za-z\-\._ ]+", "_", s)
    s = s.replace(" ", "_")
    return s[:max_len]


def parse_range(range_str: str, total: int) -> range:
    """Range string en formato 'start-end' (1-based inclusive). Devuelve objeto range con índices 0-based."""
    if not range_str:
        return range(0, total)
    if "-" in range_str:
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 1
        end = int(parts[1]) if parts[1] else total
    else:
        # single number
        start = int(range_str)
        end = start
    # convertir a 0-based and clamp
    start_idx = max(0, start - 1)
    end_idx = min(total - 1, end - 1)
    return range(start_idx, end_idx + 1)


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def order_rows(rows: List[Dict[str, str]], order: str) -> List[Dict[str, str]]:
    key = order or "original"
    if key == "original":
        return rows
    if key == "shuffle":
        shuffled = rows[:] 
        random.shuffle(shuffled)
        return shuffled
    if key == "popularity":
        if "Popularity" in rows[0]:
            return sorted(rows, key=lambda r: int(r.get("Popularity", 0) or 0), reverse=True)
        else:
            return rows
    if key == "release_date":
        if "Album Release Date" in rows[0]:
            def parse_date(r):
                v = r.get("Album Release Date", "")
                return v or "0000"
            return sorted(rows, key=parse_date)
        else:
            return rows
    return rows


def yt_search_and_download(query: str, outdir: str, prefer_webm: bool = True, verbose: bool = False) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)

    search = f"ytsearch1:{query}"
    ydl_opts_search = { 'quiet': True, 'skip_download': True }
    with yt_dlp.YoutubeDL(ydl_opts_search) as ydl:
        info = ydl.extract_info(search, download=False)
    if not info or 'entries' not in info or len(info['entries']) == 0:
        raise RuntimeError("No se encontró resultado en YouTube para: {}".format(query))
    video = info['entries'][0]

    video_url = video.get('webpage_url')
    video_title = video.get('title', 'video')
    safe_title = safe_filename(video_title)

    # preferencia por webm
    format_str = 'bestaudio[ext=webm]/bestaudio' if prefer_webm else 'bestaudio'

    # plantilla de salida: ponemos el título y extensión variable
    outtmpl = os.path.join(outdir, f"{safe_title}.%(ext)s")

    ydl_opts_dl = {
        'format': format_str,
        'outtmpl': outtmpl,
        'quiet': not verbose,
        'noplaylist': True,
    }

    downloaded_filepath = None
    downloaded_info = None
    with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
        downloaded_info = ydl.extract_info(video_url, download=True)
        # ydl will save file according to outtmpl; we can reconstruct filename
        ext = downloaded_info.get('ext') or downloaded_info.get('abr') or 'webm'
        downloaded_filepath = os.path.join(outdir, f"{safe_title}.{ext}")

    # Si el archivo no está en webm, intentamos convertir a webm para cumplir el requisito
    final_path = downloaded_filepath
    if not final_path.lower().endswith('.webm'):
        converted = os.path.join(outdir, f"{safe_title}.webm")
        try:
            # Convertir a webm usando ffmpeg (libopus)
            subprocess.run(["ffmpeg", "-y", "-i", downloaded_filepath, "-c:a", "libopus", converted], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # si conversión correcta, eliminar original y usar converted
            try:
                os.remove(downloaded_filepath)
            except Exception:
                pass
            final_path = converted
        except Exception as e:
            # si falla la conversión, dejar el original y avisar
            print(f"Advertencia: no se pudo convertir {downloaded_filepath} a webm: {e}")
            final_path = downloaded_filepath

    return {
        'video_info': {
            'id': video.get('id'),
            'title': video_title,
            'uploader': video.get('uploader'),
            'webpage_url': video_url,
            'duration': video.get('duration'),
            'view_count': video.get('view_count'),
            'like_count': video.get('like_count') if 'like_count' in video else None
        },
        'local_file': os.path.abspath(final_path)
    }


def save_metadata_entry(metadata_file: str, entry: Dict[str, Any]):
    # cargar existentes si hay
    data = []
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Batch downloader: descarga canciones a partir del CSV con track URIs de Spotify')
    parser.add_argument('--csv', default=CSV_DEFAULT, help='Ruta al CSV con Track URI (por defecto: %(default)s)')
    parser.add_argument('--range', default=None, help="Rango 1-based inclusivo (ej. 1-10) o número único (ej. 5)")
    parser.add_argument('--order', choices=['original','popularity','release_date','shuffle'], default='original', help='Orden para leer el dataset')
    parser.add_argument('--outdir', default=OUT_DIR_DEFAULT, help='Directorio destino donde se guardan los .webm (por defecto: %(default)s)')
    parser.add_argument('--metadata', default=METADATA_FILE_DEFAULT, help='Archivo JSON donde se guardan los metadatos')
    parser.add_argument('--limit', type=int, default=None, help='Máximo de canciones a descargar desde el rango seleccionado')
    parser.add_argument('--sleep', type=float, default=1.0, help='Segundos a esperar entre descargas para no sobrecargar servicios')

    args = parser.parse_args()

    rows = load_csv_rows(args.csv)
    if not rows:
        print("CSV vacío o no legible.")
        return

    rows = order_rows(rows, args.order)

    sel_range = parse_range(args.range, len(rows))
    indices = list(sel_range)
    if args.limit is not None:
        indices = indices[:args.limit]

    print(f"Seleccionadas {len(indices)} filas (de {len(rows)}) para procesar.")

    client = SpotifyClient()

    for idx in indices:
        row = rows[idx]
        track_uri = row.get('Track URI') or row.get('TrackURI') or row.get('track_uri')
        if not track_uri:
            print(f"Índice {idx+1}: no se encontró Track URI, saltando.")
            continue

        # extraer id y crear open.spotify URL
        try:
            track_id = track_uri.split(':')[-1]
            track_url = f"https://open.spotify.com/track/{track_id}"
        except Exception:
            print(f"Índice {idx+1}: formato de Track URI inválido ({track_uri}), saltando.")
            continue

        print(f"\n[{idx+1}] Procesando: {track_uri} -> {track_url}")

        try:
            track_info = client.get_track_info(track_url)
        except Exception as e:
            print(f"Error obteniendo info de Spotify para {track_uri}: {e}")
            track_info = {}

        # preparar query de búsqueda en YouTube
        name = (track_info.get('name') if isinstance(track_info, dict) else None) or row.get('Track Name') or ''
        artists = ''
        if isinstance(track_info, dict):
            artists_field = track_info.get('artists')
            if isinstance(artists_field, list):
                artists = ", ".join([a.get('name','') for a in artists_field if isinstance(a, dict)])
            else:
                artists = track_info.get('artists', '')
        if not artists:
            # intentar desde CSV
            artists = row.get('Artist Name(s)') or row.get('Artist Name') or ''

        query = f"{name} {artists}".strip()
        if not query:
            print("No hay nombre/artist para buscar en YouTube, saltando.")
            continue

        try:
            result = yt_search_and_download(query, args.outdir, prefer_webm=True, verbose=False)
        except Exception as e:
            print(f"Error buscando/descargando en YouTube para '{query}': {e}")
            continue

        metadata_entry = {
            'index_in_csv': idx + 1,
            'track_uri': track_uri,
            'spotify': track_info,
            'youtube': result.get('video_info'),
            'local_file': result.get('local_file'),
            'csv_row': row,
            'timestamp': time.time()
        }

        # guardar metadata incremental
        save_metadata_entry(args.metadata, metadata_entry)

        print(f"Descargado y guardado metadata para índice {idx+1} -> {metadata_entry['local_file']}")
        time.sleep(args.sleep)

    try:
        client.close()
    except Exception:
        pass

    print('\nProceso finalizado.')


if __name__ == '__main__':
    main()
