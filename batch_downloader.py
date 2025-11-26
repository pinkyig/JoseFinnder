import argparse
import csv
import json
import os
import re
import subprocess
import random
import time
import multiprocessing
import traceback
from typing import List, Dict, Any

import yt_dlp
from spotify_scraper import SpotifyClient

CSV_DEFAULT = "top_10000_1950-now.csv"
OUT_DIR_DEFAULT = "music_mp3"
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

"""def yt_search_and_stream_convert_to_mp3(query: str, outdir: str, prefer_webm: bool = True, bitrate: str = "320k", verbose: bool = False) -> Dict[str, Any]:
    ...
"""  # omitted large commented block for brevity


def _yt_stream_worker(q, query, outdir, prefer_webm, bitrate, verbose):
    """
    Worker que ejecuta la lógica de yt-dlp + ffmpeg. Devuelve por queue:
      ("ok", result_dict) o ("err", (msg, traceback_str))
    """
    try:
        os.makedirs(outdir, exist_ok=True)

        search = f"ytsearch1:{query}"
        ydl_opts_search = {'quiet': True, 'skip_download': True, 'socket_timeout': 30}
        with yt_dlp.YoutubeDL(ydl_opts_search) as ydl:
            info = ydl.extract_info(search, download=False)
        if not info or 'entries' not in info or len(info['entries']) == 0:
            raise RuntimeError("No se encontró resultado en YouTube para: {}".format(query))
        video = info['entries'][0]

        video_url = video.get('webpage_url')
        video_title = video.get('title', 'video')
        safe_title = safe_filename(video_title)

        # obtener info detallada de la página
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True, 'socket_timeout': 30}) as ydl:
            detailed = ydl.extract_info(video_url, download=False)

        # heurística para obtener url de mejor formato de audio
        audio_url = None
        headers = None
        if isinstance(detailed, dict):
            rf = detailed.get('requested_formats') or detailed.get('formats')
            if isinstance(rf, list) and len(rf) > 0:
                audio_candidates = [f for f in rf if (f.get('acodec') and f.get('acodec') != 'none')]
                if not audio_candidates:
                    audio_candidates = rf
                audio_candidates = sorted(audio_candidates, key=lambda x: float(x.get('tbr') or x.get('abr') or 0), reverse=True)
                audio_url = audio_candidates[0].get('url')
            if not audio_url:
                audio_url = detailed.get('url')
            http_headers = detailed.get('http_headers') or detailed.get('headers')
            if isinstance(http_headers, dict) and len(http_headers) > 0:
                headers = http_headers

        if not audio_url:
            raise RuntimeError("No se pudo obtener URL de audio para: {}".format(video_url))

        out_mp3 = os.path.join(outdir, f"{safe_title}.mp3")

        # construir cmd ffmpeg
        cmd = ["ffmpeg", "-y", "-hide_banner"]
        if headers:
            header_lines = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            cmd += ["-headers", header_lines]
        cmd += ["-i", audio_url, "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, out_mp3]

        if verbose:
            print("FFMPEG CMD:", " ".join(cmd))

        # Run ffmpeg; allow it to write stderr (we capture it to detect errors)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
            result = {
                'video_info': {
                    'id': video.get('id'),
                    'title': video_title,
                    'uploader': video.get('uploader'),
                    'webpage_url': video_url,
                    'duration': video.get('duration'),
                    'view_count': video.get('view_count'),
                    'like_count': video.get('like_count') if 'like_count' in video else None
                },
                'local_file': os.path.abspath(out_mp3)
            }
            q.put(("ok", result))
            return

        # si ffmpeg falló en streaming, intentamos fallback con descarga + convert
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        if verbose:
            print("ffmpeg stderr (stream attempt):", stderr.splitlines()[-30:])
        try:
            if os.path.exists(out_mp3) and os.path.getsize(out_mp3) == 0:
                os.remove(out_mp3)
        except Exception:
            pass

        # FALLBACK: descarga completa con yt-dlp y convertir localmente
        ydl_opts_dl = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(outdir, f"{safe_title}.%(ext)s"),
            'quiet': not verbose,
            'noplaylist': True,
            'socket_timeout': 30
        }
        with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
            info2 = ydl.extract_info(video_url, download=True)
            ext = info2.get('ext') or 'webm'
            downloaded_filepath = os.path.join(outdir, f"{safe_title}.{ext}")

        # convertir con ffmpeg el fichero descargado a mp3 (esto usa disco)
        subprocess.run(["ffmpeg", "-y", "-i", downloaded_filepath, "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, out_mp3], check=True)
        try:
            if os.path.exists(downloaded_filepath) and os.path.abspath(downloaded_filepath) != os.path.abspath(out_mp3):
                os.remove(downloaded_filepath)
        except Exception:
            pass

        result = {
            'video_info': {
                'id': video.get('id'),
                'title': video_title,
                'uploader': video.get('uploader'),
                'webpage_url': video_url,
                'duration': video.get('duration'),
                'view_count': video.get('view_count'),
                'like_count': video.get('like_count') if 'like_count' in video else None
            },
            'local_file': os.path.abspath(out_mp3)
        }
        q.put(("ok", result))
    except Exception as e:
        import traceback as _tb
        q.put(("err", (str(e), _tb.format_exc())))


def yt_search_and_stream_convert_to_mp3(query: str, outdir: str, prefer_webm: bool = True, bitrate: str = "320k", verbose: bool = False, timeout: int = 180):
    """
    Wrapper que lanza _yt_stream_worker en un Process y aplica timeout.
    Si worker no responde en `timeout` segundos, se termina y se lanza excepción.
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_yt_stream_worker, args=(q, query, outdir, prefer_webm, bitrate, verbose))
    p.start()
    p.join(timeout)
    if p.is_alive():
        # timed out -> kill and raise
        p.terminate()
        p.join()
        # try to cleanup any partial output file by safe pattern (best-effort)
        raise RuntimeError(f"Download/convert timed out after {timeout}s for query: {query}")
    # get result from queue
    if q.empty():
        raise RuntimeError("No result from worker (it may have failed silently).")
    status, payload = q.get()
    if status == "ok":
        return payload
    else:
        # payload is (err_str, traceback)
        err_str, tb = payload
        raise RuntimeError(f"Worker error: {err_str}\n{tb}")


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
    parser.add_argument('--failures', default='download_failures.json', help='Archivo JSON donde se registran descargas fallidas')
    parser.add_argument('--limit', type=int, default=None, help='Máximo de canciones a descargar desde el rango seleccionado')
    parser.add_argument('--sleep', type=float, default=5.0, help='Segundos a esperar entre descargas para no sobrecargar servicios')

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
            result = yt_search_and_stream_convert_to_mp3(query, args.outdir, prefer_webm=True, verbose=False)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error buscando/descargando en YouTube para '{query}': {e}")
            # Save failure to failures file for retry later
            failure_entry = {
                'index_in_csv': idx + 1,
                'track_uri': track_uri,
                'query': query,
                'spotify': track_info,
                'error': str(e),
                'traceback': tb,
                'timestamp': time.time()
            }
            try:
                save_metadata_entry(args.failures, failure_entry)
            except Exception as se:
                print(f"No se pudo registrar failure en {args.failures}: {se}")
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