import argparse
import glob
import os
import subprocess
from typing import List

from tqdm import tqdm

DEFAULT_INDIR = "music"
DEFAULT_OUTDIR = "music_mp3"

def find_webm_files(indir: str) -> List[str]:
    patterns = [os.path.join(indir, "**", "*.webm"), os.path.join(indir, "**", "*.WebM"), os.path.join(indir, "**", "*.WEBM")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # deduplicate and sort
    files = sorted(list(dict.fromkeys(files)))
    return files


def convert_file(in_path: str, out_path: str, bitrate: str = "192k", overwrite: bool = False) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not overwrite:
        return True  # already exists, treat as success

    ffmpeg_cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        in_path,
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        out_path,
    ]
    try:
        # Run and hide ffmpeg output unless it fails
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            # failure
            print(f"Error converting {in_path} -> {out_path}: ffmpeg rc={proc.returncode}")
            # print stderr snippet
            stderr = proc.stderr.decode('utf-8', errors='ignore')
            print(stderr.splitlines()[-5:])
            return False
        return True
    except FileNotFoundError:
        print("ffmpeg no encontrado. Instala ffmpeg y asegúrate de que esté en PATH.")
        return False
    except Exception as e:
        print(f"Exception converting {in_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert all .webm files in a folder to .mp3 using ffmpeg")
    parser.add_argument('--indir', default=DEFAULT_INDIR, help='Directorio con archivos .webm (por defecto: %(default)s)')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='Directorio donde guardar .mp3 (por defecto: %(default)s). Si coincide con indir, los mp3 se crearán junto a los webm.')
    parser.add_argument('--bitrate', default='192k', help='Bitrate para mp3 (ej. 192k, 128k)')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescribir .mp3 existentes')
    parser.add_argument('--delete-source', action='store_true', help='Eliminar el .webm original después de convertir correctamente')
    parser.add_argument('--pattern', default='*.webm', help='Patrón de búsqueda (por defecto: *.webm)')

    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir

    files = find_webm_files(indir)
    if not files:
        print(f"No se encontraron archivos .webm en {indir}")
        return

    print(f"Encontrados {len(files)} archivos .webm. Convirtiendo a {outdir} con bitrate {args.bitrate}.")

    successes = 0
    failures = 0

    for in_path in tqdm(files, desc="Convirtiendo"):
        base = os.path.splitext(os.path.basename(in_path))[0]
        # Si outdir es el mismo que indir, crear filename base + .mp3 en el mismo directorio
        if os.path.abspath(outdir) == os.path.abspath(indir):
            out_path = os.path.join(os.path.dirname(in_path), f"{base}.mp3")
        else:
            out_path = os.path.join(outdir, f"{base}.mp3")

        ok = convert_file(in_path, out_path, bitrate=args.bitrate, overwrite=args.overwrite)
        if ok:
            successes += 1
            if args.delete_source:
                try:
                    os.remove(in_path)
                except Exception as e:
                    print(f"No se pudo eliminar fuente {in_path}: {e}")
        else:
            failures += 1

    print(f"\nConversión finalizada: éxitos={successes}, fallos={failures}.")
    if failures:
        print("Revisa los errores anteriores. Asegúrate de que ffmpeg esté instalado y que los archivos no estén corruptos.")


if __name__ == '__main__':
    main()
