import os
import glob
import json
import torch
import numpy as np
import demucs.separate
import librosa
from pydub import AudioSegment

# --- CONFIGURATION ---
DIR_RESULTADOS = "resultados"
DIR_STEMS = os.path.join(DIR_RESULTADOS, "1_stems_demucs")
DIR_FRAGMENTS = os.path.join(DIR_RESULTADOS, "2_audio_fragments")

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def separar_con_demucs(ruta_cancion):
    """
    Separates Vocals using Demucs on GPU.
    """
    song_name = os.path.splitext(os.path.basename(ruta_cancion))[0]
    expected_output = os.path.join(DIR_STEMS, "htdemucs", song_name, "vocals.wav")
    
    if os.path.exists(expected_output): 
        return expected_output

    print(f"\n[GPU] Separating vocals: {song_name}...")
    try:
        demucs.separate.main([
            "-n", "htdemucs", 
            "--two-stems", "vocals", 
            "-o", DIR_STEMS, 
            "-d", DEVICE, 
            ruta_cancion
        ])
        return expected_output
    except Exception as e:
        print(f"Demucs Error: {e}")
        return None

def segmentar_stem(ruta_stem, song_name):
    """
    Cuts stem into overlapping 12s segments.
    """
    song_frag_dir = os.path.join(DIR_FRAGMENTS, song_name)
    os.makedirs(song_frag_dir, exist_ok=True)
    try:
        audio = AudioSegment.from_file(ruta_stem).set_channels(1)
        rutas = []
        step = 4000 
        duration = 12000 
        
        if len(audio) < duration: return []

        for i in range(0, len(audio) - duration, step):
            out = os.path.join(song_frag_dir, f"seg_{i}.wav")
            if not os.path.exists(out): 
                audio[i:i+duration].export(out, format="wav")
            rutas.append(out)
        return rutas
    except: return []

def generar_vector_chroma_cens(ruta_audio):
    """
    Extracts Chroma CENS features (CQT-based).
    Output shape: (Time_Steps, 12).
    CENS is robust to dynamics and timbre differences (Hum vs Studio).
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(ruta_audio, sr=22050, mono=True)
        
        # 2. Extract Chroma CENS
        # hop_length=512 -> ~43 frames per second.
        # n_octaves=5 -> Covers typical vocal range
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512, n_octaves=5)
        
        # chroma shape is (12, Time). We transpose to (Time, 12) for DTW.
        chroma = chroma.T
        
        # 3. Downsample / Compression (Optional)
        # 43 fps is high detail. We can skip frames to speed up search without losing melody shape.
        # Take every 4th frame -> ~10 fps
        chroma_downsampled = chroma[::4]
        
        # Convert to list for JSON serialization
        return chroma_downsampled.tolist()
        
    except Exception as e:
        print(f"Chroma Error: {e}")
        return None

def procesar_cancion_completa(ruta_cancion, limpiar=False):
    for d in [DIR_STEMS, DIR_FRAGMENTS]: os.makedirs(d, exist_ok=True)
    song_name = os.path.splitext(os.path.basename(ruta_cancion))[0]
    
    # 1. Demucs
    ruta_stem = separar_con_demucs(ruta_cancion)
    if not ruta_stem: return []
    
    # 2. Fragment
    wavs = segmentar_stem(ruta_stem, song_name)
    fragmentos_db = []
    
    print(f"  -> Extracting CENS Chroma for {len(wavs)} segments...")
    
    for i, wav in enumerate(wavs):
        # 3. Generate Chroma Vector
        chroma_seq = generar_vector_chroma_cens(wav)
        
        if chroma_seq is not None and len(chroma_seq) > 20:
            meta = {
                "cancion": os.path.basename(ruta_cancion),
                "segmento_idx": i,
                "contorno_json": json.dumps(chroma_seq) 
            }
            fragmentos_db.append({
                "id": f"{song_name}_{i}",
                # Dummy embedding for ChromaDB, we depend on the Sequence search
                "vector_resumen": [0.0]*10, 
                "metadata": meta
            })

    return fragmentos_db