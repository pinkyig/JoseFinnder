import os
import sys
import time
import json
import numpy as np
import sounddevice as sd
import librosa
from pydub import AudioSegment

# We use the Multi-Dimensional DTW now (12 dimensions for Chroma)
from dtaidistance import dtw_ndim

from db_connector import get_music_collection
from songConverter import generar_vector_chroma_cens

# --- CONFIGURATION ---
DURACION_GRABACION = 10
ARCHIVO_QUERY = "query_actual.mp3"

def grabar_audio_correctamente(duracion, nombre_archivo):
    # Standard recording logic with static fix
    SAMPLE_RATE = 44100
    print(f"\n--- PREPARANDO MICROFONO ({duracion}s) ---")
    time.sleep(1)
    print("3... 2... 1... CANTA!")
    
    try:
        audio = sd.rec(int(duracion * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten() # Static fix
        
        if np.max(np.abs(audio)) < 0.01:
            print("⚠️ Silencio detectado.")
            return None
            
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        seg = AudioSegment((audio * 32767).astype(np.int16).tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
        seg.export(nombre_archivo, format="mp3")
        return nombre_archivo
    except Exception as e:
        print(e)
        return None

def buscar(ruta_query):
    col = get_music_collection()
    if not col: return
    
    # 1. Generate Chroma Vector for Hum
    print(" -> Generando Chroma CENS del hum...")
    q_chroma = generar_vector_chroma_cens(ruta_query)
    
    if q_chroma is None: return
    
    # Convert to Numpy for DTW
    # Shape: (Time, 12)
    q_seq = np.array(q_chroma, dtype=np.double)
    
    all_data = col.get()
    scores = []
    
    print(f"Comparando contra {len(all_data['ids'])} fragmentos...")
    
    for i, meta in enumerate(all_data['metadatas']):
        try:
            # 2. Load Target Sequence
            t_seq = np.array(json.loads(meta['contorno_json']), dtype=np.double)
            
            if len(t_seq) < 10: continue

            # 3. Multidimensional DTW
            # Compares two matrices (TimeA x 12) vs (TimeB x 12)
            d = dtw_ndim.distance(q_seq, t_seq)
            
            # Normalize score
            score = d / (len(q_seq) + len(t_seq))
            scores.append((meta['cancion'], score))
        except Exception as e: 
            pass
        
    scores.sort(key=lambda x: x[1])
    
    print("\n=== TOP 5 ===")
    seen = set()
    for name, s in scores:
        if name not in seen:
            print(f"{name:<20} | {s:.4f}")
            seen.add(name)
        if len(seen) >= 5: break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]): buscar(sys.argv[1])
    else:
        f = grabar_audio_correctamente(DURACION_GRABACION, ARCHIVO_QUERY)
        if f: buscar(f)