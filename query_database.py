import os
import sys
import json
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from dtaidistance import dtw
from db_connector import get_music_collection
from songConverter import (
    crear_contorno_melodico, 
    crear_histograma_intervalos, 
    extraer_linea_melodica_superior
)

DURACION = 10
FILENAME = "query.mp3"

def grabar(duracion, salida):
    print(f"\n>>> GRABANDO {duracion}s... (Acércate al micro y tararea FUERTE) <<<")
    
    # 1. Record as float32
    grabacion = sd.rec(int(duracion * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    
    print(">>> PROCESANDO AUDIO... <<<")
    
    # 2. CRITICAL FIX: Remove the second dimension (Convert [[x],[x]] to [x,x])
    grabacion = grabacion.flatten()
    
    # 3. Check volume (Is the mic working?)
    max_val = np.max(np.abs(grabacion))
    if max_val < 0.01:
        print("⚠️ ADVERTENCIA: Audio muy bajo (casi silencio). Revisa tu configuración de micrófono en Windows.")
        return None
        
    # 4. Normalize (Boost Volume to Max without clipping)
    # This fixes "quiet humming" not being detected
    grabacion = grabacion / (max_val + 1e-6)
    
    # 5. Convert to Int16 carefully
    audio_int16 = (grabacion * 32767).astype(np.int16)
    
    # 6. Save
    seg = AudioSegment(
        audio_int16.tobytes(), 
        frame_rate=44100, sample_width=2, channels=1
    )
    seg.export(salida, format="mp3")
    print(f"Audio guardado en: {salida}")
    return salida
def procesar_query(ruta):
    try:
        # Umbrales bajos para detectar tarareo suave
        output, midi_data, _ = predict(ruta, ICASSP_2022_MODEL_PATH, onset_threshold=0.3, frame_threshold=0.2)
        
        # Limpieza Skyline
        midi_clean = extraer_linea_melodica_superior(midi_data)
        
        # Extraer features usando la nueva lógica "Key-Invariant"
        hist = crear_histograma_intervalos(midi_clean)
        contorno = crear_contorno_melodico(midi_clean, fs=10)
        
        return hist, contorno
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None, None

def buscar(ruta_audio):
    col = get_music_collection()
    if not col: return
    
    # 1. Extraer Features
    hist_query, contorno_query = procesar_query(ruta_audio)
    if not contorno_query:
        print("No se pudo extraer melodía. Intenta cantar más claro y con notas 'Ta-Ta-Ta'.")
        return

    # 2. Filtrado Previo (Vector de Intervalos)
    # Buscamos en DB canciones con estadísticas de intervalos similares
    res = col.query(
        query_embeddings=[hist_query],
        n_results=50, # Traer top 50 candidatos
        include=['metadatas']
    )
    
    # 3. Re-Ranking fino usando DTW sobre el Contorno
    candidatos = res['metadatas'][0]
    ids = res['ids'][0]
    scores = []
    
    q_seq = np.array(contorno_query, dtype=np.double)
    
    print(f"\nComparando {len(candidatos)} candidatos con DTW...")
    
    for i, meta in enumerate(candidatos):
        # Cargar contorno de la DB
        try:
            c_seq = np.array(json.loads(meta['contorno_json']), dtype=np.double)
            
            # Cálculo DTW (distancia de formas)
            # Usamos dtw simple 1D ya que el contorno es una línea
            d = dtw.distance(q_seq, c_seq)
            
            # Normalizamos por la longitud para ser justos
            score = d / (len(q_seq) + len(c_seq))
            
            scores.append({
                "cancion": meta['cancion'],
                "idx": meta['idx'],
                "score": score
            })
        except: pass
        
    # Ordenar (Menor score es mejor)
    scores.sort(key=lambda x: x['score'])
    
    print("\n=== RESULTADOS (MEJOR MATCH ARRIBA) ===")
    seen = set()
    for s in scores:
        if s['cancion'] not in seen:
            print(f"Canción: {s['cancion']:<25} | Error: {s['score']:.4f}")
            seen.add(s['cancion'])
            if len(seen) >= 5: break

if __name__ == "__main__":
    # Check if an argument was provided (e.g., python query_database.py my_song.mp3)
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        print(f"Usando archivo existente: {ruta}")
        
        # Verify file exists
        if not os.path.exists(ruta):
            print("Error: El archivo no existe.")
        else:
            buscar(ruta)
    else:
        # If no argument, record from microphone
        ruta = grabar(DURACION, FILENAME)
        if ruta:
            buscar(ruta)