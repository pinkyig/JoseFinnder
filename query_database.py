import os
import sys
import json
import time
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

# Importaciones de Machine Learning y Procesamiento
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from sklearn.preprocessing import StandardScaler
from dtaidistance.dtw_ndim import distance as dtw_ndim_distance

# Importaciones de nuestro proyecto
from db_connector import get_music_collection
from songConverter import crear_vector_resumen_enriquecido, crear_secuencia_de_caracteristicas

# --- Constantes ---
DURACION_GRABACION_SEGUNDOS = 10
NOMBRE_ARCHIVO_GRABADO = "query_grabada.mp3"

# --- Conexión a la Base de Datos ---
collection = get_music_collection()


# --- Funciones Auxiliares para Re-Ranking ---

def crear_ngrams_de_secuencia(secuencia: list[list], n: int = 5) -> set[tuple]:
    """
    Convierte una secuencia de características en un conjunto de n-grams.
    Usa solo el contorno de tono (el primer elemento) para mayor robustez.
    """
    contorno_tono = [punto[0] for punto in secuencia]
    if len(contorno_tono) < n:
        return set()
    return {tuple(contorno_tono[i:i+n]) for i in range(len(contorno_tono) - n + 1)}

def lcs_length(seq1: list, seq2: list) -> int:
    """Calcula la longitud de la Subsecuencia Común Más Larga (LCS)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


# --- Tubería de Procesamiento y Búsqueda ---

def grabar_audio_desde_microfono(duracion_segundos: int, nombre_archivo_salida: str) -> str | None:
    """Activa el micrófono, graba audio y lo guarda como un archivo MP3."""
    SAMPLE_RATE = 44100
    CHANNELS = 1
    try:
        print(f"\n--- Preparando para grabar ({duracion_segundos}s) ---")
        for i in range(3, 0, -1):
            print(f"Comenzando en {i}...")
            time.sleep(1)
        print("\n¡GRABANDO AHORA! Canta o toca tu melodía...")
        grabacion_np = sd.rec(int(duracion_segundos * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        print("¡Grabación finalizada!")
        grabacion_int16 = np.int16(grabacion_np * 32767)
        audio_segment = AudioSegment(grabacion_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=grabacion_int16.dtype.itemsize, channels=CHANNELS)
        audio_segment.export(nombre_archivo_salida, format="mp3")
        print(f"Audio de consulta guardado como '{nombre_archivo_salida}'")
        return nombre_archivo_salida
    except Exception as e:
        print(f"\nError durante la grabación de audio: {e}")
        return None

def procesar_audio_query(ruta_audio_query: str) -> dict | None:
    """Procesa un archivo de audio de consulta para extraer sus características en memoria."""
    print(f" -> Procesando audio de consulta: {ruta_audio_query}")
    try:
        print(" -> Ejecutando basic-pitch...")
        model_output, midi_data, note_events = predict(ruta_audio_query, ICASSP_2022_MODEL_PATH)
        if not midi_data.instruments or not midi_data.instruments[0].notes:
            print("Error: No se detectaron notas en el audio de consulta.")
            return None
        print(" -> Creando vectores de características para la consulta...")
        vector_resumen = crear_vector_resumen_enriquecido(midi_data)
        secuencia = crear_secuencia_de_caracteristicas(midi_data)
        if vector_resumen is not None and secuencia:
            print(" -> Vectores de consulta generados exitosamente.")
            return {"vector_resumen": vector_resumen.tolist(), "secuencia": secuencia}
        else:
            print("Error: No se pudo generar un vector válido para la consulta.")
            return None
    except Exception as e:
        print(f"Error fatal procesando el audio de consulta: {e}")
        return None

def buscar_melodia_similar(ruta_audio_query: str, metodo_reranking: str = "ensemble", top_k: int = 200):
    """
    Implementa la búsqueda en dos etapas, con métodos de re-ranking modulares.
    El método "ensemble" combina los scores de los otros tres.
    """
    if collection is None:
        print("No se pudo obtener la colección. Abortando búsqueda.")
        return

    # --- VALIDACIÓN DEL MÉTODO ---
    metodo_reranking = metodo_reranking.lower()
    metodos_validos = ["ngram", "dtw", "lcs", "ensemble"]
    if metodo_reranking not in metodos_validos:
        print(f"Error: Método de re-ranking '{metodo_reranking}' no es válido. Use uno de: {metodos_validos}")
        return

    print(f"\n--- Iniciando búsqueda para: {ruta_audio_query} (Método: {metodo_reranking.upper()}) ---")
    
    # --- ETAPA 0: PROCESAR EL AUDIO DE LA CONSULTA ---
    query_data = procesar_audio_query(ruta_audio_query)
    if not query_data:
        return

    # --- ETAPA 1: PRE-FILTRADO CON CHROMADB ---
    print(f"\nBuscando los {top_k} candidatos más cercanos en ChromaDB...")
    resultados_db = collection.query(
        query_embeddings=[query_data["vector_resumen"]],
        n_results=top_k,
        include=["metadatas"]
    )
    
    # ====> ¡AQUÍ SE DEFINEN LAS VARIABLES QUE FALTABAN! <====
    ids_candidatos = resultados_db['ids'][0]
    metadatas_candidatos = resultados_db['metadatas'][0]

    # --- ETAPA 2: RE-RANKING ---
    print(f"Re-rankeando {len(ids_candidatos)} candidatos usando {metodo_reranking.upper()}...")
    
    resultados_brutos = []
    
    # Preparar los datos de la consulta una sola vez para cada método
    # N-Gram & LCS
    query_contorno = [p[0] for p in query_data["secuencia"]]
    ngrams_query = crear_ngrams_de_secuencia(query_data["secuencia"], n=5)
    # DTW
    query_sec_np = np.array(query_data["secuencia"], dtype=np.double)
    scaler = StandardScaler().fit(query_sec_np) if query_sec_np.size > 0 else None
    query_sec_scaled = scaler.transform(query_sec_np) if scaler else np.array([])

    # Calcular los scores para cada candidato
    for meta in metadatas_candidatos:
        candidato_secuencia = json.loads(meta["secuencia_json"])
        
        # --- Calcular Score N-Gram ---
        ngrams_candidato = crear_ngrams_de_secuencia(candidato_secuencia, n=5)
        interseccion = len(ngrams_query.intersection(ngrams_candidato))
        union = len(ngrams_query.union(ngrams_candidato))
        meta['score_jaccard'] = interseccion / union if union > 0 else 0.0

        # --- Calcular Score LCS ---
        candidato_contorno = [p[0] for p in candidato_secuencia]
        longitud_lcs = lcs_length(query_contorno, candidato_contorno)
        meta['score_lcs'] = longitud_lcs / len(query_contorno) if len(query_contorno) > 0 else 0.0

        # --- Calcular Score DTW ---
        candidato_sec_np = np.array(candidato_secuencia, dtype=np.double)
        if scaler and candidato_sec_np.size > 0:
            candidato_sec_scaled = scaler.transform(candidato_sec_np)
            distancia = dtw_ndim_distance(query_sec_scaled, candidato_sec_scaled)
            meta['distancia_dtw'] = distancia / (len(query_sec_scaled) + len(candidato_sec_scaled))
        else:
            meta['distancia_dtw'] = float('inf')
        
        resultados_brutos.append(meta)

    # --- ETAPA 3: ORDENAR Y MOSTRAR RESULTADOS ---
    sort_key = ""
    reverse_sort = True
    
    if metodo_reranking == "ngram":
        sort_key, reverse_sort = "score_jaccard", True
    elif metodo_reranking == "lcs":
        sort_key, reverse_sort = "score_lcs", True
    elif metodo_reranking == "dtw":
        sort_key, reverse_sort = "distancia_dtw", False
    elif metodo_reranking == "ensemble":
        # Calcular los ranks y el score final del ensamble
        resultados_brutos.sort(key=lambda x: x['score_jaccard'], reverse=True)
        for i, meta in enumerate(resultados_brutos): meta['rank_ngram'] = i

        resultados_brutos.sort(key=lambda x: x['score_lcs'], reverse=True)
        for i, meta in enumerate(resultados_brutos): meta['rank_lcs'] = i

        resultados_brutos.sort(key=lambda x: x['distancia_dtw'])
        for i, meta in enumerate(resultados_brutos): meta['rank_dtw'] = i

        for meta in resultados_brutos:
            meta['score_final'] = meta['rank_ngram'] + meta['rank_lcs'] + meta['rank_dtw']

        sort_key, reverse_sort = "score_final", False # Menor score final es mejor

    # Ordenar por la clave de ordenamiento seleccionada
    resultados_finales = sorted(resultados_brutos, key=lambda x: x[sort_key], reverse=reverse_sort)

    print(f"\n--- Resultados Finales (los 10 mejores usando {metodo_reranking.upper()}) ---")
    if metodo_reranking == "ensemble":
        for res in resultados_finales[:10]:
            print(f"Canción: {res['cancion']:<30} | Score Final: {res['score_final']:<5} "
                  f"(Ranks: Ngram={res['rank_ngram']}, LCS={res['rank_lcs']}, DTW={res['rank_dtw']})")
    else:
        for res in resultados_finales[:10]:
            print(f"Canción: {res['cancion']:<30} | Rango: {res.get('rango_tono', 'N/A'):<8} | Score: {res[sort_key]:.4f}")

    return resultados_finales


if __name__ == "__main__":
    if collection is None:
        sys.exit(1)

    # --- EJEMPLO DE USO ---
    # Puedes cambiar el método aquí para experimentar
    METODO_DE_BUSQUEDA = "ensemble" # Opciones: "ngram", "dtw", "lcs"
    
    ruta_query_audio = None
    if len(sys.argv) > 1:
        ruta_query_audio = sys.argv[1]
        print(f"Usando archivo de audio proporcionado: {ruta_query_audio}")
        if not os.path.exists(ruta_query_audio):
            print(f"Error: El archivo '{ruta_query_audio}' no existe.")
            sys.exit(1)
    else:
        ruta_query_audio = grabar_audio_desde_microfono(DURACION_GRABACION_SEGUNDOS, NOMBRE_ARCHIVO_GRABADO)

    if ruta_query_audio:
        buscar_melodia_similar(
            ruta_audio_query=ruta_query_audio,
            metodo_reranking=METODO_DE_BUSQUEDA,
            top_k=50
        )