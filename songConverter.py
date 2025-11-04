import os
import shutil
import glob
import numpy as np
import pretty_midi
from pydub import AudioSegment
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# --- 0. Definición de Parámetros ---

# Parámetros de segmentación (en milisegundos)
DURACION_SEGMENTO_MS = 10 * 1000  # 10 segundos
PASO_MS = 5 * 1000               # 5 segundos (genera 5s de superposición)

# Directorios de salida
OUTPUT_DIR_AUDIO = os.path.join("resultados", "audio_fragments")
OUTPUT_DIR_MIDI = os.path.join("resultados", "midi_output")

def crear_vector_caracteristicas(midi_objeto: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """
    Toma un objeto PrettyMIDI y crea un vector de características (fingerprint).
    Devuelve un vector normalizado de 12 elementos o None si no hay notas.
    """
    if len(midi_objeto.instruments) == 0:
        print(" -> MIDI vacío, no se encontraron notas.")
        return None
    try:
        vector_chroma = midi_objeto.get_chroma(fs=5)
        vector_promedio = np.mean(vector_chroma, axis=1)
        norma = np.linalg.norm(vector_promedio)
        if norma > 0:
            return vector_promedio / norma
        else:
            return vector_promedio
    except Exception as e:
        print(f" -> Error al crear vector chroma: {e}")
        return None

# --- Funciones modulares ---

def segmentar_audio_file(ruta_audio: str, dir_salida: str, duracion_ms: int = DURACION_SEGMENTO_MS, paso_ms: int = PASO_MS, overwrite: bool = False, verbose: bool = True) -> list[str]:
    """
    Segmenta el archivo de audio en WAVs superpuestos y devuelve la lista de rutas creadas.
    Si los segmentos ya existen (y overwrite=False), los reusa y evita recrearlos.
    Crea `dir_salida` si no existe.
    """
    os.makedirs(dir_salida, exist_ok=True)
    rutas = []
    try:
        cancion = AudioSegment.from_file(ruta_audio)
    except Exception as e:
        if verbose: print(f"Error al cargar el archivo de audio {ruta_audio}: {e}")
        return []
    # calcular número esperado de segmentos
    if len(cancion) < duracion_ms:
        if verbose: print(f"Audio demasiado corto ({len(cancion)}ms < {duracion_ms}ms). No se generaron segmentos.")
        return []
    indices = list(range(0, len(cancion) - duracion_ms + 1, paso_ms))
    expected_names = [os.path.join(dir_salida, f"segmento_{i:03d}.wav") for i in range(len(indices))]
    if not overwrite and all(os.path.exists(p) for p in expected_names):
        if verbose: print(f"  -> Encontrados {len(expected_names)} segmentos existentes en '{dir_salida}', saltando creación.")
        return expected_names
    # crear segmentos (solo si no existen o overwrite=True)
    created = []
    for i, inicio_ms in enumerate(indices):
        nombre = os.path.join(dir_salida, f"segmento_{i:03d}.wav")
        if not overwrite and os.path.exists(nombre):
            created.append(nombre)
            continue
        fin_ms = inicio_ms + duracion_ms
        segmento = cancion[inicio_ms:fin_ms]
        try:
            segmento.export(nombre, format="wav")
            created.append(nombre)
        except Exception as e:
            if verbose: print(f"  -> Error exportando segmento {nombre}: {e}")
    if verbose: print(f"  -> Se crearon/obtenidos {len(created)} segmentos en '{dir_salida}'")
    return created

def obtener_ruta_midi_desde_segmento(ruta_segmento: str, dir_midi: str) -> str:
    """
    Dado un segmento WAV, devuelve la ruta del MIDI que genera basic-pitch.
    """
    base = os.path.splitext(os.path.basename(ruta_segmento))[0]
    nombre_midi = f"{base}_basic_pitch.mid"
    return os.path.join(dir_midi, nombre_midi)

def convertir_segmentos_a_midi(rutas_segmentos: list[str], dir_midi: str, model_path: str = ICASSP_2022_MODEL_PATH, overwrite: bool = False, verbose: bool = True) -> None:
    """
    Llama a basic-pitch para convertir WAVs a MIDI dentro de `dir_midi`.
    Solo procesa los segmentos cuyo MIDI destino no exista a menos que overwrite=True.
    """
    os.makedirs(dir_midi, exist_ok=True)
    if not rutas_segmentos:
        if verbose: print("No hay segmentos para convertir a MIDI.")
        return
    # filtrar segmentos que ya tienen MIDI generado
    pendientes = []
    for r in rutas_segmentos:
        midi_dest = obtener_ruta_midi_desde_segmento(r, dir_midi)
        if overwrite or not os.path.exists(midi_dest):
            pendientes.append(r)
    if not pendientes:
        if verbose: print("  -> No hay segmentos nuevos para convertir (todos los MIDIs ya existen).")
        return
    if verbose: print(f"  -> Ejecutando basic-pitch sobre {len(pendientes)} segmentos (esto puede tardar)...")
    try:
        predict_and_save(
            pendientes,
            dir_midi,
            True,   # save_midi
            False,  # sonify_midi
            False,  # save_model_outputs
            False,  # save_notes
            model_path, # ruta del modelo
        )
    except Exception as e:
        if verbose: print(f"Error al ejecutar basic-pitch: {e}")

def extraer_vectores_de_midis(rutas_midis: list[str], verbose: bool = True) -> list[np.ndarray]:
    """
    Dada una lista de rutas a archivos MIDI, intenta extraer el vector chroma normalizado
    de cada uno y devuelve la lista de vectores válidos.
    """
    vectores = []
    for ruta in rutas_midis:
        if not os.path.exists(ruta):
            if verbose: print(f"  -> Advertencia: no existe el MIDI {ruta}")
            continue
        try:
            midi_obj = pretty_midi.PrettyMIDI(ruta)
            vec = crear_vector_caracteristicas(midi_obj)
            if vec is not None:
                vectores.append(vec)
                if verbose: print(f"  -> Vector extraído de {os.path.basename(ruta)} (len={vec.shape})")
        except Exception as e:
            if verbose: print(f"  -> Error leyendo {ruta}: {e}")
    return vectores

def procesar_cancion_completa(
    ruta_archivo_audio: str,
    dir_audio: str = OUTPUT_DIR_AUDIO,
    dir_midi: str = OUTPUT_DIR_MIDI,
    limpiar: bool = False, 
    duracion_ms: int = DURACION_SEGMENTO_MS,
    paso_ms: int = PASO_MS,
    model_path: str = ICASSP_2022_MODEL_PATH,
    ) -> list[np.ndarray]:


    print(f"--- Iniciando procesamiento para: {ruta_archivo_audio} ---")

    # Limpiamos los directorios ANTES de empezar para evitar contaminación
    try:
        if os.path.exists(dir_audio):
            shutil.rmtree(dir_audio)
        if os.path.exists(dir_midi):
            shutil.rmtree(dir_midi)
        print("Directorios de salida previos eliminados.")
    except Exception as e:
        print(f"Advertencia: No se pudieron limpiar los directorios: {e}")
    
    # 1) Segmentar audio
    rutas_segmentos = segmentar_audio_file(ruta_archivo_audio, dir_audio, duracion_ms, paso_ms)
    
    if not rutas_segmentos:
        return []
    
    # 2) Convertir segmentos a MIDI
    convertir_segmentos_a_midi(rutas_segmentos, dir_midi, model_path)
    
    # 3) Construir rutas de MIDIs y extraer vectores
    rutas_midis = [obtener_ruta_midi_desde_segmento(r, dir_midi) for r in rutas_segmentos]
    vectores = extraer_vectores_de_midis(rutas_midis)
    
    print(f"--- Procesamiento finalizado. Total de vectores generados: {len(vectores)} ---")
    
    # 4) Limpieza opcional
    try:
        if limpiar:
            print("Opción 'limpiar' activada: eliminando carpetas de salida...")
            if os.path.exists(dir_audio):
                shutil.rmtree(dir_audio)
            if os.path.exists(dir_midi):
                shutil.rmtree(dir_midi)
            print("Limpieza completada.")
        else:
            print(f"Las salidas se han conservado en:\n  - {os.path.abspath(dir_audio)}\n  - {os.path.abspath(dir_midi)}")
    except Exception as e:
        print(f"Error durante la limpieza/retención de salidas: {e}")
    
    return vectores

if __name__ == "__main__":
    ruta_de_mi_cancion = "josephine.mp3"  # Yo tengo la cancion en la misma carpeta pero ahi le cambian la ruta
    if not os.path.exists(ruta_de_mi_cancion):
        print("Error: El archivo de ejemplo no existe.")
    else:
        vectores = procesar_cancion_completa(
            ruta_de_mi_cancion,
            dir_audio=os.path.join("resultados", "audio_fragments"),
            dir_midi=os.path.join("resultados", "midi_output"),
            limpiar=False,
        )
        if vectores:
            print("\nResumen de vectores generados:")
            for i, vec in enumerate(vectores[:5]):
                vec_str = ", ".join([f"{x:.4f}" for x in vec])
                print(f"Vector {i}: [{vec_str}]")
            if len(vectores) > 5:
                print(f"... y {len(vectores) - 5} vectores más.")