import os
import shutil
import glob
import numpy as np
import pretty_midi
from pydub import AudioSegment
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import json
# --- 0. Definición de Parámetros ---

# Parámetros de segmentación (en milisegundos)
DURACION_SEGMENTO_MS = 10 * 1000  # 10 segundos
PASO_MS = 5 * 1000               # 5 segundos (genera 5s de superposición)


OUTPUT_DIR_AUDIO = os.path.join("resultados", "audio_fragments")
OUTPUT_DIR_MIDI = os.path.join("resultados", "midi_output")

def filtrar_midi_por_rango_de_tonos(
    midi_objeto_original: pretty_midi.PrettyMIDI, 
    tono_min: int = 0, 
    tono_max: int = 127
) -> pretty_midi.PrettyMIDI | None:
    """
    Crea un nuevo objeto PrettyMIDI que contiene solo las notas del original
    que se encuentran dentro del rango de tonos [tono_min, tono_max].

    Devuelve None si no se encuentra ninguna nota en ese rango.
    """
    midi_filtrado = pretty_midi.PrettyMIDI()

    for instrumento_original in midi_objeto_original.instruments:
        # No procesar baterías
        if instrumento_original.is_drum:
            continue
            
        # Crear un nuevo instrumento para el objeto MIDI filtrado
        instrumento_filtrado = pretty_midi.Instrument(
            program=instrumento_original.program,
            is_drum=instrumento_original.is_drum,
            name=instrumento_original.name
        )

        # Copiar solo las notas que están en el rango deseado
        for nota in instrumento_original.notes:
            if tono_min <= nota.pitch <= tono_max:
                instrumento_filtrado.notes.append(nota)
        
        # Añadir el nuevo instrumento solo si contiene notas
        if len(instrumento_filtrado.notes) > 0:
            midi_filtrado.instruments.append(instrumento_filtrado)

    # Devolver el objeto MIDI filtrado solo si contiene algún instrumento con notas
    if len(midi_filtrado.instruments) > 0:
        return midi_filtrado
    else:
        return None

def crear_vector_resumen_enriquecido(midi_objeto: pretty_midi.PrettyMIDI) -> np.ndarray | None:
    """
    Toma un objeto PrettyMIDI y crea un vector resumen basado ÚNICAMENTE en el chroma.
    Este método es más robusto a las diferencias de tempo y ritmo entre consulta y canción.
    """
    try:
        # Asegurarnos de que hay notas para procesar
        notas = [nota for inst in midi_objeto.instruments if not inst.is_drum for nota in inst.notes]
        if not notas:
            return None

        # 1. Extraer el vector chroma
        vector_chroma = midi_objeto.get_chroma(fs=5)
        vector_promedio_chroma = np.mean(vector_chroma, axis=1) # Vector de 12 dimensiones

        # 2. Normalizar el vector para que la similitud del coseno funcione
        norma = np.linalg.norm(vector_promedio_chroma)
        if norma > 0:
            return vector_promedio_chroma / norma
        else:
            return vector_promedio_chroma # Devuelve un vector de ceros si no hay notas
            
    except Exception as e:
        print(f" -> Error al crear vector chroma simple: {e}")
        return None
    
def crear_secuencia_de_caracteristicas(midi_objeto: pretty_midi.PrettyMIDI) -> list[list]:
    """
    Crea una secuencia de características quantizadas, capturando la magnitud del
    contorno de tono y ritmo de forma abstracta.
    """
    todas_las_notas = []
    for instrument in midi_objeto.instruments:
        if not instrument.is_drum:
            todas_las_notas.extend(instrument.notes)

    if len(todas_las_notas) < 2:
        return []

    notas_ordenadas = sorted(todas_las_notas, key=lambda n: n.start)
    
    secuencia_quantizada = []
    
    for i in range(1, len(notas_ordenadas)):
        nota_anterior = notas_ordenadas[i-1]
        nota_actual = notas_ordenadas[i]

        # --- 1. Quantización del Tono ---
        diferencia_tono = nota_actual.pitch - nota_anterior.pitch
        
        # Categorías: Salto Grande Arriba, Salto Arriba, Paso Arriba, Repetición, Paso Abajo, etc.
        q_tono = 0
        if diferencia_tono > 7:   q_tono = 3  # Salto grande arriba (> 5a justa)
        elif diferencia_tono > 2: q_tono = 2  # Salto arriba (3a menor a 5a justa)
        elif diferencia_tono > 0: q_tono = 1  # Paso arriba (1 o 2 semitonos)
        elif diferencia_tono < -7: q_tono = -3 # Salto grande abajo
        elif diferencia_tono < -2: q_tono = -2 # Salto abajo
        elif diferencia_tono < 0: q_tono = -1  # Paso abajo
        # si es 0, se queda como 0 (Repetición)

        # --- 2. Quantización del Ritmo (Duración Relativa) ---
        duracion_actual = nota_actual.end - nota_actual.start
        duracion_anterior = nota_anterior.end - nota_anterior.start
        
        q_ritmo = 0 # Duración similar
        if duracion_anterior > 0.01:
            ratio_duracion = duracion_actual / duracion_anterior
            if ratio_duracion > 1.8: q_ritmo = 1   # Nota actual es mucho más larga
            elif ratio_duracion < 0.5: q_ritmo = -1  # Nota actual es mucho más corta
        
        caracteristicas_nota = [float(q_tono), float(q_ritmo)]
        secuencia_quantizada.append(caracteristicas_nota)

    return secuencia_quantizada

def extraer_linea_melodica_superior(midi_objeto: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """
    Toma un objeto PrettyMIDI y devuelve uno nuevo que contiene solo la línea melódica superior.
    Fusiona todas las notas de todos los instrumentos y, en cada paso de tiempo,
    se queda únicamente con la nota de tono (pitch) más alto.
    """
    if not midi_objeto.instruments:
        return midi_objeto

    # 1. Juntar todas las notas de todos los instrumentos (no de percusión) en una sola lista
    todas_las_notas = []
    for inst in midi_objeto.instruments:
        if not inst.is_drum:
            todas_las_notas.extend(inst.notes)
    
    if not todas_las_notas:
        return midi_objeto

    # 2. Ordenar las notas por tiempo de inicio
    todas_las_notas.sort(key=lambda x: x.start)

    # 3. Iterar y construir la melodía, resolviendo solapamientos
    melodia_final = []
    if todas_las_notas:
        # Añadimos la primera nota para empezar
        melodia_final.append(todas_las_notas[0])

        for nota_actual in todas_las_notas[1:]:
            nota_anterior = melodia_final[-1]
            
            # Si la nota actual empieza antes de que la anterior termine (hay solapamiento)
            if nota_actual.start < nota_anterior.end:
                # Nos quedamos con la nota de mayor tono (pitch)
                if nota_actual.pitch > nota_anterior.pitch:
                    # La nota actual es más aguda. Recortamos la anterior para que no se solapen
                    # y añadimos la nueva.
                    if nota_anterior.start < nota_actual.start:
                         nota_anterior.end = nota_actual.start
                    else: # Si empiezan al mismo tiempo, la anterior se descarta
                         melodia_final.pop()
                    
                    melodia_final.append(nota_actual)
                # Si la nota actual es más grave, simplemente la ignoramos.
            else:
                # No hay solapamiento, simplemente añadimos la nota
                melodia_final.append(nota_actual)

    # 4. Crear un nuevo objeto MIDI con solo la melodía extraída
    midi_melodia = pretty_midi.PrettyMIDI()
    instrumento_melodia = pretty_midi.Instrument(program=0, name="Melodia Extraida")
    instrumento_melodia.notes = melodia_final
    midi_melodia.instruments.append(instrumento_melodia)
    
    return midi_melodia

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
            vec = crear_vector_resumen_enriquecido(midi_obj)
            if vec is not None:
                vectores.append(vec)
                if verbose: print(f"  -> Vector extraído de {os.path.basename(ruta)} (len={vec.shape})")
        except Exception as e:
            if verbose: print(f"  -> Error leyendo {ruta}: {e}")
    return vectores

# En songConverter.py, reemplaza tu función procesar_cancion_completa con esta:

def procesar_cancion_completa(
    ruta_archivo_audio: str,
    dir_audio: str = OUTPUT_DIR_AUDIO,
    dir_midi: str = OUTPUT_DIR_MIDI,
    limpiar: bool = False, 
    duracion_ms: int = DURACION_SEGMENTO_MS,
    paso_ms: int = PASO_MS,
    model_path: str = ICASSP_2022_MODEL_PATH,
) -> list[dict]:

    print(f"--- Iniciando procesamiento para: {ruta_archivo_audio} ---")
    
    # ... (limpieza y segmentación sin cambios) ...
    try:
        if os.path.exists(dir_audio): shutil.rmtree(dir_audio)
        if os.path.exists(dir_midi): shutil.rmtree(dir_midi)
        print("Directorios de salida previos eliminados.")
    except Exception as e:
        print(f"Advertencia: No se pudieron limpiar los directorios: {e}")
    
    rutas_segmentos = segmentar_audio_file(ruta_archivo_audio, dir_audio, duracion_ms, paso_ms)
    if not rutas_segmentos:
        return []
    
    convertir_segmentos_a_midi(rutas_segmentos, dir_midi, model_path)
    
    pros_fragment = []
    song_name = os.path.basename(ruta_archivo_audio)

    RANGOS_DE_TONO = {
        "graves": (0, 47),
        "medio":  (48, 83),
        "altas":  (84, 127)
    }

    for i, ruta_segmento in enumerate(rutas_segmentos):
        ruta_midi = obtener_ruta_midi_desde_segmento(ruta_segmento, dir_midi)
        if not os.path.exists(ruta_midi):
            continue

        print(f"\n--- Procesando Segmento {i} de {song_name} ---")
        try:
            midi_obj_completo = pretty_midi.PrettyMIDI(ruta_midi)
            notas_originales = sum(len(inst.notes) for inst in midi_obj_completo.instruments)
            print(f"  Paso 1: MIDI original tiene {notas_originales} notas.")

            for nombre_rango, (tono_min, tono_max) in RANGOS_DE_TONO.items():
                print(f"    -> Rango '{nombre_rango.upper()}'")
                
                midi_obj_filtrado = filtrar_midi_por_rango_de_tonos(midi_obj_completo, tono_min, tono_max)
                
                if midi_obj_filtrado:
                    notas_filtradas = sum(len(inst.notes) for inst in midi_obj_filtrado.instruments)
                    print(f"      Paso 2: MIDI filtrado por rango tiene {notas_filtradas} notas.")

                    midi_obj_melodia = extraer_linea_melodica_superior(midi_obj_filtrado)
                    notas_melodia = sum(len(inst.notes) for inst in midi_obj_melodia.instruments)
                    print(f"      Paso 3: Melodía extraída tiene {notas_melodia} notas.")

                    # Comprobación crucial: ¿tenemos suficientes notas para crear una secuencia?
                    if notas_melodia < 2:
                        print("      FALLO: Menos de 2 notas. No se puede generar secuencia. Saltando.")
                        continue # Pasa al siguiente rango

                    vector_sequence = crear_secuencia_de_caracteristicas(midi_obj_melodia)
                    vector_resumen = crear_vector_resumen_enriquecido(midi_obj_melodia)
                    print(f"      Paso 4: Secuencia generada con {len(vector_sequence)} puntos.")

                    if vector_resumen is not None and vector_sequence:
                        fragmento_id = f"{song_name}_{i}_{nombre_rango}"
                        metadata = { "cancion": song_name, "segmento_idx": i, "inicio_segundos": (i * paso_ms) / 1000.0, "rango_tono": nombre_rango, "secuencia_json": json.dumps(vector_sequence) }
                        pros_fragment.append({ "id": fragmento_id, "vector_resumen": vector_resumen, "metadata": metadata })
                        print(f"      ÉXITO: Fragmento para rango '{nombre_rango.upper()}' añadido.")
                    else:
                        print(f"      FALLO: Vector resumen o secuencia vacía después del procesamiento.")
                else:
                    print(f"      Paso 2: No se encontraron notas en este rango.")
        except Exception as e:
            print(f" -> ERROR INESPERADO procesando MIDI {ruta_midi}: {e}")
    
    # ... (limpieza final sin cambios) ...
    # ...
    return pros_fragment

if __name__ == "__main__":
    ruta_de_mi_cancion = "Humoresque.mp3" # Asegúrate de que este archivo exista
    if not os.path.exists(ruta_de_mi_cancion):
        print(f"Error: El archivo de ejemplo '{ruta_de_mi_cancion}' no existe.")
    else:
        resultados = procesar_cancion_completa(
            ruta_de_mi_cancion, limpiar=False,
        )
        if resultados:
            print(f"\n--- Resumen de {len(resultados)} fragmentos generados ---")
            for i, res_dict in enumerate(resultados[:5]):
                secuencia = json.loads(res_dict["metadata"]["secuencia_json"])
                num_notas = len(secuencia)
                vector_resumen = res_dict["vector_resumen"]
                
                print(f"Fragmento {i} (ID: {res_dict['id']}):")
                print(f"  - Vector Resumen (dims={len(vector_resumen)}): [{vector_resumen[0]:.2f}, ...]")
                print(f"  - Secuencia de Notas: {num_notas} notas encontradas.")