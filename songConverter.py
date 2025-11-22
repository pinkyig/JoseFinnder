import os
import shutil
import glob
import json
import numpy as np
import pretty_midi
from pydub import AudioSegment
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
from scipy.ndimage import median_filter

# --- CONSTANTS ---
DURACION_SEGMENTO_MS = 12 * 1000
PASO_MS = 4 * 1000
OUTPUT_DIR_AUDIO = os.path.join("resultados", "audio_fragments")
OUTPUT_DIR_MIDI = os.path.join("resultados", "midi_output")

# --- 1. FAST FREQUENCY FILTERING ---
def segmentar_audio_filtrado(ruta_audio, dir_salida, duracion_ms, paso_ms):
    """
    Segment audio with AGGRESSIVE frequency filtering.
    We stack the filters 3 times to act like a steep 'brick wall' EQ.
    Range: 250Hz (removes all bass) to 2500Hz (removes all cymbals).
    """
    os.makedirs(dir_salida, exist_ok=True)
    try:
        audio = AudioSegment.from_file(ruta_audio)
        
        # 1. Convert to Mono
        audio = audio.set_channels(1)
        
        # 2. THE "BRICK WALL" FILTER STRATEGY
        # Pydub filters are gentle (6dB/oct). We apply them 3 times 
        # to create a sharp 18dB/oct cut.
        
        # Remove Bass (Keep above 250Hz)
        # Normal Male fundamental is ~100Hz, but melody tracks clearer at 250+
        audio = audio.high_pass_filter(250).high_pass_filter(250).high_pass_filter(250)
        
        # Remove Treble/Hiss (Keep below 2500Hz)
        audio = audio.low_pass_filter(2500).low_pass_filter(2500).low_pass_filter(2500)
        
        # 3. Normalize after filtering so the remaining melody is loud
        audio = audio.normalize()

    except Exception as e:
        print(f"Error reading audio {ruta_audio}: {e}")
        return []
    
    rutas = []
    for i in range(0, len(audio) - duracion_ms, paso_ms):
        out_name = os.path.join(dir_salida, f"seg_{i}.wav")
        if not os.path.exists(out_name):
            audio[i:i+duracion_ms].export(out_name, format="wav")
        rutas.append(out_name)
    return rutas

# --- 2. VECTOR LOGIC (Derivative Contour) ---
# We stick with the Derivative Contour (Changes) as it handles Key Transposition best.

def extraer_linea_melodica_superior(midi_obj):
    """Keeps only the highest pitch note at any time step."""
    if not midi_obj.instruments: return midi_obj
    
    midi_melodia = pretty_midi.PrettyMIDI()
    inst_melodia = pretty_midi.Instrument(program=0, name="Skyline")
    
    # Gather all notes from non-drum instruments
    all_notes = []
    for inst in midi_obj.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)
            
    if not all_notes: return midi_obj
    
    # Sort by start time
    all_notes.sort(key=lambda x: x.start)
    
    # Skyline Algorithm
    skyline_notes = []
    if all_notes:
        skyline_notes.append(all_notes[0])
        for curr in all_notes[1:]:
            prev = skyline_notes[-1]
            
            # If overlap
            if curr.start < prev.end:
                # If current is higher pitch, it wins (assuming melody is on top)
                if curr.pitch > prev.pitch:
                    # Trim previous
                    if prev.start < curr.start:
                        prev.end = curr.start
                    else:
                        skyline_notes.pop() # Replaced completely
                    skyline_notes.append(curr)
                # Else ignore current (it's accompaniment)
            else:
                skyline_notes.append(curr)
                
    inst_melodia.notes = skyline_notes
    midi_melodia.instruments.append(inst_melodia)
    return midi_melodia

def crear_histograma_intervalos(midi_obj):
    """Generates 16-dim vector for the Database Index (ChromaDB)."""
    notes = [n for i in midi_obj.instruments for n in i.notes]
    if len(notes) < 3: return np.zeros(16).tolist()
    notes.sort(key=lambda x: x.start)
    
    diffs = []
    for i in range(1, len(notes)):
        d = notes[i].pitch - notes[i-1].pitch
        if -8 <= d <= 8: diffs.append(d)
        
    hist, _ = np.histogram(diffs, bins=range(-9, 9), density=True)
    return hist.tolist()

def crear_contorno_melodico(midi_obj, fs=10):
    """Generates the Derivative Sequence for DTW comparison."""
    try:
        piano_roll = midi_obj.get_piano_roll(fs=fs)
        pitches = []
        
        for t in range(piano_roll.shape[1]):
            active = np.where(piano_roll[:, t] > 0)[0]
            if len(active) > 0:
                pitches.append(float(active[-1]))
                
        if len(pitches) < 10: return []
        
        # Median filter to remove random MIDI jitter
        smooth = median_filter(pitches, size=3)
        
        # Derivative: Track movement, not absolute pitch
        diffs = np.diff(smooth)
        
        return diffs.tolist()
    except: return []

# --- 3. HELPERS ---
def convertir_a_midi_lote(wavs, dir_midi, model):
    os.makedirs(dir_midi, exist_ok=True)
    # Logic to skip existing midis to save time
    todo = []
    for w in wavs:
        midi_name = os.path.basename(w).replace(".wav", "_basic_pitch.mid")
        if not os.path.exists(os.path.join(dir_midi, midi_name)):
            todo.append(w)
            
    if todo:
        # Suppress logs if you want it cleaner
        predict_and_save(todo, dir_midi, True, False, False, False, model)

def obtener_ruta_midi(wav_path, dir_midi):
    return os.path.join(dir_midi, os.path.basename(wav_path).replace(".wav", "_basic_pitch.mid"))

# --- 4. MAIN ---
def procesar_cancion_completa(
    ruta_audio, 
    dir_audio_base=OUTPUT_DIR_AUDIO, 
    dir_midi_base=OUTPUT_DIR_MIDI, 
    limpiar=False, 
    model_path=ICASSP_2022_MODEL_PATH
):
    nombre_cancion = os.path.splitext(os.path.basename(ruta_audio))[0]
    
    # Create song-specific folders to prevent overwriting
    song_audio_dir = os.path.join(dir_audio_base, nombre_cancion)
    song_midi_dir = os.path.join(dir_midi_base, nombre_cancion)
    
    # We don't clean here anymore (controlled by main script), 
    # but we ensure folders exist.
    
    # 1. SEGMENT + FILTER (Fast)
    wavs = segmentar_audio_filtrado(ruta_audio, song_audio_dir, DURACION_SEGMENTO_MS, PASO_MS)
    
    if not wavs: return []

    # 2. CONVERT
    convertir_a_midi_lote(wavs, song_midi_dir, model_path)
    
    fragmentos = []
    
    # 3. PROCESS
    for i, wav in enumerate(wavs):
        mid = obtener_ruta_midi(wav, song_midi_dir)
        if not os.path.exists(mid): continue
        
        try:
            midi_obj = pretty_midi.PrettyMIDI(mid)
            
            # Isolate Melody using Skyline
            midi_melodia = extraer_linea_melodica_superior(midi_obj)
            
            # Check if we have enough notes
            if sum(len(inst.notes) for inst in midi_melodia.instruments) < 5:
                continue
            
            vec = crear_histograma_intervalos(midi_melodia)
            cont = crear_contorno_melodico(midi_melodia, fs=10)
            
            if len(cont) > 8:
                meta = {
                    "cancion": os.path.basename(ruta_audio),
                    "idx": i,
                    "contorno_json": json.dumps(cont)
                }
                fragmentos.append({
                    "id": f"{nombre_cancion}_{i}",
                    "vector_resumen": vec,
                    "metadata": meta
                })
        except Exception as e:
            print(f"Skipping segment {i}: {e}")
            
    return fragmentos