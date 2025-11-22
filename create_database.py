import os
import glob
import chromadb
import shutil # Import needed for manual cleaning
from db_connector import create_and_get_music_collection
from songConverter import procesar_cancion_completa, OUTPUT_DIR_AUDIO, OUTPUT_DIR_MIDI

CANCIONES_DIR = "music"
DB_PATH = "music_database"

def poblar_base_de_datos():
    # --- FIX 1: Clean directories ONCE before the loop starts ---
    print("--- Cleaning previous results ---")
    if os.path.exists(OUTPUT_DIR_AUDIO):
        shutil.rmtree(OUTPUT_DIR_AUDIO)
    if os.path.exists(OUTPUT_DIR_MIDI):
        shutil.rmtree(OUTPUT_DIR_MIDI)

    # Re-create empty parent directories
    os.makedirs(OUTPUT_DIR_AUDIO, exist_ok=True)
    os.makedirs(OUTPUT_DIR_MIDI, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = create_and_get_music_collection()
    
    archivos_audio = glob.glob(os.path.join(CANCIONES_DIR, "*.mp3")) + glob.glob(os.path.join(CANCIONES_DIR, "*.wav"))
    print(f"Found {len(archivos_audio)} songs.")

    for ruta_cancion in archivos_audio:
        nombre_cancion = os.path.basename(ruta_cancion)

        # Check if exists in DB
        existing = collection.get(where={"cancion": nombre_cancion}, limit=1)
        if existing["ids"]:
            print(f"Skipping '{nombre_cancion}' (Already in DB)")
            continue

        # --- FIX 2: Pass limpiar=False so we don't delete previous songs ---
        fragmentos = procesar_cancion_completa(ruta_cancion, limpiar=False)

        if not fragmentos: continue
        
        # Fix for the "AttributeError" you saw earlier
        valid_fragments = [f for f in fragmentos if f.get("vector_resumen") is not None]
        
        if not valid_fragments: continue

        ids = [f['id'] for f in valid_fragments]
        # Handle numpy array vs list conversion safely
        vectores = [f['vector_resumen'] if isinstance(f['vector_resumen'], list) else f['vector_resumen'].tolist() for f in valid_fragments]
        metadatas = [f['metadata'] for f in valid_fragments]

        try:
            collection.add(ids=ids, embeddings=vectores, metadatas=metadatas)
            print(f" -> Added {len(ids)} fragments for {nombre_cancion}")
        except Exception as e:
            print(f"Error DB: {e}")
        
    print("Done.")

if __name__ == "__main__":
     poblar_base_de_datos()