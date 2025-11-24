import os
import glob
import chromadb
import shutil
import time

print("--- [DEBUG] Importing modules... ---")
from db_connector import create_and_get_music_collection
from songConverter import procesar_cancion_completa

# Configuration
CANCIONES_DIR = "music"
DB_PATH = "music_database"
DIR_RESULTADOS = "resultados"

def poblar_base_de_datos():
    print("--- [DEBUG] Starting DB population ---")
    
    # 1. Clean old files
    if os.path.exists(DB_PATH):
        print(f"--- [DEBUG] Deleting old database at {DB_PATH} ---")
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            print(f"Warning: Could not delete old DB: {e}")

    # Note: We DON'T delete 'resultados' anymore so you can see the Demucs output
    
    # 2. Init DB
    print("--- [DEBUG] Connecting to ChromaDB... ---")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = create_and_get_music_collection()
    except Exception as e:
        print(f"CRITICAL ERROR connecting to DB: {e}")
        return

    # 3. Find files
    print(f"--- [DEBUG] Looking for songs in '{CANCIONES_DIR}'... ---")
    if not os.path.exists(CANCIONES_DIR):
        print(f"CRITICAL ERROR: The folder '{CANCIONES_DIR}' does not exist!")
        print(f"Please create a folder named '{CANCIONES_DIR}' next to this script and put MP3s inside.")
        return

    archivos = glob.glob(os.path.join(CANCIONES_DIR, "*.mp3")) + glob.glob(os.path.join(CANCIONES_DIR, "*.wav"))
    
    if len(archivos) == 0:
        print(f"CRITICAL ERROR: Found 0 files in '{CANCIONES_DIR}'.")
        print("Make sure files end in .mp3 or .wav")
        return

    print(f"--- [DEBUG] Found {len(archivos)} songs. Processing begins now... ---")

    for i, ruta in enumerate(archivos):
        print(f"\n[{i+1}/{len(archivos)}] Starting process for: {os.path.basename(ruta)}")
        start_time = time.time()
        
        try:
            # Call the GPU Heavy function
            fragmentos = procesar_cancion_completa(ruta, limpiar=False)
            
            if fragmentos:
                ids = [f['id'] for f in fragmentos]
                metadatas = [f['metadata'] for f in fragmentos]
                embeddings = [f['vector_resumen'] for f in fragmentos]
                
                print(f"   -> Saving {len(ids)} fragments to DB...")
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                print(f"   -> Done in {time.time() - start_time:.2f}s")
            else:
                print("   -> Warning: No fragments generated (Audio too short or silent?)")
                
        except Exception as e:
            print(f"   -> ERROR processing song: {e}")

    print("\n--- [DEBUG] Database Population Finished! ---")
    print(f"Total entries in DB: {collection.count()}")

if __name__ == "__main__":
    poblar_base_de_datos()