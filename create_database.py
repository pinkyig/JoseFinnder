import os
import glob
import chromadb
import json
from db_connector import create_and_get_music_collection

from songConverter import procesar_cancion_completa

CANCIONES_DIR = "music"

DB_PATH = "music_database"

COLLECTION_NAME = "fragmentos_musicales"

client = chromadb.PersistentClient(path=DB_PATH)

collection = create_and_get_music_collection()

def poblar_base_de_datos():
        archivos_audio = glob.glob(os.path.join(CANCIONES_DIR, "*.mp3")) + glob.glob(os.path.join(CANCIONES_DIR, "*.wav"))

        print(f"Se encontraron {len(archivos_audio)} canciones para procesar.")

        for ruta_cancion in archivos_audio:
                nombre_cancion = os.path.basename(ruta_cancion)

                resultados_existentes = collection.get(where={"cancion": nombre_cancion}, limit=1)
                if resultados_existentes["ids"]:
                    print(f"-> La canción '{nombre_cancion}' ya existe en la base de datos, saltando")
                    continue

                print(f"\nProcesando '{nombre_cancion}'...")

                fragmentos = procesar_cancion_completa(ruta_cancion, limpiar=True)

                if not fragmentos:
                    print(f"No se generaron vectores para '{nombre_cancion}'.")
                    continue
                
                valid_fragments = [f for f in fragmentos if f.get("vector_resumen") is not None]
                if not valid_fragments:
                    print(f"No se generaron vectores resumen válidos para '{nombre_cancion}'.")
                    continue

                ids = [f['id'] for f in valid_fragments]
                vectores_resumen = [f['vector_resumen'].tolist() for f in valid_fragments]
                metadatas = [f['metadata'] for f in valid_fragments]

                try:
                    collection.add(
                        ids=ids,
                        embeddings=vectores_resumen,
                        metadatas=metadatas
                    )
                    print(f"-> Se añadieron {len(ids)} fragmentos de {nombre_cancion} a la base de datos.")
                except Exception as e:
                    print(f"Error al añadir fragmentos a la DB: {e}")
                
            
        print("\n--- Proceso de población de la base de datos finalizado. ---")
        print(f"Total de fragmentos en la base de datos: {collection.count()}")


if __name__ == "__main__":
     poblar_base_de_datos()
            