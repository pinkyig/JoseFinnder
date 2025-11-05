import os
import sys
import shutil
import numpy as np
import pretty_midi
import chromadb
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

from songConverter import crear_vector_caracteristicas

DB_PATH = "music_database"
COLLECTION_NAME = "fragmentos_musicales"

def obtener_vector_de_query(ruta_audio_query: str) -> np.ndarray | None:
    print(f"Procesando audio de consulta: {ruta_audio_query}")

    try:
        print("-> Ejecutando basic-pitch sobre el audio...")
        model_output, midi_data, note_events = predict(ruta_audio_query, ICASSP_2022_MODEL_PATH)

        if len(midi_data.instruments) == 0:
            print("No se detectaron notas en el audio de consulta.")
            return None
    
        print(" -> Creando vector de características para consulta...")
        vector = crear_vector_caracteristicas(midi_data)

        if vector is not None:
            print(" -> Vector de consulta generado exitosamente.")
            return vector
        else:
            print(" -> No se pudo generar un vector para la consulta.")
            return None

    except Exception as e:
        print(f"Error procesando audio de consulta: {e}")
        return None

def buscar_similares(vector_query: np.ndarray, top_k: int = 5):

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        print(f"Error: La colección '{COLLECTION_NAME}' no fue encontrada.")
        return

    results = collection.query(
        query_embeddings=[vector_query.tolist()],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    if not results["ids"][0]:
        print("No se encontraron coincidencias.")
        return

    print("\n--- Resultados de la búsqueda ---")
    for i in range(len(results["ids"][0])):
        metadata = results['metadatas'][0][i]
        distancia = results['distances'][0][i]

        cancion = metadata.get('cancion', 'Desconocida')
        inicio = metadata.get('inicio_segundos', 0)

        print(f"{i+1}. Canción: '{cancion}'")
        print(f"   - Coincidencia cerca del segundo: {inicio:.1f}")
        print(f"   - Distancia (menor es mejor): {distancia:.4f}")
        print("-" * 20)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python query_database.py <ruta_al_archivo_de_audio_query>")
        sys.exit(1)
    
    ruta_query = sys.argv[1]

    if not os.path.exists(ruta_query):
        print(f"Error: El archivo '{ruta_query}' no existe.")
        sys.exit(1)
    
    vector_caracteristicas = obtener_vector_de_query(ruta_query)

    if vector_caracteristicas is not None:
        buscar_similares(vector_caracteristicas, top_k=1)