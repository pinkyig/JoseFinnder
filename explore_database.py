import chromadb
import numpy as np

# --- 1. Parámetros de la Base de Datos (DEBEN COINCIDIR con tus otros scripts) ---
DB_PATH = "music_database"
COLLECTION_NAME = "fragmentos_musicales"

def explorar_base_de_datos(limite: int = None):
    """
    Se conecta a la base de datos ChromaDB y muestra sus contenidos.
    
    Args:
        limite (int, optional): El número máximo de elementos a mostrar.
                                Si es None, se mostrarán todos.
    """
    print(f"--- Conectando a la base de datos en '{DB_PATH}' ---")
    
    try:
        # Conectarse al cliente de ChromaDB que apunta a tu base de datos en disco
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Obtener la colección (si no existe, dará un error)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Colección '{COLLECTION_NAME}' cargada exitosamente.")
        
    except ValueError:
        print(f"\nError: La colección '{COLLECTION_NAME}' no fue encontrada.")
        print("Asegúrate de haber ejecutado 'crear_base_de_datos.py' primero.")
        return
    except Exception as e:
        print(f"Ocurrió un error inesperado al conectar a la base de datos: {e}")
        return

    # Obtener el número total de elementos en la colección
    total_items = collection.count()
    if total_items == 0:
        print("\nLa base de datos está vacía.")
        return
        
    print(f"\nLa base de datos contiene un total de {total_items} fragmentos.")
    
    # Usar collection.get() para obtener los datos
    # 'include' nos permite especificar qué campos queremos recibir.
    # Pedimos todo: IDs, metadatos y los vectores (embeddings).
    print("Obteniendo registros...")
    data = collection.get(
        ids=None, # Pasar None o no especificarlo para obtener todos los IDs
        limit=limite, # Aplica el límite si se especificó uno
        include=["metadatas", "embeddings"] 
    )
    
    # Extraer las listas de la respuesta
    ids = data['ids']
    metadatas = data['metadatas']
    embeddings = data['embeddings']
    
    print(f"\n--- Mostrando los primeros {len(ids)} registros ---")
    
    # Iterar y mostrar cada registro de forma legible
    for i in range(len(ids)):
        item_id = ids[i]
        item_metadata = metadatas[i]
        item_embedding = np.array(embeddings[i]) # Convertir a array de numpy para mejor visualización
        
        print(f"\nID: {item_id}")
        print(f"  - Metadata:")
        # Imprimir cada par de clave-valor en la metadata
        for key, value in item_metadata.items():
            print(f"    - {key}: {value}")
        
        # Imprimir una vista previa del vector para no llenar la pantalla
        vector_preview = ", ".join([f"{x:.4f}" for x in item_embedding[:4]])
        print(f"  - Vector (primeros 4 de {item_embedding.shape[0]} dims): [{vector_preview}, ...]")
        print("-" * 30)

    if limite and total_items > limite:
        print(f"\nSe mostraron {limite} de {total_items} registros.")

if __name__ == "__main__":
    # Por defecto, muestra todos los registros.
    # Si quieres ver solo los primeros 10, por ejemplo, ejecuta:
    # explorar_base_de_datos(limite=10)
    explorar_base_de_datos()