import chromadb
import os

# --- Constantes de la Base de Datos ---
# Define las constantes de tu base de datos en un solo lugar para que
# todos los scripts usen la misma configuración.

# El nombre del directorio donde se guardarán los archivos de la base de datos.
DB_PATH = "music_database"

# El nombre de la colección donde se almacenarán los fragmentos musicales.
COLLECTION_NAME = "fragmentos_musicales"

def get_music_collection() -> chromadb.Collection | None:
    """
    Se conecta a la base de datos ChromaDB persistente y devuelve el objeto 
    de la colección musical.

    Esta función está diseñada para ser usada por scripts que LEEN de la base de datos
    (como el script de búsqueda).
    
    Returns:
        Un objeto chromadb.Collection si la conexión es exitosa y la colección existe.
        None si ocurre un error (ej. la base de datos o la colección no se han creado aún).
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: El directorio de la base de datos '{DB_PATH}' no existe.")
        print("Por favor, ejecuta primero el script 'poblar_base_de_datos.py' para crearla.")
        return None
        
    try:
        # Crea un cliente persistente que lee/escribe los datos en el directorio DB_PATH.
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Intenta obtener la colección. Esto fallará si la colección no existe.
        # Usamos get_collection() en lugar de get_or_create_collection() porque
        # en un script de búsqueda, esperamos que la colección ya exista.
        collection = client.get_collection(name=COLLECTION_NAME)
        
        print(f"✅ Conexión exitosa a la colección '{COLLECTION_NAME}' con {collection.count()} elementos.")
        return collection
    
    except ValueError as e:
        # Este error es común si la colección no se encuentra.
        print(f"Error: No se pudo encontrar la colección '{COLLECTION_NAME}'.")
        print("Asegúrate de haber ejecutado el script 'poblar_base_de_datos.py' correctamente.")
        return None
        
    except Exception as e:
        print(f"❌ Error fatal al conectarse a la base de datos: {e}")
        return None

def create_and_get_music_collection() -> chromadb.Collection:
    """
    Se conecta a la base de datos ChromaDB y crea la colección si no existe.

    Esta función está diseñada para ser usada por el script que ESCRIBE en la base
    de datos por primera vez (poblar_base_de_datos.py).
    
    Returns:
        Un objeto chromadb.Collection listo para ser usado para añadir datos.
    """
    print(f"Inicializando la base de datos en: '{os.path.abspath(DB_PATH)}'")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Usa get_or_create_collection para crear la colección la primera vez
    # con la configuración de distancia correcta (similitud coseno).
    # Si la colección ya existe, simplemente la devuelve sin hacer cambios.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"✅ Colección '{COLLECTION_NAME}' lista con {collection.count()} elementos.")
    return collection