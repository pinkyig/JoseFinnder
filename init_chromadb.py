#!/usr/bin/env python3
"""
init_chromadb.py

Crea la base de datos ChromaDB persistente y la colección (si no existen).
No puebla la colección con datos.

Opciones:
  --reset        : elimina el directorio DB si ya existe (útil para reiniciar)
  --path <ruta>  : sobreescribe la ruta del DB (por defecto usa la constante de `db_connector`)
"""
import argparse
import os
import shutil
import chromadb
from db_connector import DB_PATH as DEFAULT_DB_PATH, COLLECTION_NAME

def main():
    parser = argparse.ArgumentParser(description="Inicializar ChromaDB (sin poblar)")
    parser.add_argument("--reset", action="store_true", help="Eliminar DB existente antes de crearla")
    parser.add_argument("--path", default=DEFAULT_DB_PATH, help="Ruta al directorio de la DB (por defecto: el de db_connector)")
    args = parser.parse_args()

    db_path = args.path

    if args.reset and os.path.exists(db_path):
        print(f"Eliminando base de datos existente en: {db_path}")
        try:
            shutil.rmtree(db_path)
        except Exception as e:
            print(f"Error eliminando {db_path}: {e}")
            return

    os.makedirs(db_path, exist_ok=True)
    print(f"Inicializando cliente persistente en: {os.path.abspath(db_path)}")

    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"ERROR: No se pudo crear el cliente ChromaDB: {e}")
        return

    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        print(f"✅ Colección '{COLLECTION_NAME}' lista. Elementos actuales: {collection.count()}")
    except Exception as e:
        print(f"ERROR creando/obteniendo la colección '{COLLECTION_NAME}': {e}")

if __name__ == "__main__":
    main()