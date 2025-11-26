# JoseFinder
**Best Dj Mendez App**


```
python3 init_chromadb.py
```

Usar ```--range``` para elegir un slice del dataset (primer indice es 1).

- Descargar archivos mp3
    ```
    python3 batch_downloader.py --range 1-100
    ```

- Separar en pistas y segmentar.
    ```
    python3 split_and_segment.py --range 1-100
    ```

- Vectorizar los segmentos y guardar en ChromaDB.
    ```
    python3 vectorize.py --range 1-100
    ```

- Query usando ```tarareo.mp3```.
    ```
    python3 hum_query.py --file tarareo.mp3 --topk 10 --rerank-dtw --unique
    ```


metadata y links en `downloads_metadata.json`.