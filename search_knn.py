import psycopg2
import pickle
import numpy as np
import base64
from blink.indexer.faiss_indexer import DenseHNSWFlatIndexer


# Funzione per decriptare l'encoding base64
def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v


# Funzione per connettersi al database PostgreSQL
def connect_to_db(port, db_name, pw):
    conn = psycopg2.connect(
        dbname=db_name,  # Nome del database ripristinato
        user="postgres",  # Utente di PostgreSQL
        password=pw,  # La password che hai impostato
        host="localhost",  # Host locale
        port=port  # Porta PostgreSQL di default
    )

    return conn


# Funzione per eseguire la query nel database per recuperare i dati delle entità
def query_db_for_entities(conn, candidate_ids):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, wikipedia_id, type_, wikidata_qid, redirects_to
            FROM entities
            WHERE id IN ({})
        """.format(','.join([str(int(id)) for id in candidate_ids])))
        results = cur.fetchall()
    return results


# Funzione per caricare l'indice FAISS HNSW
def load_faiss_index(index_path):
    indexer = DenseHNSWFlatIndexer(1)
    indexer.deserialize_from(index_path)
    return indexer


# Funzione per eseguire la ricerca KNN
def search_knn(indexer, encodings, top_k):
    encodings = np.array([vector_decode(e) for e in encodings])  # Decriptiamo l'encoding
    scores, candidates = indexer.search_knn(encodings, top_k)
    return scores, candidates


# Funzione principale per la ricerca KNN
def main(doc, index_path, port, db_name, pw, top_k=10):
    # 1. Caricare l'indice FAISS
    print(f"Caricamento dell'indice da: {index_path}")
    indexer = load_faiss_index(index_path)

    # 2. Connettersi al database PostgreSQL
    print(f"Connessione al database: {db_name}")
    conn = connect_to_db(port=port, db_name=db_name, pw=pw)

    # 3. Estrazione degli encodings dal dizionario
    encodings = []
    mentions = []
    for annotation in doc['annotations']:
        if 'linking' in annotation and 'encoding' in annotation['linking']:
            enc = annotation['linking']['encoding']
            encodings.append(enc)
            mentions.append(annotation)

    if not encodings:
        print("Nessun encoding trovato nel documento.")
        return

    # 4. Eseguire la ricerca KNN
    print(f"Esecuzione della ricerca KNN per {len(encodings)} entità...")
    scores, candidates = search_knn(indexer, encodings, top_k)

    # 5. Recuperare informazioni dal database per ciascun candidato
    all_candidates_info = []
    for i, candidate_ids in enumerate(candidates):
        candidate_ids_set = set(candidate_ids)
        candidate_info = query_db_for_entities(conn, candidate_ids_set)
        all_candidates_info.append(candidate_info)

    # 6. Aggiungere le informazioni di ritorno alle annotazioni nel documento
    for mention, candidate_info in zip(mentions, all_candidates_info):
        if candidate_info:
            mention['linking']['candidates'] = candidate_info
        else:
            mention['linking']['candidates'] = []

    # Chiudere la connessione al database
    conn.close()

    print("Risultati della ricerca KNN completati.")
    return doc


if __name__ == "__main__":
    # Dizionario di esempio
    doc = {
        'text': 'Leonardo da Vinci fu un genio poliedrico.',
        'annotations': [
            {
                'start_pos': 0,
                'end_pos': 17,
                'type': 'PER',
                'linking': {
                    'encoding': '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'
                }
            }
        ]
    }

    # Percorso dell'indice FAISS HNSW
    index_path = 'models/faiss_hnsw_ita_index.pkl'

    # Dati del database PostgreSQL (modificalo secondo le tue credenziali)
    db_name = "wikipedia_it"
    port=5432
    pw="TortaDatteri044"

    # Numero di vicini più vicini (K)
    top_k = 10


    # Esegui la ricerca
    result = main(doc, index_path, port, db_name, pw, top_k)

    print("Risultato finale:")
    print(result)
