import sqlite3
import numpy as np
import base64
import sys
from blink.main_dense import load_biencoder, _process_biencoder_dataloader
from blink.indexer.faiss_indexer import DenseHNSWFlatIndexer


# Funzione per decriptare l'encoding base64
def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v


# Funzione per connettersi al database SQLite
def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn


def query_db_for_entities(conn, candidate_ids):
    results = []
    for candidate_id in candidate_ids:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, wikipedia_id, type_, wikidata_qid, descr, min_date
            FROM entities
            WHERE id = ?
        """, (int(candidate_id),))
        results.extend(cursor.fetchall())
    return results


# Funzione per caricare l'indice FAISS HNSW
def load_faiss_index(index_path):
    indexer = DenseHNSWFlatIndexer(1)
    indexer.deserialize_from(index_path)
    return indexer


def search_knn(indexer, encodings, top_k):
    encodings = np.array([vector_decode(e) for e in encodings])
    scores, candidates = indexer.search_knn(encodings, top_k)
    return scores, candidates

def load_resources(params):
    index_path = params["index_path"]
    print(f"Caricamento dell'indice da: {index_path}")
    indexer = load_faiss_index(index_path)

    db_path = params["db_path"]
    conn = connect_to_db(db_path)
    return indexer, conn

def search_index_from_dict(doc, indexer, conn, top_k):
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


    print(f"Esecuzione della ricerca KNN per {len(encodings)} entità...")
    scores, candidates = search_knn(indexer, encodings, top_k)

    all_candidates_info = []
    all_scores = []
    for i, score_and_candidate_ids in enumerate(zip(scores, candidates)):
        score_ranks = score_and_candidate_ids[0]
        candidate_ids = score_and_candidate_ids[1]
        candidate_info = query_db_for_entities(conn, candidate_ids)
        all_candidates_info.append(candidate_info)
        all_scores.append(score_ranks.tolist())

    for mention, score, candidate_info in zip(mentions, all_scores, all_candidates_info):
        if candidate_info:
            mention['linking']['candidates'] = candidate_info
            mention["linking"]["scores"] = score
        else:
            mention['linking']['candidates'] = []

    print("Risultati della ricerca KNN completati.")
    return doc
