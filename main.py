import csv
from elite.biencoder import encode_mention_from_dict
from elite.indexer import search_index_from_dict
from elite.reranker import rerank_candidates_from_dict
from elite.utils import load_from_config


def main(doc, config_file):
    biencoder, biencoder_params, indexer, conn, rf_classifier = load_from_config(config_file)

    # encode mention into a base64 vector
    doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)

    # find candidates into faiss index by using encoded vector
    doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)

    # reranks candidates based on random forest
    doc_reranked = rerank_candidates_from_dict(doc_with_candidates, rf_classifier)
    return doc_reranked


doc = {
    "id":"no_id",
    "text":"Ho appena letto le Operette Morali, una serie di dialoghi filosofici scritti da Giacomo Leopardi.",
    "publication_date":"2025",
    "annotations":[
        {"doc_id":"no_id",
         "surface":"Operette Morali",
         "start_pos":19,
         "end_pos":34,
         "type":"WORK"
         }
    ]
}

output = main(doc, "config.json")
print(output)



