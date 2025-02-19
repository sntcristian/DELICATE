import csv
from elite.biencoder import encode_mention_from_dict
from elite.indexer import search_index_from_dict
from elite.utils import load_from_config, load_csv_dataset
from elite.reranker import rerank_candidates_from_dict


def process_documents(documents, config_file):
    output = []
    biencoder, biencoder_params, indexer, conn, rf_classifier = load_from_config(config_file)
    for doc in documents:
        print("Encoding mentions in document: ", doc["doc_id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=50)
        doc_reranked = rerank_candidates_from_dict(doc_with_candidates, rf_classifier)
        for result in doc_reranked["annotations"]:
            if result["best_linking"]["rf_score"] < 0.5:
                identifier = "NIL"
                wiki_title = ""
            else:
                identifier = result["best_linking"]["q_id"]
                wiki_title = result["best_linking"]["title"]
            item = {"doc_id": result["doc_id"], "start_pos": result["start_pos"], "end_pos": result["end_pos"],
                    "surface": result["surface"], "type": result["type"], "identifier": identifier,
                    "wiki_title": wiki_title, "score": result["best_linking"]["rf_score"]}
            output.append(item)
    return output



paragraphs_path = "../ENEIDE/DZ/v0.1/paragraphs_test.csv"
annotations_path = "../ENEIDE/DZ/v0.1/annotations_test.csv"

input_data = load_csv_dataset(paragraphs_path, annotations_path)

results_rerankered = process_documents(input_data, "config.json")

with open("DZ_results/ed_b50_n10/output.csv", "w", encoding="utf-8") as out_f:
    dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(results_rerankered)

