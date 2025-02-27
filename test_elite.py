import csv
from elite.biencoder import encode_mention_from_dict
from elite.indexer import search_index_from_dict
from elite.utils import load_from_config, load_csv_dataset
from elite.reranker import disambiguate_mentions_and_rerank



params = {
    "paragraphs_path": "../ENEIDE/DZ/v0.1/paragraphs_test.csv",
    "annotations_path": "./DZ_results/gliner_dz/output.csv",
    "config_file": "config.json",
    "top_k": 50,
    "threshold_nil": 0.5
}


def process_documents(params):
    output = []
    documents = load_csv_dataset(params["paragraphs_path"], params["annotations_path"])
    biencoder, biencoder_params, indexer, conn, gbt_classifier = load_from_config(params["config_file"])
    for doc in documents:
        doc_reranked = disambiguate_mentions_and_rerank(doc, biencoder, biencoder_params,
                                                        indexer, conn, gbt_classifier,
                                                        params["top_k"], params["threshold_nil"])
        output.extend(doc_reranked["entities"])
    return output




results_rerankered = process_documents(params)

with open("DZ_results/elite_el_no_stddev.csv", "w", encoding="utf-8") as out_f:
    dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(results_rerankered)

