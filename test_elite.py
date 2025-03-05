import csv
import os
from elite.utils import load_from_config, load_csv_dataset
from elite.reranker import disambiguate_mentions_and_rerank



params = {
    "paragraphs_path": "../ENEIDE/AMD/v1.0/paragraphs_test.csv",
    "annotations_path": "../ENEIDE/AMD/v1.0/annotations_test.csv",
    "config_file": "config.json",
    "top_k": 20,
    "threshold_nil": 0.3,
    "output_path":"./results/elite_ed"
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

if not os.path.exists(params["output_path"]):
    os.makedirs(params["output_path"])

with open(os.path.join(params["output_path"],"output.csv"), "w", encoding="utf-8") as out_f:
    dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(results_rerankered)

