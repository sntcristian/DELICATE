import csv
import os
from elite.utils import load_from_config, load_csv_dataset
from elite.reranker import disambiguate_mentions_and_rerank
import argparse




def process_documents(params):
    output = []
    documents = load_csv_dataset(params["documents"], params["annotations"])
    biencoder, biencoder_params, indexer, conn, gbt_classifier = load_from_config(params["config"])
    for doc in documents:
        doc_reranked = disambiguate_mentions_and_rerank(doc, biencoder, biencoder_params,
                                                        indexer, conn, gbt_classifier,
                                                        params["top_k"], params["threshold_nil"])
        output.extend(doc_reranked["entities"])
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ELITE for Entity Disambiguation")
    parser.add_argument("--documents", type=str, required=True, help="Path to CSV containing documents.")
    parser.add_argument("--annotations", type=str, required=True, help="Path to CSV containing annotations.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--output_dir", type=str, default="./", help="Path to output directory.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of candidates to return for kNN Search.")
    parser.add_argument("--threshold_nil", type=float, default=0.3, help="Upper bound for NIL tag.")
    args = parser.parse_args()
    params = vars(args)
    results_rerankered = process_documents(params)

    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    with open(os.path.join(params["output_dir"],"output.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(results_rerankered)

