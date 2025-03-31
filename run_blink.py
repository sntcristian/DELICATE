import csv
import os.path

from elite.biencoder import encode_mention_from_dict
from elite.indexer import search_index_from_dict
from elite.utils import load_from_config, load_csv_dataset, shape_result_lookup
import argparse


def process_documents(documents, config_file, top_k):
    output = []
    biencoder, biencoder_params, indexer, conn, rf_classifier = load_from_config(config_file)
    for doc in documents:
        print("Encoding mentions in document: ", doc["doc_id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=top_k)
        annotations_with_candidates = shape_result_lookup(doc_with_candidates)
        for annotation in annotations_with_candidates:
            candidate = annotation["candidates"][0]
            output.append({
                "doc_id":annotation["doc_id"],
                "surface":annotation["surface"],
                "start_pos":annotation["start_pos"],
                "end_pos":annotation["end_pos"],
                "type":annotation["type"],
                "identifier":candidate["q_id"],
                "score":candidate["score"]
            })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BLINK Retriever for Entity Disambiguation")
    parser.add_argument("--documents", type=str, required=True, help="Path to CSV containing documents.")
    parser.add_argument("--annotations", type=str, required=True, help="Path to CSV containing annotations.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--output_dir", type=str, default="./", help="Path to output directory.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidates to return for kNN Search.")
    args = parser.parse_args()
    params = vars(args)

    paragraphs_path = params["documents"]
    annotations_path = params["annotations"]
    top_k = params["top_k"]
    input_data = load_csv_dataset(paragraphs_path, annotations_path)
    results_rerankered = process_documents(input_data, params["config"], params["top_k"])
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    with open(os.path.join(params["output_dir"],"output.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(results_rerankered)



