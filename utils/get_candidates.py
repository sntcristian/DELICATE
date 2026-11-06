from delicate.biencoder import encode_mention_from_dict
from delicate.indexer import search_index_from_dict
from delicate.utils import load_from_config, shape_result_lookup, load_csv_dataset
from delicate.reranker import rerank_candidates_from_dict 
import os
import argparse
import json



def retrieve_candidates(documents, block_size, biencoder, biencoder_params, indexer, conn):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["doc_id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=block_size)
        output.append(doc_with_candidates)
    return output



def main():
    parser = argparse.ArgumentParser(description="Entity disambiguation with candidate generation")
    parser.add_argument("--documents_path", type=str, default="../ENEIDE/AMD/v1.0/paragraphs_test.csv", help="Path to paragraphs dataset")
    parser.add_argument("--annotations_path", type=str, default="../ENEIDE/AMD/v1.0/annotations_test.csv", help="Path to annotations dataset")
    parser.add_argument("--output_dir", type=str, default="./camera-ready/EL/AMD/", help="Output directory for results")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top candidates to retrieve")
    parser.add_argument("--config_path", type=str, default="configs/config_all.json", help="Number of documents in a batch")

    args = parser.parse_args()

    biencoder, biencoder_params, indexer, conn, rf_classifier = load_from_config(args.config_path)

    input_data = load_csv_dataset(args.documents_path, args.annotations_path)

    docs_with_candidates = retrieve_candidates(input_data, args.top_k, biencoder, biencoder_params, indexer, conn)

    docs_reranked = [rerank_candidates_from_dict(d, rf_classifier) for d in docs_with_candidates]
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"candidates_top{args.top_k}.json"), "w", encoding="utf-8") as out_f:
        json.dump(docs_reranked, out_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()




