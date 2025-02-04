import csv
import os
from elite.biencoder import encode_mention_from_dict, get_mentions_with_ner
from elite.indexer import search_index_from_dict
from elite.utils import load_from_config, load_csv_dataset
from elite.reranker import rerank_candidates_from_dict
import argparse


def run_ed_pipeline(params):
    documents = load_csv_dataset(params["paragraphs_path"], params["annotations_path"])
    output = []
    biencoder, biencoder_params, indexer, conn, gbt_classifier = load_from_config(params["config_path"])
    for doc in documents:
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=params["top_k"])
        doc_reranked = rerank_candidates_from_dict(doc_with_candidates, gbt_classifier)
        for result in doc_reranked["annotations"]:
            if result["best_linking"]["rf_score"] < params["threshold_nil"]:
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



def run_nel_pipeline(params):
    output = []
    biencoder, biencoder_params, indexer, conn, gbt_classifier, ner_model = load_from_config(params["config_path"],
                                                                                             ner=True)
    csv_file = open(params["paragraphs_path"], "r", encoding="utf-8")
    dict_reader = csv.DictReader(csv_file)
    dict_reader = list(dict_reader)
    for row in dict_reader:
        doc = {
            "id":row["doc_id"],
            "text":row["text"],
            "publication_date":row["publication_date"]
        }
        labels = ["persona", "luogo", "opera"] if params["tagset"] == "DZ" else ["persona", "luogo", "organizzazione"]
        doc_with_mentions = get_mentions_with_ner(doc, ner_model, labels=labels, threshold=params["threshold_ner"])
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc_with_mentions, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=params["top_k"])
        doc_reranked = rerank_candidates_from_dict(doc_with_candidates, gbt_classifier)
        for result in doc_reranked["annotations"]:
            if result["best_linking"]["rf_score"] < params["threshold_nil"]:
                identifier = "NIL"
                wiki_title = ""
            else:
                identifier = result["best_linking"]["q_id"]
                wiki_title = result["best_linking"]["title"]
            item = {"doc_id": result["doc_id"], "start_pos": result["start_pos"], "end_pos": result["end_pos"],
                    "surface": result["surface"], "type": result["type"], "identifier": identifier,
                    "wiki_title": wiki_title, "nel_score": result["best_linking"]["rf_score"]}
            output.append(item)
    return output





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for ENEIDE dataset")
    parser.add_argument("--paragraphs_path", required=True, type=str,
                        default="../ENEIDE/DZ/v0.1/paragraphs_test.csv", help="path to text data in ENEIDE")
    parser.add_argument("--annotations_path", required=True, type=str,
                        default="../ENEIDE/DZ/v0.1/annotations_test.csv", help="path to annotation data.")
    parser.add_argument("--output_dir", required=True, type=str, default="results/ed/")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--ner", type=bool, default=False, help="Enable or disable NER (default: False)")
    parser.add_argument("--tagset", type=str, default="DZ", choices=["DZ", "AMD"],
                        help="Tagset to be used: 'DZ' or 'AMD'")
    parser.add_argument("--threshold_ner", type=float, default=0.5, help="Threshold of GliNER model")
    parser.add_argument("--top_k", type=int, default=100, choices=range(10, 101),
                        help=("Number of candidates to return"))
    parser.add_argument("--threshold_nil", type=float, default=0.51, help="Threshold of GliNER model")

    args = parser.parse_args()
    params = vars(args)

    if params["ner"] == True:
        result = run_nel_pipeline(params)
    else:
        result = run_ed_pipeline(params)

    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    with open(os.path.join(params["output_dir"], "output.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, result[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(result)

