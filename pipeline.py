from delicate.reranker import disambiguate_mentions_and_rerank
import os
import argparse
import json
import gc
from delicate.ner import load_ner_model, get_mentions_with_ner
from delicate.utils import shape_doc, load_from_config, generate_html_from_json


def main(params):
    ner_model = load_ner_model(params["config_file"])
    doc = shape_doc(params["text"], params["title"], params["publication_date"])
    print("Detecting entities in document: ", doc["doc_id"])
    doc_with_mentions = get_mentions_with_ner(doc, ner_model, params["tagset"], params["threshold_ner"])
    del ner_model
    gc.collect()
    biencoder, biencoder_params, indexer, conn, rf_classifier = load_from_config(params["config_file"])
    doc_with_entities = disambiguate_mentions_and_rerank(doc_with_mentions, biencoder, biencoder_params,
                                                         indexer, conn,
                                                         rf_classifier, top_k = params["top_k"],
                                                         threshold_nil=params["threshold_nil"])
    return doc_with_entities



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Named Entity Recognition and Linking Pipeline")
    parser.add_argument("--text", type=str, required=True, help="Input text for processing")
    parser.add_argument("--publication_date", type=str, default="", help="Optional publication date")
    parser.add_argument("--title", type=str, default="", help="Optional document title")
    parser.add_argument("--config_file", type=str, required=True, help="Path to configuration JSON file, e.g. configs/config_dz.json")
    parser.add_argument("--tagset", type=str, default="ALL", choices=["DZ", "AMD", "ALL"],
                        help="Tagset to be used: 'DZ' or 'AMD' or 'ALL'")
    parser.add_argument("--threshold_ner", type=float, default=0.9, help="Threshold of GliNER model")
    parser.add_argument("--top_k", type=int, default=50, choices=range(10, 101),
                        help=("Number of candidates to return"))
    parser.add_argument("--threshold_nil", type=float, default=0.5, help="Threshold of reranker")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output for annotated document")
    parser.add_argument("--format", type=str, default="JSON", choices=["JSON", "HTML"],
                        help="format to be used for output, optional: JSON or HTML")

    args = parser.parse_args()
    params = vars(args)

    result = main(params)
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    if params["format"] == "JSON":
        with open(os.path.join(params["output_dir"], "document.json"), "w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=4, ensure_ascii=False)
        out_f.close()

    elif params["format"] == "HTML":
        html_content = generate_html_from_json(result)
        with open(os.path.join(params["output_dir"], "document.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        f.close()