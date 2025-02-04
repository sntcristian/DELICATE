import argparse
import json
from elite.biencoder import encode_mention_from_dict, get_mentions_with_ner
from elite.indexer import search_index_from_dict
from elite.reranker import rerank_candidates_from_dict
from elite.utils import load_from_config


def main(params):
    if params["ed_only"] == True:
        biencoder, biencoder_params, indexer, conn, gbt_classifier = load_from_config(params["config_path"])
        try:
            print(params["annotations"])
            annotations = json.loads(params["annotations"]) if params["annotations"] else []
            doc_id = annotations[0]["doc_id"] if annotations else ""
            doc_with_mentions = {
                "id": doc_id,
                "text": params["text"],
                "publication_date": params.get("publication_date", ""),
                "annotations": annotations
            }
            doc_with_linking = encode_mention_from_dict(doc_with_mentions, biencoder, biencoder_params)
            doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=params["top_k"])
            doc_reranked = rerank_candidates_from_dict(doc_with_candidates, gbt_classifier)
            for annotation in doc_reranked["annotations"]:
                if annotation["best_linking"]["rf_score"] < params["threshold_nil"]:
                    annotation["best_linking"]["q_id"] = "NIL"
            return doc_reranked

        except:
            raise ValueError("NER annotations should be provided in JSON formatted string.")

    elif params["ner"] == True:
        doc = {
            "id": "",
            "text": params["text"],
            "publication_date": params.get("publication_date", "")
        }
        biencoder, biencoder_params, indexer, conn, gbt_classifier, ner_model = load_from_config(
            params["config_path"], ner=True
        )
        labels = ["persona", "luogo", "opera"] if params["tagset"] == "DZ" else ["persona", "luogo", "organizzazione"]
        doc_with_mentions = get_mentions_with_ner(doc, ner_model, labels=labels, threshold=params["threshold_ner"])
        doc_with_linking = encode_mention_from_dict(doc_with_mentions, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=params["top_k"])
        doc_reranked = rerank_candidates_from_dict(doc_with_candidates, gbt_classifier)
        for annotation in doc_reranked["annotations"]:
            if annotation["best_linking"]["rf_score"] < params["threshold_nil"]:
                annotation["best_linking"]["q_id"] = "NIL"
                annotation["best_linking"]["wiki_title"] = ""
        return doc_reranked



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Named Entity Recognition and Linking Pipeline")
    parser.add_argument("--ner", type=bool, default=True, help="Enable or disable NER (default: True)")
    parser.add_argument("--text", type=str, required=True, help="Input text for processing")
    parser.add_argument("--publication_date", type=str, default="", help="Optional publication date")
    parser.add_argument("--ed_only", type=bool, default=False, help="Enable or disable NER (default: True)")
    parser.add_argument("--annotations", type=str, default=None, help="JSON formatted annotations if NER is disabled")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--tagset", type=str, default="DZ", choices=["DZ", "AMD"],
                        help="Tagset to be used: 'DZ' or 'AMD'")
    parser.add_argument("--threshold_ner", type=float, default=0.5, help="Threshold of GliNER model")
    parser.add_argument("--top_k", type=int, default=10, choices=range(10, 101),
                        help=("Number of candidates to return"))
    parser.add_argument("--threshold_nil", type=float, default=0.51, help="Threshold of GliNER model")

    args = parser.parse_args()
    params = vars(args)

    result = main(params)
    print(json.dumps(result, indent=4, ensure_ascii=False))
