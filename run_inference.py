import argparse
import json
from elite.biencoder import encode_mention_from_dict, get_mentions_with_ner
from elite.indexer import search_index_from_dict
from elite.reranker import rerank_candidates_from_dict
from elite.utils import load_from_config


def main(params):
    if not params["ner"]:
        biencoder, biencoder_params, indexer, conn, gbt_classifier = load_from_config(params["config_path"])
        annotations = json.loads(params["annotations"]) if params["annotations"] else []
        doc_id = annotations[0]["doc_id"] if annotations else ""
        doc_with_mentions = {
            "id": doc_id,
            "text": params["text"],
            "publication_date": params.get("publication_date", ""),
            "annotations": annotations
        }
    else:
        doc = {
            "id": "",
            "text": params["text"],
            "publication_date": params.get("publication_date", "")
        }
        biencoder, biencoder_params, indexer, conn, gbt_classifier, ner_model = load_from_config(
            params["config_path"], ner=True
        )

        if params["tagset"] not in {"DZ", "AMD"}:
            raise ValueError("tagset value has to be either 'DZ' or 'AMD'")

        labels = ["persona", "luogo", "opera"] if params["tagset"] == "DZ" else ["persona", "luogo", "organizzazione"]

        doc_with_mentions = get_mentions_with_ner(doc, ner_model, labels=labels)

    doc_with_linking = encode_mention_from_dict(doc_with_mentions, biencoder, biencoder_params)
    doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)
    doc_reranked = rerank_candidates_from_dict(doc_with_candidates, gbt_classifier)

    return doc_reranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Named Entity Recognition and Linking Pipeline")
    parser.add_argument("--ner", type=bool, default=True, help="Enable or disable NER (default: True)")
    parser.add_argument("--text", type=str, required=True, help="Input text for processing")
    parser.add_argument("--publication_date", type=str, default="", help="Optional publication date")
    parser.add_argument("--annotations", type=str, default=None, help="JSON formatted annotations if NER is disabled")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--tagset", type=str, default="DZ", choices=["DZ", "AMD"],
                        help="Tagset to be used: 'DZ' or 'AMD'")

    args = parser.parse_args()
    params = vars(args)

    result = main(params)
    print(json.dumps(result, indent=4, ensure_ascii=False))
