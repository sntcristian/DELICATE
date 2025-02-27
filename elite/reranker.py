from sklearn.ensemble import RandomForestClassifier
from elite.feature_selector import compute_features
from elite.utils import shape_result_lookup
from elite.biencoder import encode_mention_from_dict
from elite.indexer import search_index_from_dict
import pandas as pd
import numpy as np

def get_rf_scores(model, features):
    input_data = pd.DataFrame(features)
    input_features = input_data[
        ['min_score', 'max_score', 'mean_score', 'median_score', 'cand_score', 'lev_dist', 'jacc_dist',
         'time_delta', 'type_match']]
    probabilities = model.predict_proba(input_features)
    matching_probabilities = [prob[1] for prob in probabilities]
    return matching_probabilities



def rerank_candidates_from_dict(doc, rf_classifier):
    annotations_with_candidates = shape_result_lookup(doc)
    for idx, item in enumerate(annotations_with_candidates):
        results_rerankered = []
        similarity_features = compute_features(item)
        matching_probabilities = get_rf_scores(rf_classifier, similarity_features)
        zipped_candidates = zip(matching_probabilities, item["candidates"])
        sorted_candidates = sorted(zipped_candidates, reverse=True, key=lambda x: x[0])
        for rf_score, candidate in sorted_candidates:
            results_rerankered.append(
                {
                    "title": candidate["title"],
                    "q_id": candidate["q_id"],
                    "blink_score": candidate["score"],
                    "type":candidate["type"],
                    "min_date": candidate["min_date"],
                    "rf_score":rf_score
                }
            )
        item["best_linking"]=results_rerankered[0]
        item["candidates"]=results_rerankered
        doc["annotations"][idx]=item
    return doc



def disambiguate_mentions_and_rerank(doc,
                                     biencoder,
                                     biencoder_params,
                                     indexer,
                                     conn,
                                     rf_classifier,
                                     top_k=50,
                                     threshold_nil=0.5):

    entities = []
    print("Encoding mentions in document: ", doc["doc_id"])
    doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
    doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=top_k)
    doc_reranked = rerank_candidates_from_dict(doc_with_candidates, rf_classifier)
    for result in doc_reranked["annotations"]:
        if result["best_linking"]["rf_score"] < threshold_nil:
            identifier = "NIL"
            wiki_title = ""
        else:
            identifier = result["best_linking"]["q_id"]
            wiki_title = result["best_linking"]["title"]
        entities.append({"doc_id": result["doc_id"], "start_pos": result["start_pos"], "end_pos": result["end_pos"],
                    "surface": result["surface"], "type": result["type"], "identifier": identifier,
                    "wiki_title": wiki_title, "score": result["best_linking"]["rf_score"]})
    output = {
        "doc_id":doc["doc_id"],
        "title":doc["title"],
        "text":doc["text"],
        "publication_date":doc["publication_date"],
        "entities":entities
    }
    return output