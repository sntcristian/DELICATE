from sklearn.ensemble import RandomForestClassifier
from elite.feature_selector import compute_features
from elite.utils import shape_result_lookup
import pandas as pd
import numpy as np

def get_rf_scores(model, features):
    input_data = pd.DataFrame(features)
    input_features = input_data[
        ['min_score', 'max_score', 'mean_score', 'median_score', 'stddev_score', 'cand_score', 'lev_dist', 'jacc_dist',
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