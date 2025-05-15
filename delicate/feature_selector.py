import json
import numpy as np
import statistics
from fuzzywuzzy import fuzz
import re
import random
import os



def load_json_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f1:
        data = json.load(f1)
    return data



# function to compute jaccard distance of surface form
def compute_jaccard_distance(set1, set2):
    set_intersection = len(set1.intersection(set2))
    set_union = len(set1.union(set2))
    return set_intersection/set_union


# function to compute time difference between document and candidate

def compute_time_difference(date1, date2):
    delta = date1 - date2
    days = delta.astype('timedelta64[D]').astype(int)
    return days


# compute features on dict of candidates
def compute_features(result):
    all_features = []
    identifier1 = result["identifier"]
    type1 = result["type"]
    surface = result["surface"].lower().strip()
    surface_tokens = set(re.split("\W", surface))
    scores = [float(candidate["score"]) for candidate in result["candidates"]]
    min_score = min(scores)
    max_score = max(scores)
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    publication_year = result.get("publication_date", None)
    for candidate in result["candidates"]:
        # extract candidate attributes
        type2 = candidate["type"]
        identifier2 = candidate["q_id"]
        cand_score = float(candidate["score"])
        min_date = candidate.get("min_date", None)
        wikipedia_title = re.sub("\(.*?\)", "", candidate["title"]).lower().strip()
        wikipedia_title_tokens = set(re.split("\W", wikipedia_title))

        # compute surface similarity
        lev_distance = fuzz.ratio(surface, wikipedia_title)
        jaccard_distance = compute_jaccard_distance(surface_tokens, wikipedia_title_tokens)

        # compute date difference in delta of days
        if min_date and publication_year:
            document_date = np.datetime64(publication_year)
            candidate_date = np.datetime64(min_date)
            time_delta = compute_time_difference(document_date, candidate_date)
        else:
            time_delta = 0

        # compute 1 if same type of 0 if not the same
        if type1 == type2 or type1=="WORK" and type2=="MISC":
            type_match = 1
        else:
            type_match = 0

        # compute qid match
        if identifier1 == identifier2:
            qid_match = 1
        else:
            qid_match = 0

        # get all features in one dict
        features = {
            "min_score":min_score,
            "max_score":max_score,
            "mean_score":mean_score,
            "median_score":median_score,
            "cand_score":cand_score,
            "lev_dist":lev_distance,
            "jacc_dist":jaccard_distance,
            "time_delta":time_delta,
            "type_match":type_match,
            "qid_match":qid_match
        }
        all_features.append(features)
    return all_features


def get_training_features(list_of_results, k):
    output = []
    for result_lookup in list_of_results:
        all_features = compute_features(result_lookup)
        positive_sample = [sample for sample in all_features if sample["qid_match"] == 1]
        median_score = all_features[0]["median_score"]
        if len(positive_sample) == 1:
            positive_sample = positive_sample[0]
            easy_samples = [sample for sample in all_features if sample["cand_score"] > median_score and sample[
                "qid_match"] == 0]
            hard_samples = [sample for sample in all_features if sample["cand_score"] <= median_score and sample[
                "qid_match"] == 0]
            easy_negatives = random.sample(easy_samples, k)
            hard_negatives = random.sample(hard_samples, k)
            output.extend([positive_sample] + easy_negatives + hard_negatives)
    return output
