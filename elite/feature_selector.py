import json
import numpy as np
import statistics
from fuzzywuzzy import fuzz
import re
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
def compute_features(json_f):
    all_features = []
    for entry in json_f:
        identifier1 = entry["identifier"]
        type1 = entry["type"]
        surface = entry["surface"].lower().strip()
        surface_tokens = set(re.split("\W", surface))
        scores = [float(candidate["score"]) for candidate in entry["candidates"]]
        min_score = min(scores)
        max_score = max(scores)
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores)
        median_score = statistics.median(scores)
        publication_year = entry.get("publication_date", None)
        for candidate in entry["candidates"]:

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
                "stddev_score":std_dev,
                "cand_score":cand_score,
                "lev_dist":lev_distance,
                "jacc_dist":jaccard_distance,
                "time_delta":time_delta,
                "type_match":type_match,
                "qid_match":qid_match
            }
            all_features.append(features)
    return all_features