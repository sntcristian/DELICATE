import csv
import os
import pandas as pd
import numpy as np


def reshape_data_input(data, annotations):
    output = []
    for row1 in data:
        annotations_list = [row2 for row2 in annotations if row2["doc_id"] == row1["doc_id"]]
        if len(annotations_list)>0:
            doc = {
                "id":row1["doc_id"],
                "text":row1["text"],
                "annotations":annotations_list,
                "publication_date":row1["publication_date"]
            }
            output.append(doc)
    return output



def load_csv_dataset(paragraphs_path, annotations_path):
    with open(paragraphs_path, "r", encoding="utf-8") as f1:
        paragraphs = csv.DictReader(f1)
        paragraphs = list(paragraphs)
    f1.close()
    
    with open(annotations_path, "r", encoding="utf-8") as f2:
        annotations = csv.DictReader(f2)
        annotations = list(annotations)
    f2.close()

    input_data = reshape_data_input(paragraphs, annotations)
    return input_data




def load_csv_from_directory(dataset_path):
    paragraphs_train_path = os.path.join(dataset_path, "paragraphs_train.csv")
    annotations_train_path = os.path.join(dataset_path, "annotations_train.csv")
    input_train = load_csv_dataset(paragraphs_train_path, annotations_train_path)
    
    paragraphs_dev_path = os.path.join(dataset_path, "paragraphs_dev.csv")
    annotations_dev_path = os.path.join(dataset_path, "annotations_dev.csv")
    input_dev = load_csv_dataset(paragraphs_dev_path, annotations_dev_path)
    
    paragraphs_test_path = os.path.join(dataset_path, "paragraphs_test.csv")
    annotations_test_path = os.path.join(dataset_path, "annotations_test.csv")
    input_test = load_csv_dataset(paragraphs_test_path, annotations_test_path)
    return input_train, input_dev, input_test



# shape output of candidate lookup into dict
def shape_result_lookup(output):
    list_of_dict = []
    for result in output:
        for annotation in result["annotations"]:
            _id = annotation["doc_id"]
            surface = annotation["surface"]
            start_pos = annotation["start_pos"]
            end_pos = annotation["end_pos"]
            _type = annotation["type"]
            identifier = annotation.get("identifier", "None")
            candidates = []
            for candidate, score in zip(annotation["linking"]["candidates"], annotation["linking"]["scores"]):
                alias = candidate[1]
                q_id = "Q" + str(candidate[4])
                wikidata_type = candidate[3]
                min_date = candidate[6]
                candidates.append({
                    "title": alias,
                    "q_id": q_id,
                    "score": score,
                    "type": wikidata_type,
                    "min_date": min_date
                })
            list_of_dict.append({
                "doc_id": _id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "identifier": identifier,
                "type": _type,
                "surface": surface,
                "publication_date": result["publication_date"],
                "candidates": candidates
            })
    return list_of_dict


def get_rf_scores(model, features):
    input_data = pd.DataFrame(features)
    input_features = input_data[
        ['min_score', 'max_score', 'mean_score', 'median_score', 'stddev_score', 'cand_score', 'lev_dist', 'jacc_dist',
         'time_delta', 'type_match']]
    probabilities = model.predict_proba(input_features)
    matching_probabilities = [prob[1] for prob in probabilities]
    return matching_probabilities