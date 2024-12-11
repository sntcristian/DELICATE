import csv
from biencoder import encode_mention_from_dict
from indexer import load_resources, search_index_from_dict
import os
import json

params = {
    "db_path":"ELITE_models/wikipedia_it.sqlite",
    "index_path":"ELITE_models/faiss_hnsw_ita_index.pkl"
}


indexer, conn = load_resources(params)
print("Loading index and database complete.")


def reshape_data(data, annotations):
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

with open("../ENEIDE/DZ/paragraphs_test.csv", "r", encoding="utf-8") as f1:
    data = csv.DictReader(f1)
    data = list(data)

with open("../ENEIDE/DZ/annotations_test.csv", "r", encoding="utf-8") as f2:
    annotations = csv.DictReader(f2)
    annotations = list(annotations)

documents = reshape_data(data[:10], annotations[:10])

def main(documents):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)
        output.append(doc_with_candidates)
    return output


output = main(documents)

list_to_dump = []
for result in output:
    for annotation in result["annotations"]:
        _id = annotation["doc_id"]
        surface = annotation["surface"]
        start_pos = annotation["start_pos"]
        end_pos = annotation["end_pos"]
        _type = annotation["type"]
        identifier = annotation["identifier"]
        candidates = []
        for candidate, score in zip(annotation["linking"]["candidates"], annotation["linking"]["scores"]):
            alias = candidate[1]
            q_id = "Q"+str(candidate[4])
            descr = candidate[5]
            wikidata_type = candidate[3]
            min_date = candidate[6]
            candidates.append({
                "title":alias,
                "q_id":q_id,
                "score":score,
                "type":wikidata_type,
                "min_date":min_date
            })

        list_to_dump.append({
            "doc_id":_id,
            "start_pos":start_pos,
            "end_pos":end_pos,
            "identifier":identifier,
            "type":_type,
            "surface":surface,
            "publication_date":result["publication_date"],
            "candidates":candidates
        })


with open("candidates.json", "w", encoding="utf-8") as f1:
    json.dump(list_to_dump, f1, ensure_ascii=False, indent=4)

