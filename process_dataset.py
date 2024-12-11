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


def load_csv_datasets(dataset_path):
    with open(os.path.join(dataset_path, "paragraphs_train.csv"), "r", encoding="utf-8") as f1:
        paragraphs_train = csv.DictReader(f1)
        paragraphs_train = list(paragraphs_train)
    f1.close()
    with open(os.path.join(dataset_path, "annotations_train.csv"), "r", encoding="utf-8") as f2:
        annotations_train = csv.DictReader(f2)
        annotations_train = list(annotations_train)
    f2.close()
    with open(os.path.join(dataset_path, "paragraphs_dev.csv"), "r", encoding="utf-8") as f3:
        paragraphs_dev = csv.DictReader(f3)
        paragraphs_dev = list(paragraphs_dev)
    f3.close()
    with open(os.path.join(dataset_path, "annotations_dev.csv"), "r", encoding="utf-8") as f4:
        annotations_dev = csv.DictReader(f4)
        annotations_dev = list(annotations_dev)
    f4.close()
    with open(os.path.join(dataset_path, "paragraphs_test.csv"), "r", encoding="utf-8") as f5:
        paragraphs_test = csv.DictReader(f5)
        paragraphs_test = list(paragraphs_test)
    f3.close()
    with open(os.path.join(dataset_path, "annotations_test.csv"), "r", encoding="utf-8") as f6:
        annotations_test = csv.DictReader(f6)
        annotations_test = list(annotations_test)
    f4.close()

    input_train = reshape_data_input(paragraphs_train, annotations_train)
    input_dev = reshape_data_input(paragraphs_dev, annotations_dev)
    input_test = reshape_data_input(paragraphs_test, annotations_test)

    return input_train, input_dev, input_test




# shape output of candidate lookup into dict
def reshape_data_output(output):
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

            list_to_dump.append({
                "doc_id": _id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "identifier": identifier,
                "type": _type,
                "surface": surface,
                "publication_date": result["publication_date"],
                "candidates": candidates
            })
    return list_to_dump




def process_documents(documents):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)
        output.append(doc_with_candidates)
    output = reshape_data_output(output)
    return output


dataset_path = "../ENEIDE/DZ/v0.1"
output_directory = "DZ_results"

train_documents, dev_documents, test_documents = load_csv_datasets(dataset_path=dataset_path)


print("\n-------------\n Processing train dataset: \n")
output_train = process_documents(train_documents)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(os.path.join(output_directory, "candidates_train.json"), "w", encoding="utf-8") as out_f:
    json.dump(output_train, out_f, ensure_ascii=False, indent=4)
out_f.close()


print("\n-------------\n Processing dev dataset: \n")
output_dev = process_documents(dev_documents)

with open(os.path.join(output_directory, "candidates_dev.json"), "w", encoding="utf-8") as out_f:
    json.dump(output_dev, out_f, ensure_ascii=False, indent=4)
out_f.close()

print("\n-------------\n Processing test dataset: \n")
output_test = process_documents(test_documents)

with open(os.path.join(output_directory, "candidates_test.json"), "w", encoding="utf-8") as out_f:
    json.dump(output_test, out_f, ensure_ascii=False, indent=4)
out_f.close()