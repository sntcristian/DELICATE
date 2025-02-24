from elite.biencoder import encode_mention_from_dict, load_models
from elite.indexer import load_resources, search_index_from_dict
from elite.utils import load_csv_from_directory, shape_result_lookup
import os
import json

models_path = "ELITE_models"
dataset_path = "../ENEIDE/DZ/v0.1"
output_directory = "DZ_results"

params = {
    "db_path": os.path.join(models_path, "KB/wikipedia_it.sqlite"),
    "index_path": os.path.join(models_path, "KB/faiss_hnsw_ita_index.pkl"),
    "biencoder_model": os.path.join(models_path, "blink_biencoder_base_wikipedia_ita/pytorch_model.bin"),
    "biencoder_config": os.path.join(models_path, "blink_biencoder_base_wikipedia_ita/config.json"),
}

print('Loading biencoder...')
biencoder, biencoder_params = load_models(params)
print('Device:', biencoder.device)
print('Loading complete.')

print("Loading index and database...")
indexer, conn = load_resources(params)
print("Loading complete.")

all_k = [20]


def process_documents(documents, k):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["doc_id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=k)
        output.append(doc_with_candidates)
    output = shape_result_lookup(output)
    tot_entities = 0
    found_entities = 0
    for item in output:
        if item["identifier"].startswith("Q"):
            entity = item["identifier"]
            tot_entities += 1
            candidate_ids = [candidate["q_id"] for candidate in item["candidates"]]
            candidate_ids = set(candidate_ids)
            if entity in candidate_ids:
                found_entities += 1
    recall = found_entities / tot_entities
    return output, recall


train_documents, dev_documents, test_documents = load_csv_from_directory(dataset_path=dataset_path)
result_summary = list()

for k in all_k:

    print("\n-------------\n Processing train dataset for k = "+str(k)+": \n")
    output_train, recall_train = process_documents(train_documents, k)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(os.path.join(output_directory, "candidates_train"+str(k)+".json"), "w", encoding="utf-8") as out_f:
        json.dump(output_train, out_f, ensure_ascii=False, indent=4)
    out_f.close()


    print("\n-------------\n Processing dev dataset for k = "+str(k)+": \n")
    output_dev, recall_dev = process_documents(dev_documents, k)

    with open(os.path.join(output_directory, "candidates_dev"+str(k)+".json"), "w", encoding="utf-8") as out_f:
        json.dump(output_dev, out_f, ensure_ascii=False, indent=4)
    out_f.close()

    print("\n-------------\n Processing test dataset for k = "+str(k)+": \n")
    output_test, recall_test = process_documents(test_documents, k)

    with open(os.path.join(output_directory, "candidates_test"+str(k)+".json"), "w", encoding="utf-8") as out_f:
        json.dump(output_test, out_f, ensure_ascii=False, indent=4)
    out_f.close()

    result_summary.append({
        "k": k,
        "recall_train":recall_train,
        "recall_dev":recall_dev,
        "recall_test":recall_test,
        "avg_recall": (recall_train+recall_dev+recall_test) / 3
    })

with open(os.path.join(output_directory, "result_summary2.json"), "w", encoding="utf-8") as out_f:
    json.dump(result_summary, out_f, ensure_ascii=False, indent=4)
out_f.close()
