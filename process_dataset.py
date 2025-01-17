from elite.biencoder import encode_mention_from_dict, load_models
from elite.indexer import load_resources, search_index_from_dict
from elite.utils import load_csv_from_directory, shape_result_lookup
import os
import json

models_path = "ELITE_models"
dataset_path = "../ENEIDE/DZ/v0.1"
output_directory = "DZ_results"

params = {
    "db_path": os.path.join(models_path, "wikipedia_it.sqlite"),
    "index_path": os.path.join(models_path, "faiss_hnsw_ita_index.pkl"),
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




def process_documents(documents):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)
        output.append(doc_with_candidates)
    output = shape_result_lookup(output)
    return output


train_documents, dev_documents, test_documents = load_csv_from_directory(dataset_path=dataset_path)


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