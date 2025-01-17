import csv
from elite.biencoder import encode_mention_from_dict, load_models
from elite.indexer import load_resources, search_index_from_dict
from elite.utils import load_csv_dataset, shape_result_lookup, get_rf_scores
from elite.feature_selector import compute_features
import os
from sklearn.ensemble import RandomForestClassifier
from joblib import load



models_path = "ELITE_models"
paragraphs_path = "../ENEIDE/DZ/v0.1/paragraphs_test.csv"
annotations_path = "../ENEIDE/DZ/v0.1/annotations_test.csv"

params = {
    "db_path": os.path.join(models_path, "wikipedia_it.sqlite"),
    "index_path": os.path.join(models_path, "faiss_hnsw_ita_index.pkl"),
    "biencoder_model": os.path.join(models_path, "blink_biencoder_base_wikipedia_ita/pytorch_model.bin"),
    "biencoder_config": os.path.join(models_path, "blink_biencoder_base_wikipedia_ita/config.json"),
    "rf_classifier_path": os.path.join(models_path, "rf_classifier.joblib")
}

print('Loading biencoder...')
biencoder, biencoder_params = load_models(params)
print('Device:', biencoder.device)
print('Loading complete.')

print("Loading index and database...")
indexer, conn = load_resources(params)
print("Loading complete.")


rf_classifier = load(params["rf_classifier_path"])
print("Loading reranker complete.")


input_data = load_csv_dataset(paragraphs_path, annotations_path)



def process_documents(documents):
    output = []
    rf_threshold = 0.2

    for doc in documents:
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)
        output.append(doc_with_candidates)

    annotations_with_candidates = shape_result_lookup(output)
    results_rerankered = []
    for annotation in annotations_with_candidates:
        data_for_features = [annotation]
        similarity_features = compute_features(data_for_features)
        matching_probabilities = get_rf_scores(rf_classifier, similarity_features)
        candidate_id = "NIL"
        candidate_title = ""
        max_score = 0
        for candidate, rf_score in zip(annotation["candidates"], matching_probabilities):
            if rf_score >= rf_threshold and rf_score > max_score:
                candidate_id = candidate["q_id"]
                candidate_title = candidate["title"]
                max_score = rf_score
        item = {"doc_id": annotation["doc_id"], "start_pos": annotation["start_pos"], "end_pos": annotation["end_pos"],
                "surface": annotation["surface"], "type": annotation["type"], "identifier": candidate_id,
                "wiki_title": candidate_title, "score": max_score}
        results_rerankered.append(item)
    return results_rerankered




def process_documents_no_reranker(documents):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=10)
        output.append(doc_with_candidates)
    annotations_with_candidates = shape_result_lookup(output)
    results_no_reranker = list()
    for annotation in annotations_with_candidates:
        candidate_id = annotation["candidates"][0]["q_id"]
        wiki_title = annotation["candidates"][0]["title"]
        score = annotation["candidates"][0]["score"]
        item = {"doc_id":annotation["doc_id"], "start_pos":annotation["start_pos"], "end_pos":annotation["end_pos"],
                "surface": annotation["surface"], "type":annotation["type"], "identifier":candidate_id,
                "wiki_title":wiki_title, "score":score}
        results_no_reranker.append(item)
    return results_no_reranker

results_rerankered = process_documents(input_data)

with open("DZ_results/output.csv", "w", encoding="utf-8") as out_f:
    dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(results_rerankered)

