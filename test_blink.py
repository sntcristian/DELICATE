import csv
from elite.biencoder import encode_mention_from_dict
from elite.indexer import search_index_from_dict
from elite.utils import load_from_config, load_csv_dataset, shape_result_lookup



def process_documents(documents, config_file):
    output = []
    biencoder, biencoder_params, indexer, conn, rf_classifier = load_from_config(config_file)
    for doc in documents:
        print("Encoding mentions in document: ", doc["doc_id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=100)
        annotations_with_candidates = shape_result_lookup(doc_with_candidates)
        for annotation in annotations_with_candidates:
            candidate = annotation["candidates"][0]
            output.append({
                "doc_id":annotation["doc_id"],
                "surface":annotation["surface"],
                "start_pos":annotation["start_pos"],
                "end_pos":annotation["end_pos"],
                "type":annotation["type"],
                "identifier":candidate["q_id"],
                "score":candidate["score"]
            })
    return output



paragraphs_path = "../ENEIDE/DZ/v0.1/paragraphs_test.csv"
annotations_path = "./DZ_results/ner/output.csv"

input_data = load_csv_dataset(paragraphs_path, annotations_path)

results_rerankered = process_documents(input_data, "config.json")

with open("DZ_results/blink/output.csv", "w", encoding="utf-8") as out_f:
    dict_writer = csv.DictWriter(out_f, results_rerankered[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(results_rerankered)

