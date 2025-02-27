from elite.ner import load_ner_model, get_mentions_with_ner
import csv
import os


paragraphs_path = "../ENEIDE/DZ/v0.1/paragraphs_test.csv"
config_file = "config.json"
tagset = "DZ"
threshold_ner = 0.9
output_path = "./DZ_results/ner/"



def main(config_file, documents, threshold_ner, tagset):
    output = []
    ner_model = load_ner_model(config_file)
    for doc in documents:
        print("Detecting entities in document: ", doc["doc_id"])
        doc_with_mentions = get_mentions_with_ner(doc, ner_model, tagset, threshold_ner)
        output.extend(doc_with_mentions["annotations"])
    return output




with open(paragraphs_path, "r", encoding="utf-8") as f:
    paragraphs = csv.DictReader(f)
    paragraphs = list(paragraphs)
f.close()



result = main(config_file, paragraphs, threshold_ner, tagset)

with open(os.path.join(output_path, "output.csv"), "w", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, result[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(result)
