from delicate.ner import load_ner_model, get_mentions_with_ner
import csv
import os
import argparse



def main(config_file, documents, threshold_ner, tagset):
    output = []
    ner_model = load_ner_model(config_file)
    for doc in documents:
        print("Detecting entities in document: ", doc["doc_id"])
        doc_with_mentions = get_mentions_with_ner(doc, ner_model, tagset, threshold_ner)
        output.extend(doc_with_mentions["annotations"])
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BLINK Retriever for Entity Disambiguation")
    parser.add_argument("--documents", type=str, required=True, help="Path to CSV containing documents.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--output_dir", type=str, default="./", help="Path to output directory.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for GliNER model.")
    parser.add_argument("--tagset", type=str, default="DZ", choices=["DZ", "AMD"],
                        help="Tagset to be used: 'DZ' or 'AMD'")
    args = parser.parse_args()
    params = vars(args)



    with open(params["documents"], "r", encoding="utf-8") as f:
        paragraphs = csv.DictReader(f)
        paragraphs = list(paragraphs)
    f.close()



    result = main(params["config"], paragraphs, params["threshold"], params["tagset"])

    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    with open(os.path.join(params["output_dir"], "output.csv"), "w", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, result[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(result)
