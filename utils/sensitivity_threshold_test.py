import json
import csv

thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
json_f = "../camera-ready/ED/MHERCL-ITA/candidates_top50.json"
csv_f = "../camera-ready/ED/MHERCL-ITA/DELICATE-LLM/output.csv"
ground_truth_f = "../../ENEIDE/MHERCL-ITA/annotations_test.csv"

with open(json_f, "r") as f1:
    candidates = json.load(f1)
    candidates_list = []
    for c in candidates:
        candidates_list.extend(c["annotations"])
f1.close()

with open(csv_f, "r") as f2:
    llm_preds = csv.DictReader(f2)
    llm_preds = list(llm_preds)
f2.close()


with open(ground_truth_f, "r") as f3:
    annotations = csv.DictReader(f3)
    annotations = list(llm_preds)
f3.close()



print("Accuracy test of ensemble model at different thresholds:")

for theta in thresholds:
    tp = 0
    for c_dict, llm_pred, true_pred in zip(candidates_list, llm_preds, annotations):
        if c_dict["best_linking"]["rf_score"] >= theta:
            id_predicted = c_dict["best_linking"]["q_id"]
        else:
            id_predicted = llm_pred["identifier"]
        if id_predicted == true_pred["identifier"]:
            tp += 1
    accuracy = tp/len(true_pred)
    print("Threshold: ", theta)
    print("Accuracy: ", accuracy)


