import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from elite.feature_selector import load_json_data, compute_features
from elite.reranker import get_rf_scores
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

json_data_path = "./retrieval_results"
model_directory = "./ELITE_models/GBT"
threshold_nil = 0.3


def get_features_with_reranker(list_of_results, rf_classifier):
    output = []
    for result_lookup in list_of_results:
        similarity_features = compute_features(result_lookup)
        matching_probabilities = get_rf_scores(rf_classifier, similarity_features)
        zipped_features = zip(matching_probabilities, similarity_features)
        sorted_features = sorted(zipped_features, reverse=True, key=lambda x: x[0])
        best_score, feature = sorted_features[0]
        if best_score < threshold_nil and not result_lookup["identifier"].startswith("Q"):
            feature["qid_match"] = 1
        output.append(feature)
    return output





final_model = load(os.path.join(model_directory, 'gbt_amd_b20_n6.joblib'))

test_json_path = os.path.join(json_data_path, "candidates_test20.json")


test_data = load_json_data(test_json_path)

print("\n -------------------------\n Computing features for final GBT model: \n")
test_features = get_features_with_reranker(test_data, final_model)


df_test = pd.DataFrame(test_features)
y_test = df_test['qid_match'].values
X_test = df_test.drop('qid_match', axis=1)


y_pred_base = final_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_base)
print(f"Accuracy on AMD Test Set (without permutation): {baseline_accuracy:.4f}")


pfi_results = permutation_importance(
    estimator=final_model,
    X=X_test,
    y=y_test,
    n_repeats=30,
    scoring='accuracy',
    random_state=42
)

# order results by importance
feature_names = ["min", "max", "mean", "median", "L2", "levenshtein", "jaccard", "time_delta", "type_match"]
importances_mean = pfi_results.importances_mean
importances_std = pfi_results.importances_std

# get index that sorts features from min to max decrement
sorted_idx = np.argsort(importances_mean)

# plot
plt.figure(figsize=(8, 6))
plt.barh(
    [feature_names[i] for i in sorted_idx],
    importances_mean[sorted_idx],
    xerr=importances_std[sorted_idx],
    align='center',
    color='skyblue',
    ecolor='black'
)
plt.xlabel("Mean Accuracy Loss (Mean Importance) after 30 permutations")
plt.title("Permutation Feature Importance (Baseline Accuracy = {:.4f})".format(baseline_accuracy))
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



