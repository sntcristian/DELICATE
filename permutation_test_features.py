import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from elite.feature_selector import load_json_data, get_training_features
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score

json_data_path = "./DZ_results"
model_directory = "./ELITE_models/GBT"

training_json_path = os.path.join(json_data_path, "candidates_train50.json")
dev_json_path = os.path.join(json_data_path, "candidates_dev50.json")
test_json_path = os.path.join(json_data_path, "candidates_test50.json")

print("\n -------------------------\n Computing features for train data: \n")
train_data = load_json_data(training_json_path)
train_features = get_training_features(train_data, k=5)

print("\n -------------------------\n Computing features for dev data: \n")
dev_data = load_json_data(dev_json_path)
dev_features = get_training_features(dev_data, k=5)

print("\n -------------------------\n Computing features for test data: \n")
test_data = load_json_data(test_json_path)
test_features = get_training_features(test_data, k=5)

df_train = pd.DataFrame(train_features)
df_dev = pd.DataFrame(dev_features)
df_test = pd.DataFrame(test_features)
y_train = df_train['qid_match'].values
y_dev = df_dev['qid_match'].values
y_test = df_test['qid_match'].values
X_train = df_train.drop('qid_match', axis=1)
X_dev = df_dev.drop('qid_match', axis=1)
X_test = df_test.drop('qid_match', axis=1)



final_model = load(os.path.join(model_directory, 'gbt_dz_b50_n10.joblib'))


y_pred_base = final_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_base)
print(f"Accuracy on DZ Test Set (without permutation): {baseline_accuracy:.4f}")


pfi_results = permutation_importance(
    estimator=final_model,
    X=X_test,
    y=y_test,
    n_repeats=30,
    scoring='accuracy',
    random_state=42
)

# order results by importance
feature_names = X_test.columns.tolist()
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
plt.xlabel("Mean Accuracy Loss (Mean Importance)")
plt.title("Permutation Feature Importance (Baseline Accuracy = {:.4f})".format(baseline_accuracy))
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



