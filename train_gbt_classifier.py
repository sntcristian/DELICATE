from elite.feature_selector import load_json_data, get_training_features
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from joblib import dump
import numpy as np

json_data_path = "./DZ_results"
output_directory = "./ELITE_models/GBT"

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

# Definizione dello spazio di ricerca per l'ottimizzazione degli iperparametri
space = {
    'n_estimators': hp.choice('n_estimators', range(50, 501, 50)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.choice('max_depth', range(3, 21)),
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.1),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
}

# Funzione obiettivo per l'ottimizzazione
def objective(params):
    clf = GradientBoostingClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_dev)[:, 1]
    auc = roc_auc_score(y_dev, preds)
    return {'loss': -auc, 'status': STATUS_OK}

# Avvio dell'ottimizzazione degli iperparametri
print("\n -------------------------\n Starting hyper-parameter optimization: \n")
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Adattamento dei parametri selezionati per l'utilizzo
best_params['n_estimators'] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500][best_params['n_estimators']]
best_params['max_depth'] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20][best_params['max_depth']]
best_params['max_features'] = ['sqrt', 'log2', None][best_params['max_features']]
best_params['learning_rate'] = best_params['learning_rate']

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(os.path.join(output_directory, "params_gbt_dz_b50_n10_no_stddev.txt"), "w") as f:
    f.write(str(best_params))
f.close()
print("Best parameters: ", best_params)

final_model = GradientBoostingClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
final_preds = final_model.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, final_preds)
print(f"ROC AUC Score on the Test Set: {final_auc}")

dump(final_model, os.path.join(output_directory, 'gbt_dz_b50_n10_no_stddev.joblib'))
