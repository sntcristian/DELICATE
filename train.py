from elite.biencoder import encode_mention_from_dict, load_models
from elite.indexer import load_resources, search_index_from_dict
from elite.utils import load_csv_from_directory, shape_result_lookup
from elite.feature_selector import get_training_features
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from joblib import dump
import numpy as np
import os
import argparse



def retrieve_candidates(documents, block_size, biencoder, biencoder_params, indexer, conn):
    output = []
    for doc in documents:
        print("Encoding mentions in document: ", doc["doc_id"])
        doc_with_linking = encode_mention_from_dict(doc, biencoder, biencoder_params)
        doc_with_candidates = search_index_from_dict(doc_with_linking, indexer, conn, top_k=block_size)
        output.append(doc_with_candidates)
    output = shape_result_lookup(output)
    return output





class TrainingLoop:
    def __init__(self, train_data, dev_data, test_data, output_dir, negatives):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.output_dir = output_dir
        self.negatives = negatives
        self.k_neg = negatives // 2 # negatives are split in two sets, hard and easy
        self._prepare_data()

    def _prepare_data(self):
        self.train_features = get_training_features(self.train_data, k=self.k_neg)
        self.dev_features = get_training_features(self.dev_data, k=self.k_neg)
        self.test_features = get_training_features(self.test_data, k=self.k_neg)

        df_train = pd.DataFrame(self.train_features)
        df_dev = pd.DataFrame(self.dev_features)
        df_test = pd.DataFrame(self.test_features)

        self.y_train = df_train['qid_match'].values
        self.y_dev = df_dev['qid_match'].values
        self.y_test = df_test['qid_match'].values

        self.X_train = df_train.drop('qid_match', axis=1)
        self.X_dev = df_dev.drop('qid_match', axis=1)
        self.X_test = df_test.drop('qid_match', axis=1)

    def _objective(self, params):
        clf = GradientBoostingClassifier(**params, random_state=42)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict_proba(self.X_dev)[:, 1]
        auc = roc_auc_score(self.y_dev, preds)
        return {'loss': -auc, 'status': STATUS_OK}

    def optimize_hyperparameters(self, max_evals=100):
        space = {
            'n_estimators': hp.choice('n_estimators', range(50, 501, 50)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'max_depth': hp.choice('max_depth', range(3, 21)),
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.1),
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.1),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
        }

        print("\n -------------------------\n Starting hyper-parameter optimization: \n")
        trials = Trials()
        best_params = fmin(fn=self._objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        best_params['n_estimators'] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500][best_params['n_estimators']]
        best_params['max_depth'] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20][
            best_params['max_depth']]
        best_params['max_features'] = ['sqrt', 'log2', None][best_params['max_features']]
        best_params['learning_rate'] = best_params['learning_rate']

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "params_gbt.txt"), "w") as f:
            f.write(str(best_params))

        print("Best parameters: ", best_params)
        self.best_params = best_params

    def train_and_evaluate(self):
        final_model = GradientBoostingClassifier(**self.best_params, random_state=42)
        final_model.fit(self.X_train, self.y_train)
        final_preds = final_model.predict_proba(self.X_test)[:, 1]
        final_auc = roc_auc_score(self.y_test, final_preds)
        print(f"ROC AUC Score on the Test Set: {final_auc}")
        print("final model available in "+ str(os.path.join(self.output_dir, 'gbt_reranker.joblib')))
        dump(final_model, os.path.join(self.output_dir, 'gbt_reranker.joblib'))

    def run(self):
        self.optimize_hyperparameters()
        self.train_and_evaluate()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop for training GBT reranker")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CSV Entity Linking dataset")
    parser.add_argument("--models_dir", type=str, required=True, help="directory which contains biencoder and knowledge base")
    parser.add_argument("--block_size", type=int, required=True, help="Number of candidates returned by biencoder")
    parser.add_argument("--negatives", type=int, required=True, help="Number of non-matching entities per block")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where GBT model is stored")
    args = parser.parse_args()
    params = vars(args)

    print("Loading documents from", params["dataset_path"])
    train_documents, dev_documents, test_documents = load_csv_from_directory(dataset_path=params["dataset_path"])
    models_path = {
        "db_path": os.path.join(params["models_dir"], "KB/wikipedia_it.sqlite"),
        "index_path": os.path.join(params["models_dir"], "KB/faiss_hnsw_ita_index.pkl"),
        "biencoder_model": os.path.join(params["models_dir"], "blink_biencoder_base_wikipedia_ita/pytorch_model.bin"),
        "biencoder_config": os.path.join(params["models_dir"], "blink_biencoder_base_wikipedia_ita/config.json"),
    }

    print('Loading models from', params["models_dir"])
    biencoder, biencoder_params = load_models(models_path)
    indexer, conn = load_resources(models_path)
    print("Loading complete.")


    train_data = retrieve_candidates(train_documents, 50, biencoder,
                                     biencoder_params, indexer, conn)
    dev_data = retrieve_candidates(dev_documents, 50, biencoder,
                                     biencoder_params, indexer, conn)
    test_data = retrieve_candidates(test_documents, 50, biencoder,
                                   biencoder_params, indexer, conn)

    print("Starting training loop for supervised classifier:")
    training_loop = TrainingLoop(train_data=train_data, dev_data=dev_data, test_data=test_data,
                                 output_dir=params["output_dir"], negatives=params["negatives"])
    training_loop.run()




