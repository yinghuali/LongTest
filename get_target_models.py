import pickle
import joblib
import numpy as np
from read_data import *
from utils import *
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_file_embedding_X", type=str)
ap.add_argument("--path_file_embedding_y", type=str)
ap.add_argument("--path_save_model", type=str)
ap.add_argument("--model_name", type=str)
args = ap.parse_args()


path_file_embedding_X = args.path_file_embedding_X
path_file_embedding_y = args.path_file_embedding_y
path_save_model = args.path_save_model
model_name = args.model_name

# path_file_embedding_X = './data/embedding_data/EURLEX57K_file_X.pkl'
# path_file_embedding_y = './data/embedding_data/EURLEX57K_file_y.pkl'
# path_save_model = './target_models/MiniLM-L6-v2-rf.pkl'
# model_name = 'rf'

# python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-rf.model' --model_name 'rf'
# python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-lr.model' --model_name 'lr'
# python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-xgb.model' --model_name 'xgb'


def select_model(model_name):
    if model_name == 'rf':
        model = RandomForestClassifier(n_estimators=10, max_depth=3)
    elif model_name == 'lr':
        model = LogisticRegression(max_iter=20)
    elif model_name == 'xgb':
        model = XGBClassifier(n_estimators=10)
    return model


def main():

    embedding_vec = pickle.load(open(path_file_embedding_X, 'rb'))
    y = pickle.load(open(path_file_embedding_y, 'rb'))
    print(embedding_vec.shape)
    print(y.shape)

    embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(embedding_vec, y, test_size=0.3, random_state=0)

    model = select_model(model_name)
    model.fit(embedding_train_vec, y_train)
    joblib.dump(model, path_save_model)

    y_pre_test = model.predict(embedding_test_vec)
    y_pre_train = model.predict(embedding_train_vec)
    acc_train = accuracy_score(y_train, y_pre_train)
    acc_test = accuracy_score(y_test, y_pre_test)

    print('acc_train:', acc_train)
    print('acc_test:', acc_test)


if __name__ == '__main__':
    main()

