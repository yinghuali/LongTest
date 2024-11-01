import pickle
import joblib
import numpy as np
from read_data import *
from utils import *
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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


def select_model(model_name):
    if model_name == 'rf':
        model = RandomForestClassifier(n_estimators=5, max_depth=5)
    elif model_name == 'lr':
        model = LogisticRegression(max_iter=4)
    elif model_name == 'dt':
        model = DecisionTreeClassifier(max_depth=6)
    return model


def main():

    embedding_vec = pickle.load(open(path_file_embedding_X, 'rb'))
    y = pickle.load(open(path_file_embedding_y, 'rb'))
    print(embedding_vec.shape)
    print(y.shape)

    embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(embedding_vec, y, test_size=0.2, random_state=0)
    embedding_train_vec, embedding_val_vec, y_train, y_val = train_test_split(embedding_train_vec, y_train, test_size=0.5, random_state=0)

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

