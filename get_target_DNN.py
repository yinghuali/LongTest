import pickle
import numpy as np
import os
from read_data import *
from utils import *
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_file_embedding_X", type=str)
ap.add_argument("--path_file_embedding_y", type=str)
ap.add_argument("--path_save_model", type=str)
args = ap.parse_args()


path_file_embedding_X = args.path_file_embedding_X
path_file_embedding_y = args.path_file_embedding_y
path_save_model = args.path_save_model


# path_file_embedding_X = './data/embedding_data/EURLEX57K_file_X.pkl'
# path_file_embedding_y = './data/embedding_data/EURLEX57K_file_y.pkl'
# path_save_model = './target_models/MiniLM-L6-v2-dnn.model'

# python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-dnn.h5'

def build_dnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(16, activation='sigmoid', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def main():

    embedding_vec = pickle.load(open(path_file_embedding_X, 'rb'))
    y = pickle.load(open(path_file_embedding_y, 'rb'))

    embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(embedding_vec, y, test_size=0.2, random_state=0)
    embedding_train_vec, embedding_val_vec, y_train, y_val = train_test_split(embedding_train_vec, y_train, test_size=0.5, random_state=0)

    num_classes = len(set(list(y)))
    input_shape = embedding_vec.shape[1]
    model = build_dnn_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(embedding_train_vec, y_train, epochs=3, batch_size=128)
    model.save(path_save_model)

    y_pre_train = np.argmax(model.predict(embedding_train_vec), axis=1)
    y_pre_test = np.argmax(model.predict(embedding_test_vec), axis=1)

    acc_train = accuracy_score(y_train, y_pre_train)
    acc_test = accuracy_score(y_test, y_pre_test)

    print('acc_train:', acc_train)
    print('acc_test:', acc_test)


if __name__ == '__main__':
    main()

