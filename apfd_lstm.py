import numpy as np
from tensorflow.keras.models import load_model
from read_data import *
from utils import *


path_model = './save_models/20new_lstm_6.h5'
data_name = '20news'
path_embedding =

def select_data(data_name):
    if data_name == '20news':
        texts_train, texts_test, y_train, y_test = read_normal_20news()
        return texts_train, texts_test, y_train, y_test


def get_miss_label(y_pre, y_label):
    miss_label = []
    for i in range(len(y_pre)):
        if y_pre[i] == y_label[i]:
            miss_label.append(0)
        else:
            miss_label.append(1)
    miss_label_np = np.array(miss_label)
    return miss_label_np


x_train, x_test, y_train, y_test = read_lstm_gru_20news()

y_train_label = np.argmax(y_train, axis=1)
y_test_label = np.argmax(y_test, axis=1)

model = load_model(path_model, compile=False)
y_pre_train = model.predict(x_train)
y_pre_test = model.predict(x_test)

y_pre_train_label = np.argmax(y_pre_train, axis=1)
y_pre_test_label = np.argmax(y_pre_test, axis=1)


miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(y_pre_train_label, y_pre_test_label, y_train_label, y_train_label)


