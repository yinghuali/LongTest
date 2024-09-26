import pickle
import numpy as np
from tensorflow.keras.models import load_model
from read_data import *
from utils import *
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier

path_model = './save_models/20new_lstm_6.h5'
data_name = '20news'
path_embedding_train = './data/embedding_data/20news_20_train.pkl'
path_embedding_test = './data/embedding_data/20news_20_test.pkl'


def load_data(data_name):
    if data_name == '20news':
        x_train, x_test, y_train, y_test = read_lstm_gru_20news()
        return x_train, x_test, y_train, y_test


def get_miss_label(y_pre, y_label):
    miss_label = []
    for i in range(len(y_pre)):
        if y_pre[i] == y_label[i]:
            miss_label.append(0)
        else:
            miss_label.append(1)
    miss_label_np = np.array(miss_label)
    return miss_label_np


def get_compare_method_apfd(x_test_target_model_pre, idx_miss_test_list):
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(x_test_target_model_pre)
    pcs_rank_idx = PCS_rank_idx(x_test_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    random_apfd = apfd(idx_miss_test_list, random_rank_idx)
    deepGini_apfd = apfd(idx_miss_test_list, deepGini_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_test_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_test_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_test_list, entropy_rank_idx)

    dic = {
        'random_apfd': random_apfd,
        'deepGini_apfd': deepGini_apfd,
        'vanillasoftmax_apfd': vanillasoftmax_apfd,
        'pcs_apfd': pcs_apfd,
        'entropy_apfd': entropy_apfd
    }

    return dic


x_train, x_test, y_train, y_test = load_data(data_name)

model = load_model(path_model, compile=False)
final_layer_train_vec = model.predict(x_train)
final_layer_test_vec = model.predict(x_test)


y_train_label = np.argmax(y_train, axis=1)
y_test_label = np.argmax(y_test, axis=1)

model = load_model(path_model, compile=False)
y_pre_train = model.predict(x_train)
y_pre_test = model.predict(x_test)

y_pre_train_label = np.argmax(y_pre_train, axis=1)
y_pre_test_label = np.argmax(y_pre_test, axis=1)

miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(y_pre_train_label, y_pre_test_label, y_train_label, y_test_label)



embedding_train_vec = pickle.load(open(path_embedding_train, 'rb'))
embedding_test_vec = pickle.load(open(path_embedding_test, 'rb'))


embedding_train_vec = embedding_train_vec.reshape(embedding_train_vec.shape[0], embedding_train_vec.shape[1] * embedding_train_vec.shape[2])
embedding_test_vec = embedding_test_vec.reshape(embedding_test_vec.shape[0], embedding_test_vec.shape[1] * embedding_test_vec.shape[2])

print(embedding_test_vec.shape)
feature_x_train = np.hstack((final_layer_train_vec, embedding_train_vec))
feature_x_test = np.hstack((final_layer_test_vec, embedding_test_vec))

print(feature_x_test.shape)

dic_res = get_compare_method_apfd(final_layer_test_vec, idx_miss_test_list)
print(dic_res)


model = RandomForestClassifier()
model.fit(final_layer_train_vec, miss_train_label)
feature_pre = model.predict_proba(final_layer_test_vec)[:, 1]

feature_rank_idx = feature_pre.argsort()[::-1].copy()
feature_apfd = apfd(idx_miss_test_list, feature_rank_idx)
print(feature_apfd)

model = RandomForestClassifier()
model.fit(feature_x_train, miss_train_label)
feature_pre = model.predict_proba(feature_x_test)[:, 1]

feature_rank_idx = feature_pre.argsort()[::-1].copy()
feature_apfd = apfd(idx_miss_test_list, feature_rank_idx)
print(feature_apfd)










