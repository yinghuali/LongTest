import pickle
import numpy as np
from tensorflow.keras.models import load_model
from read_data import *
from utils import *
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data_name = '20news'
path_file_embedding_train = './data/embedding_file_data/20news_train_all-MiniLM-L6-v2.pkl'
path_file_embedding_test = './data/embedding_file_data/20news_test_all-MiniLM-L6-v2.pkl'


def load_select_data(data_name):
    if data_name == '20news':
        texts_train, texts_test, y_train, y_test = read_normal_20news()
        return texts_train, texts_test, y_train, y_test


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

texts_train, texts_test, y_train, y_test = load_select_data(data_name)

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

embedding_train_vec = pickle.load(open(path_file_embedding_train, 'rb'))
embedding_test_vec = pickle.load(open(path_file_embedding_test, 'rb'))


embedding_train_vec, embedding_val_vec, y_train, y_val = train_test_split(embedding_train_vec, y_train, test_size=0.3, random_state=0)

model = LogisticRegression()
model.fit(embedding_train_vec, y_train)

y_pre_val = model.predict(embedding_val_vec)
y_pre_test = model.predict(embedding_test_vec)
final_feature_val = model.predict_proba(embedding_val_vec)
final_feature_test = model.predict_proba(embedding_test_vec)


y_pre_train = model.predict(embedding_train_vec)

acc = accuracy_score(y_train, y_pre_train)
print('acc:', acc)
acc = accuracy_score(y_val, y_pre_val)
print('acc:', acc)
acc = accuracy_score(y_test, y_pre_test)
print('acc:', acc)


miss_val_label, miss_test_label, idx_miss_test_list = get_miss_lable(y_pre_val, y_pre_test, y_val, y_test)
model = XGBClassifier()
model.fit(final_feature_val, miss_val_label)
feature_pre = model.predict_proba(final_feature_test)[:, 1]
feature_rank_idx = feature_pre.argsort()[::-1].copy()
feature_apfd = apfd(idx_miss_test_list, feature_rank_idx)
print(feature_apfd)

dic_res = get_compare_method_apfd(final_feature_test, idx_miss_test_list)
print(dic_res)

