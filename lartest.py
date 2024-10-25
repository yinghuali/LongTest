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

path_file_embedding_X= './data/embedding_data/EURLEX57K_file_X.pkl'
path_file_embedding_y = './data/embedding_data/EURLEX57K_file_y.pkl'


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


embedding_vec = pickle.load(open(path_file_embedding_X, 'rb'))
y = pickle.load(open(path_file_embedding_y, 'rb'))


embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(embedding_vec, y, test_size=0.3, random_state=0)

model = RandomForestClassifier()
model.fit(embedding_train_vec, y_train)

y_pre_test = model.predict(embedding_test_vec)
final_feature_test = model.predict_proba(embedding_test_vec)


y_pre_train = model.predict(embedding_train_vec)
acc = accuracy_score(y_train, y_pre_train)
print('acc:', acc)

acc = accuracy_score(y_test, y_pre_test)
print('acc:', acc)


miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(y_pre_train, y_pre_test, y_train, y_test)

model = XGBClassifier()
feature_pre = model.predict_proba(final_feature_test)[:, 1]
feature_rank_idx = feature_pre.argsort()[::-1].copy()
feature_apfd = apfd(idx_miss_test_list, feature_rank_idx)
print(feature_apfd)

dic_res = get_compare_method_apfd(final_feature_test, idx_miss_test_list)
print(dic_res)

