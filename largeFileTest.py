import pickle
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from read_data import *
from utils import *
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

path_file_embedding_X = './data/embedding_data/EURLEX57K_file_X.pkl'
path_file_y = './data/embedding_data/EURLEX57K_file_y.pkl'
path_chunk_embedding_X = './data/embedding_data/EURLEX57K_chunk_X_5.pkl'
path_target_model = './target_models/MiniLM-L6-v2-rf.model'


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


file_embedding_X = pickle.load(open(path_file_embedding_X, 'rb'))  # (57000, 384)
y = pickle.load(open(path_file_y, 'rb'))

chunk_embedding_X = pickle.load(open(path_chunk_embedding_X, 'rb')) # (57000, 5, 384)
chunk_embedding_X = chunk_embedding_X.reshape(chunk_embedding_X.shape[0], chunk_embedding_X.shape[1] * chunk_embedding_X.shape[2])
pca = PCA(n_components=128)
chunk_embedding_X = pca.fit_transform(chunk_embedding_X)

target_model = joblib.load(path_target_model)
y_pre = target_model.predict(file_embedding_X)

embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(file_embedding_X, y, test_size=0.2, random_state=0)
y_pre_train = target_model.predict(embedding_train_vec)
y_pre_test = target_model.predict(embedding_test_vec)

miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(y_pre_train, y_pre_test, y_train, y_test)


chunk_embedding_train_vec, chunk_embedding_test_vec, _, _ = train_test_split(chunk_embedding_X, y, test_size=0.2, random_state=0)





print(chunk_embedding_train_vec.shape)

model = RandomForestClassifier()
model.fit(chunk_embedding_train_vec, miss_train_label)
model_pre = model.predict_proba(chunk_embedding_test_vec)[:, 1]
model_rank_idx = model_pre.argsort()[::-1].copy()
model_apfd = apfd(idx_miss_test_list, model_rank_idx)
print(model_apfd)

final_feature_test = target_model.predict_proba(embedding_test_vec)
dic_res = get_compare_method_apfd(final_feature_test, idx_miss_test_list)
print(dic_res)




