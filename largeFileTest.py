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

path_chunk_embedding_X = './data/embedding_data/EURLEX57K_chunk_X.pkl'
path_chunk_embedding_y = './data/embedding_data/EURLEX57K_chunk_y.pkl'
path_target_model = './target_models/MiniLM-L6-v2-rf.model'





model = joblib.load(path_target_model)

chunk_embedding_X = pickle.load(open(path_chunk_embedding_X, 'rb'))
y = pickle.load(open(path_chunk_embedding_y, 'rb'))

embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(chunk_embedding_X, y, test_size=0.2, random_state=0)
#
# y_pre_train = model.predict(embedding_train_vec)
# y_pre_test = model.predict(embedding_test_vec)


print(embedding_train_vec.shape)
print(y_train.shape)


path_file_embedding_X = './data/embedding_data/EURLEX57K_file_X.pkl'
path_file_embedding_y = './data/embedding_data/EURLEX57K_file_y.pkl'



file_embedding_X = pickle.load(open(path_file_embedding_X, 'rb'))
print(file_embedding_X.shape)
file_embedding_y = pickle.load(open(path_file_embedding_y, 'rb'))
print(file_embedding_y.shape)


