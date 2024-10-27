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

path_file_embedding_X = './data/embedding_data/EURLEX57K_file_X.pkl'
path_file_embedding_y = './data/embedding_data/EURLEX57K_file_y.pkl'


embedding_vec = pickle.load(open(path_file_embedding_X, 'rb'))
y = pickle.load(open(path_file_embedding_y, 'rb'))
print(embedding_vec.shape)
print(y.shape)


embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(embedding_vec, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(n_estimators=10, max_depth=3)
model.fit(embedding_train_vec, y_train)

y_pre_test = model.predict(embedding_test_vec)
y_pre_train = model.predict(embedding_train_vec)
acc_train = accuracy_score(y_train, y_pre_train)
acc_test = accuracy_score(y_test, y_pre_test)
print('acc_train:', acc_train)
print('acc_test:', acc_test)


