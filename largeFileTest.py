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

file_embedding_X = pickle.load(open(path_file_embedding_X, 'rb'))  # (57000, 384)
y = pickle.load(open(path_file_y, 'rb'))
# model = joblib.load(path_target_model)




chunk_embedding_X = pickle.load(open(path_chunk_embedding_X, 'rb')) # (57000, 5, 384)
chunk_embedding_X = chunk_embedding_X.reshape(chunk_embedding_X.shape[0], chunk_embedding_X.shape[1] * chunk_embedding_X.shape[2])
pca = PCA(n_components=128)
chunk_embedding_X = pca.fit_transform(chunk_embedding_X)


print(file_embedding_X.shape)
print(chunk_embedding_X.shape)





