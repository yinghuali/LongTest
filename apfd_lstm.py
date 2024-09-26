import os
from tensorflow.keras.models import load_model
from read_data import *
from utils import *
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_model = './save_models/20new_lstm_6.h5'
data_name = '20news'
num_chunks = 20


def select_data(data_name):
    if data_name == '20news':
        texts_train, texts_test, y_train, y_test = read_normal_20news()
        return texts_train, texts_test, y_train, y_test


texts_train, texts_test, y_train, y_test = select_data(data_name)


chunks_list = get_chunks_list(texts_train[1], num_chunks)

print(texts_train[1])
print('====')
print(chunks_list)




# model = load_model(path_model, compile=False)
# pre = model.predict(x_test)
# print(pre.shape)



# print(texts_train[0])
#
# print(len(texts_train))