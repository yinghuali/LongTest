from tensorflow.keras.models import load_model
from read_data import *

path_model = './save_models/20new_lstm_6.h5'
data_name = '20news'


def select_data(data_name):
    if data_name == '20news':
        x_train, x_test, y_train, y_test = read_lstm_gru_20news()
        return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = select_data(data_name)

model = load_model(path_model)
pre = model.predict(x_test)
print(pre.shape)
