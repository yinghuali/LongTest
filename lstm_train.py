from read_data import *
from get_models import *
from tensorflow.keras.models import save_model
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ap = argparse.ArgumentParser()
ap.add_argument("--data_name", type=str)
ap.add_argument("--epochs", type=int)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--n_classes", type=int)
ap.add_argument("--save_model_path", type=str)
args = ap.parse_args()

data_name = args.data_name
epochs = args.epochs
batch_size = args.batch_size
n_classes = args.n_classes
save_model_path = args.save_model_path

# python lstm_train.py --data_name '20news' --epochs 6 --batch_size 64 --n_classes 20 --save_model_path './save_models/20new_lstm_6.h5'


def load_data(data_name):
    if data_name == '20news':
        x_train, x_test, y_train, y_test = read_lstm_gru_20news()
        return x_train, x_test, y_train, y_test


def main():
    x_train, x_test, y_train, y_test = load_data(data_name)
    model = get_lstm_model(n_classes)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    loss, accuracy = model.evaluate(x_train, y_train, batch_size=batch_size)
    print('train acc:', accuracy)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test acc:', accuracy)
    save_model(model, save_model_path)


if __name__ == '__main__':
    main()