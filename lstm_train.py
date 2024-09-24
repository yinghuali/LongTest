from read_data import *
from get_models import *

epochs = 5
batch_size = 64
n_classes = 20

def main():
    x_train, x_test, y_train, y_test = read_lstm_gru_20news()
    model = get_lstm_model(n_classes)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('测试集损失值:', loss)
    print('测试集准确率:', accuracy)


if __name__ == '__main__':
    main()