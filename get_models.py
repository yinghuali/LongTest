from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def get_lstm_model(n_classes):
    embedding_dim = 128
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=200))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


