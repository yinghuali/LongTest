import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def read_lstm_gru_20news():
    newsgroups = fetch_20newsgroups(subset='all')
    texts = newsgroups.data
    labels = newsgroups.target

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_len = 200
    data = pad_sequences(sequences, maxlen=max_len)
    labels = to_categorical(np.asarray(labels))

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test





