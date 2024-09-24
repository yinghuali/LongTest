import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# 加载数据集
newsgroups = fetch_20newsgroups(subset='all')
texts = newsgroups.data
labels = newsgroups.target

print(len(texts))

# # 文本预处理
# max_words = 10000
# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
#
# max_len = 200
# data = pad_sequences(sequences, maxlen=max_len)
# labels = to_categorical(np.asarray(labels))
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
