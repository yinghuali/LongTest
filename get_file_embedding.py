import pickle
import numpy as np
from read_data import *
from utils import *
from sentence_transformers import SentenceTransformer

data_name = '20news'
path_save_train = './data/embedding_file_data/20news_20_train_all-MiniLM-L6-v2.pkl'
path_save_test = './data/embedding_file_data/20news_20_test_all-MiniLM-L6-v2.pkl'
transformer_name = 'all-MiniLM-L6-v2'


def select_data(data_name):
    if data_name == '20news':
        texts_train, texts_test, y_train, y_test = read_normal_20news()
        return texts_train, texts_test, y_train, y_test


def main():
    texts_train, texts_test, y_train, y_test = select_data(data_name)

    model = SentenceTransformer(transformer_name)
    texts_train_list = [clean_text(text) for text in texts_train]
    texts_test_list = [clean_text(text) for text in texts_test]

    embeddings_train = model.encode(texts_train_list)
    embeddings_test = model.encode(texts_test_list)

    pickle.dump(embeddings_train, open(path_save_train, 'wb'), protocol=4)
    pickle.dump(embeddings_test, open(path_save_test, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
