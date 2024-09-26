import pickle
import numpy as np
from read_data import *
from utils import *
from sentence_transformers import SentenceTransformer

data_name = '20news'
num_chunks = 20
path_save = './data/embedding_data/20news_20.pkl'


def select_data(data_name):
    if data_name == '20news':
        texts_train, texts_test, y_train, y_test = read_normal_20news()
        return texts_train, texts_test, y_train, y_test


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts_train, texts_test, y_train, y_test = select_data(data_name)
    all_embeddings = []  # n_text * n_chunks * dimension_vec
    for text in texts_train:
        chunks_list = get_chunks_list(text, num_chunks)
        embeddings = model.encode(chunks_list)
        all_embeddings.append(embeddings)
    all_embeddings = np.array(all_embeddings)

    pickle.dump(all_embeddings, open(path_save, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
