import pickle
import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer

num_chunks = 10
path_data = './data/EURLEX57K/df_all_EURLEX57K.csv'
path_save_X = './data/embedding_data/EURLEX57K_chunk_X.pkl'
path_save_y = './data/embedding_data/EURLEX57K_chunk_y.pkl'


def main():
    df = pd.read_csv(path_data)
    content_list = list(df['content'])
    y_list = list(df['y'])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_embeddings = []  # n_text * n_chunks * dimension_vec
    for text in content_list:
        chunks_list = get_chunks_list(text, num_chunks)
        embeddings = model.encode(chunks_list)
        all_embeddings.append(embeddings)
    all_embeddings = np.array(all_embeddings)
    pickle.dump(all_embeddings, open(path_save_X, 'wb'), protocol=4)
    y = np.array(y_list)
    pickle.dump(y, open(path_save_y, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
