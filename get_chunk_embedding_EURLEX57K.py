import pickle
import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--path_save_X", type=str)
ap.add_argument("--path_save_ID", type=str)
ap.add_argument("--num_chunks", type=int)
args = ap.parse_args()


path_data = args.path_data
path_save_X = args.path_save_X
path_save_ID = args.path_save_ID
num_chunks = args.num_chunks

# python get_chunk_embedding_EURLEX57K.py --num_chunks 5 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X.pkl' --path_save_ID './data/embedding_data/EURLEX57K_chunk_ID.pkl'

# num_chunks = 10
# path_data = './data/EURLEX57K/df_all_EURLEX57K.csv'
# path_save_X = './data/embedding_data/EURLEX57K_chunk_X_10.pkl'
# path_save_ID = './data/embedding_data/EURLEX57K_chunk_ID_10.pkl'


def main():
    df = pd.read_csv(path_data)
    content_list = list(df['content'])
    ID_list = list(df['ID'])

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_embeddings = []  # n_text * n_chunks * dimension_vec
    res_ID_list = []

    for i in range(len(content_list)):
        text = content_list[i]
        chunks_list = get_chunks_list(text, num_chunks)
        embeddings = model.encode(chunks_list)

        all_embeddings.append(embeddings)
        res_ID_list.append(ID_list[i])

    all_embeddings = np.array(all_embeddings)
    res_ID_list = np.array(res_ID_list)

    pickle.dump(all_embeddings, open(path_save_X, 'wb'), protocol=4)

    pickle.dump(res_ID_list, open(path_save_ID, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
