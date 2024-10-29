import pickle
import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--path_save_X", type=str)
ap.add_argument("--num_chunks", type=int)
args = ap.parse_args()


path_data = args.path_data
path_save_X = args.path_save_X
num_chunks = args.num_chunks

# python get_chunk_embedding.py --num_chunks 5 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X.pkl'

# num_chunks = 10
# path_data = './data/EURLEX57K/df_all_EURLEX57K.csv'
# path_save_X = './data/embedding_data/EURLEX57K_chunk_X_10.pkl'


def main():
    df = pd.read_csv(path_data)
    content_list = list(df['content'])

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_embeddings = []  # n_text * n_chunks * dimension_vec

    for i in range(len(content_list)):
        text = content_list[i]
        chunks_list = get_chunks_list(text, num_chunks)
        embeddings = model.encode(chunks_list)
        all_embeddings.append(embeddings)

    all_embeddings = np.array(all_embeddings)

    pickle.dump(all_embeddings, open(path_save_X, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
