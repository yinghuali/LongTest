import pickle
import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--path_save_X", type=str)
ap.add_argument("--path_save_y", type=str)
ap.add_argument("--embedding_approach", type=str)
args = ap.parse_args()


path_data = args.path_data
path_save_X = args.path_save_X
path_save_y = args.path_save_y
embedding_approach = args.embedding_approach


def main():
    df = pd.read_csv(path_data)
    content_list = list(df['content'])
    y_list = list(df['type'])
    content_clean_list = [clean_text(text) for text in content_list]

    model = SentenceTransformer(embedding_approach)
    all_embeddings = model.encode(content_clean_list)

    pickle.dump(all_embeddings, open(path_save_X, 'wb'), protocol=4)
    y = np.array(y_list)
    pickle.dump(y, open(path_save_y, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
