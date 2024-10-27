import pickle
import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer

num_chunks = 10
path_data = './data/EURLEX57K/df_all_EURLEX57K.csv'
path_save_X = './data/embedding_data/EURLEX57K_file_X.pkl'
path_save_y = './data/embedding_data/EURLEX57K_file_y.pkl'


def main():
    df = pd.read_csv(path_data)
    content_list = list(df['content'])
    y_list = list(df['type'])
    content_clean_list = [clean_text(text) for text in content_list]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_embeddings = model.encode(content_clean_list)

    pickle.dump(all_embeddings, open(path_save_X, 'wb'), protocol=4)
    y = np.array(y_list)
    pickle.dump(y, open(path_save_y, 'wb'), protocol=4)




if __name__ == '__main__':
    main()
