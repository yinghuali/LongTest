import pickle
import random
import numpy as np
import json
import joblib
from read_data import *
from utils import *
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_file_embedding_X", type=str)
ap.add_argument("--path_file_y", type=str)
ap.add_argument("--path_chunk_embedding_X", type=str)
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--path_save_res", type=str)
args = ap.parse_args()

path_file_embedding_X = args.path_file_embedding_X
path_file_y = args.path_file_y
path_chunk_embedding_X = args.path_chunk_embedding_X
path_target_model = args.path_target_model
path_save_res = args.path_save_res


def get_pairs_train(miss_train_label, chunk_embedding_train_vec):
    wrong_id_list, correct_id_list = get_select_id(miss_train_label)
    label_train = []
    pairs_train_left = []
    pairs_train_right = []
    for i in range(len(chunk_embedding_train_vec)):
        if miss_train_label[i] == 1:
            pairs_train_left.append(chunk_embedding_train_vec[i])
            pairs_train_right.append(chunk_embedding_train_vec[random.choice(correct_id_list)])
            label_train.append(0)

            pairs_train_left.append(chunk_embedding_train_vec[i])
            pairs_train_right.append(chunk_embedding_train_vec[random.choice(wrong_id_list)])
            label_train.append(1)
        else:
            pairs_train_left.append(chunk_embedding_train_vec[i])
            pairs_train_right.append(chunk_embedding_train_vec[random.choice(correct_id_list)])
            label_train.append(1)

            pairs_train_left.append(chunk_embedding_train_vec[i])
            pairs_train_right.append(chunk_embedding_train_vec[random.choice(wrong_id_list)])
            label_train.append(0)

    label_train = np.array(label_train)
    pairs_train_left = np.array(pairs_train_left)
    pairs_train_right = np.array(pairs_train_right)

    return label_train, pairs_train_left, pairs_train_right


def get_contrastive_model():
    input_dim = 128

    def create_base_network(input_dim):
        input = Input(shape=(input_dim,))
        x = Dense(128, activation='sigmoid')(input)
        x = Dense(128, activation='sigmoid')(x)
        x = Dense(128, activation='sigmoid')(x)
        return Model(input, x)

    def contrastive_loss(y_true, y_pred):
        margin = 3
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    base_network = create_base_network(input_dim)

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(lambda embeddings: K.sqrt(K.sum(K.square(embeddings[0] - embeddings[1]), axis=-1, keepdims=True)))([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(loss=contrastive_loss, optimizer='adam')
    return model, base_network


def get_compare_method_apfd(x_test_target_model_pre, idx_miss_test_list):
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(x_test_target_model_pre)
    pcs_rank_idx = PCS_rank_idx(x_test_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    random_apfd = apfd(idx_miss_test_list, random_rank_idx)
    deepGini_apfd = apfd(idx_miss_test_list, deepGini_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_test_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_test_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_test_list, entropy_rank_idx)

    dic = {
        'random_apfd': random_apfd,
        'deepGini_apfd': deepGini_apfd,
        'vanillasoftmax_apfd': vanillasoftmax_apfd,
        'pcs_apfd': pcs_apfd,
        'entropy_apfd': entropy_apfd
    }

    return dic


def main():
    file_embedding_X = pickle.load(open(path_file_embedding_X, 'rb'))
    y = pickle.load(open(path_file_y, 'rb'))

    chunk_embedding_X = pickle.load(open(path_chunk_embedding_X, 'rb'))
    chunk_embedding_X = chunk_embedding_X.reshape(chunk_embedding_X.shape[0], chunk_embedding_X.shape[1] * chunk_embedding_X.shape[2])
    pca = PCA(n_components=128)
    chunk_embedding_X = pca.fit_transform(chunk_embedding_X)

    target_model = joblib.load(path_target_model)
    y_pre = target_model.predict(file_embedding_X)

    embedding_train_vec, embedding_test_vec, y_train, y_test = train_test_split(file_embedding_X, y, test_size=0.2, random_state=0)
    y_pre_train = target_model.predict(embedding_train_vec)
    y_pre_test = target_model.predict(embedding_test_vec)

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(y_pre_train, y_pre_test, y_train, y_test)

    chunk_embedding_train_vec, chunk_embedding_test_vec, _, _ = train_test_split(chunk_embedding_X, y, test_size=0.2, random_state=0)

    label_train, pairs_train_left, pairs_train_right = get_pairs_train(miss_train_label, chunk_embedding_train_vec)

    contrastive_model, base_network = get_contrastive_model()

    contrastive_model.fit([pairs_train_left, pairs_train_right], label_train, batch_size=64, epochs=20)

    chunk_embedding_train_vec = base_network.predict(chunk_embedding_train_vec)
    chunk_embedding_test_vec = base_network.predict(chunk_embedding_test_vec)

    final_feature_test = target_model.predict_proba(embedding_test_vec)
    dic_res = get_compare_method_apfd(final_feature_test, idx_miss_test_list)

    model = RandomForestClassifier()

    max_float32 = np.finfo(np.float32).max
    chunk_embedding_train_vec = np.where(chunk_embedding_train_vec > max_float32, max_float32, chunk_embedding_train_vec)
    chunk_embedding_train_vec = np.where(chunk_embedding_train_vec < -max_float32, -max_float32, chunk_embedding_train_vec)
    chunk_embedding_test_vec = np.where(chunk_embedding_test_vec > max_float32, max_float32, chunk_embedding_test_vec)
    chunk_embedding_test_vec = np.where(chunk_embedding_test_vec < -max_float32, -max_float32, chunk_embedding_test_vec)

    model.fit(chunk_embedding_train_vec, miss_train_label)
    model_pre = model.predict_proba(chunk_embedding_test_vec)[:, 1]

    model_rank_idx = model_pre.argsort()[::-1].copy()
    model_apfd = apfd(idx_miss_test_list, model_rank_idx)
    dic_res['model'] = model_apfd
    print(dic_res)
    json.dump(dic_res, open(path_save_res, 'w'))

    # def build_dnn_model(input_shape):
    #     model = Sequential()
    #     model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(1, activation='sigmoid'))
    #
    #     return model
    #
    # model = build_dnn_model(chunk_embedding_train_vec.shape[1])
    # model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(chunk_embedding_train_vec, miss_train_label, epochs=20, batch_size=32)
    # model_pre = model.predict(chunk_embedding_test_vec).flatten()
    # model_rank_idx = model_pre.argsort()[::-1].copy()
    # model_apfd = apfd(idx_miss_test_list, model_rank_idx)
    # dic_res['model'] = model_apfd
    # print(dic_res)
    # json.dump(dic_res, open(path_save_res, 'w'))


if __name__ == '__main__':
    main()
