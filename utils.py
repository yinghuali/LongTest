import re
import numpy as np


def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text


def split_text_into_chunks(text, num_chunks):
    words = text.split()
    chunk_size = len(words) // num_chunks
    remainder = len(words) % num_chunks

    chunks_list = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        sentence = ' '.join(words[start:end])
        sentence = sentence.strip()
        chunks_list.append(sentence)
        start = end

    return chunks_list


def get_chunks_list(text, num_chunks):
    text_clean = clean_text(text)
    chunks_list = split_text_into_chunks(text_clean, num_chunks)
    return chunks_list


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def get_miss_lable(target_train_pre, target_test_pre, y_train, y_test):
    idx_miss_train_list = get_idx_miss_class(target_train_pre, y_train)
    idx_miss_test_list = get_idx_miss_class(target_test_pre, y_test)
    miss_train_label = [0]*len(y_train)
    for i in idx_miss_train_list:
        miss_train_label[i]=1
    miss_train_label = np.array(miss_train_label)

    miss_test_label = [0]*len(y_test)
    for i in idx_miss_test_list:
        miss_test_label[i] = 1
    miss_test_label = np.array(miss_test_label)

    return miss_train_label, miss_test_label, idx_miss_test_list


def get_select_id(miss_train_label):
    wrong_id_list = []
    correct_id_list = []
    for i in range(len(miss_train_label)):
        if miss_train_label[i] == 1:
            wrong_id_list.append(i)
        else:
            correct_id_list.append(i)
    return wrong_id_list, correct_id_list


def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd

