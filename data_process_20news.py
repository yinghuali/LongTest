import os
import json
import pandas as pd

path_dir_compile = './data/20news/'
path_save = './data/20news/df_all_20news.csv'


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                path_list.append(file_absolute_path)
    return path_list


def read_txt(path):
    f = open(path, 'r', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    lines_list = [i.strip() for i in lines]
    text = ' '.join(lines_list)
    return text


def main():
    path_list = get_path(path_dir_compile)
    content_list = []
    type_list = []
    for path in path_list:
        content = read_txt(path)
        content_list.append(content)
        type_content = path.split('/')[-2]
        type_list.append(type_content)

    print(set(type_list))

    type_set = list(set(type_list))
    type_map = list(range(len(type_set)))
    dic_type_map = dict(zip(type_set, type_map))
    type_list = [dic_type_map[i] for i in type_list]

    df = pd.DataFrame(columns=['content'])
    df['content'] = content_list
    df['type'] = type_list
    df.to_csv(path_save, index=False, sep=',')


if __name__ == '__main__':
    main()
