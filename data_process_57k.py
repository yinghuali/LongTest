import os
import json
import pandas as pd

path_dir_compile = './data/EURLEX57K/'
path_save = './data/EURLEX57K/df_all_EURLEX57K.csv'


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.json'):
                    path_list.append(file_absolute_path)
    return path_list


def main():
    path_list = get_path(path_dir_compile)
    label_list = []
    content_list = []
    for path in path_list:
        dic = json.load(open(path, 'r'))
        label = dic['concepts'][0]
        content = str(dic['title'])+str(dic['header'])+str(dic['recitals'])+str(dic['main_body'])+str(dic['attachments'])
        content_list.append(content)
        label_list.append(label)

    label_set = list(set(label_list))
    y_map = list(range(len(label_set)))
    dic_map = dict(zip(label_set, y_map))

    y_list = [dic_map[i] for i in label_list]

    df = pd.DataFrame(columns=['content'])
    df['content'] = content_list
    df['y'] = y_list
    df.to_csv(path_save, index=False, sep=',')


if __name__ == '__main__':
    main()
