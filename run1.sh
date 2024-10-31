#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G

python get_file_embedding_select.py --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_X.pkl' --path_save_y './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_y.pkl' --embedding_approach 'all-mpnet-base-v2'
python get_file_embedding_select.py --path_data './data/FakeNews/df_all_FakeNews.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_save_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --embedding_approach 'all-mpnet-base-v2'
python get_file_embedding_select.py --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_X.pkl' --path_save_y './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_y.pkl' --embedding_approach 'all-mpnet-base-v2'
python get_file_embedding_select.py --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_save_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --embedding_approach 'all-distilroberta-v1'
python get_file_embedding_select.py --path_data './data/FakeNews/df_all_FakeNews.csv' --path_save_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_save_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --embedding_approach 'all-distilroberta-v1'
python get_file_embedding_select.py --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_save_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --embedding_approach 'all-distilroberta-v1'


