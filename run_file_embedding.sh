#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch
#SBATCH --mem 10G

# python get_file_embedding.py --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_save_y './data/embedding_data/EURLEX57K_file_y.pkl'

python get_file_embedding.py --path_data './data/20news/df_all_20news.csv' --path_save_X './data/embedding_data/20news_file_X.pkl' --path_save_y './data/embedding_data/20news_file_y.pkl'

