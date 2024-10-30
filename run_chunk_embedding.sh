#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch
#SBATCH --mem 10G

#python get_chunk_embedding.py --num_chunks 5 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X_5.pkl'
#python get_chunk_embedding.py --num_chunks 10 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl'
#python get_chunk_embedding.py --num_chunks 15 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X_15.pkl'
#python get_chunk_embedding.py --num_chunks 20 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X_20.pkl'


python get_chunk_embedding.py --num_chunks 5 --path_data './data/FakeNews/df_all_FakeNews.csv' --path_save_X './data/embedding_data/FakeNews_chunk_X_5.pkl'
python get_chunk_embedding.py --num_chunks 10 --path_data './data/FakeNews/df_all_FakeNews.csv' --path_save_X './data/embedding_data/FakeNews_chunk_X_10.pkl'
python get_chunk_embedding.py --num_chunks 15 --path_data './data/FakeNews/df_all_FakeNews.csv' --path_save_X './data/embedding_data/FakeNews_chunk_X_15.pkl'
python get_chunk_embedding.py --num_chunks 20 --path_data './data/FakeNews/df_all_FakeNews.csv' --path_save_X './data/embedding_data/FakeNews_chunk_X_20.pkl'
