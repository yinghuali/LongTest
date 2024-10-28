#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:00:00

python largeFileTest.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/MiniLM-L6-v2-rf.model' --path_save_res './result/MiniLM-L6-v2-rf_10.json'

