#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:00:00

python test.py --num_chunks 5 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/test2.pkl'