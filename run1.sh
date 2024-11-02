#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch
#SBATCH --mem 10G

python get_chunk_embedding.py --num_chunks 5 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/test2.pkl'
