#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G

python get_chunk_embedding_EURLEX57K.py --num_chunks 15 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X_15.pkl' --path_save_ID './data/embedding_data/EURLEX57K_chunk_ID_15.pkl'

