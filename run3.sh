#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G

python get_chunk_embedding.py --num_chunks 15 --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/CancerDoc_chunk_X_15.pkl'

