#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G

python get_chunk_embedding_select.py --num_chunks 5 --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_5.pkl' --embedding_approach 'all-mpnet-base-v2'
python get_chunk_embedding_select.py --num_chunks 10 --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_10.pkl' --embedding_approach 'all-mpnet-base-v2'
python get_chunk_embedding_select.py --num_chunks 15 --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_15.pkl' --embedding_approach 'all-mpnet-base-v2'
python get_chunk_embedding_select.py --num_chunks 20 --path_data './data/CancerDoc/df_all_CancerDoc.csv' --path_save_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_20.pkl' --embedding_approach 'all-mpnet-base-v2'

