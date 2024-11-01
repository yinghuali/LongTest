#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G

python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-rf.model' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dt.model' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-lr.model' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dnn.h5' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-tabnet.pth.zip' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-tabnet_10_pfd.json'

