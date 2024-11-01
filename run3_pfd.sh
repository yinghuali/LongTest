#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-rf.model' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-dt.model' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-lr.model' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-dnn.h5' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-tabnet.pth.zip' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-tabnet_10_pfd.json'
