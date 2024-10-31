#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:00:00

python largeFileTest.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/EURLEX57K_MiniLM-L6-v2-rf.model' --path_save_res './result/EURLEX57K_MiniLM-L6-v2-rf_10.json'
python largeFileTest.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/EURLEX57K_MiniLM-L6-v2-dt.model' --path_save_res './result/EURLEX57K_MiniLM-L6-v2-dt_10.json'
python largeFileTest.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/EURLEX57K_MiniLM-L6-v2-lr.model' --path_save_res './result/EURLEX57K_MiniLM-L6-v2-lr_10.json'
python largeFileTest_DNN.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/EURLEX57K_MiniLM-L6-v2-dnn.h5' --path_save_res './result/EURLEX57K_MiniLM-L6-v2-dnn_10.json'
python largeFileTest_TabNet.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/EURLEX57K_MiniLM-L6-v2-tabnet.pth.zip' --path_save_res './result/EURLEX57K_MiniLM-L6-v2-tabnet_10.json'


python largeFileTest.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/FakeNews_MiniLM-L6-v2-rf.model' --path_save_res './result/FakeNews_MiniLM-L6-v2-rf_10.json'
python largeFileTest.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/FakeNews_MiniLM-L6-v2-dt.model' --path_save_res './result/FakeNews_MiniLM-L6-v2-dt_10.json'
python largeFileTest.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/FakeNews_MiniLM-L6-v2-lr.model' --path_save_res './result/FakeNews_MiniLM-L6-v2-lr_10.json'
python largeFileTest_DNN.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/FakeNews_MiniLM-L6-v2-dnn.h5' --path_save_res './result/FakeNews_MiniLM-L6-v2-dnn_10.json'
python largeFileTest_TabNet.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/FakeNews_MiniLM-L6-v2-tabnet.pth.zip' --path_save_res './result/FakeNews_MiniLM-L6-v2-tabnet_10.json'


python largeFileTest.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/CancerDoc_MiniLM-L6-v2-rf.model' --path_save_res './result/CancerDoc_MiniLM-L6-v2-rf_10.json'
python largeFileTest.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/CancerDoc_MiniLM-L6-v2-dt.model' --path_save_res './result/CancerDoc_MiniLM-L6-v2-dt_10.json'
python largeFileTest.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/CancerDoc_MiniLM-L6-v2-lr.model' --path_save_res './result/CancerDoc_MiniLM-L6-v2-lr_10.json'
python largeFileTest_DNN.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/CancerDoc_MiniLM-L6-v2-dnn.h5' --path_save_res './result/CancerDoc_MiniLM-L6-v2-dnn_10.json'
python largeFileTest_TabNet.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/CancerDoc_MiniLM-L6-v2-tabnet.pth.zip' --path_save_res './result/CancerDoc_MiniLM-L6-v2-tabnet_10.json'


