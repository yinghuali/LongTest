#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:00:00

python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-rf.model' --path_save_res './result/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-dt.model' --path_save_res './result/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-lr.model' --path_save_res './result/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-dnn.h5' --path_save_res './result/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-tabnet.pth.zip' --path_save_res './result/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-tabnet_10_pfd.json'


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-rf.model' --path_save_res './result/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-dt.model' --path_save_res './result/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-lr.model' --path_save_res './result/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-dnn.h5' --path_save_res './result/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-tabnet.pth.zip' --path_save_res './result/all-MiniLM-L6-v2/FakeNews_MiniLM-L6-v2-tabnet_10_pfd.json'


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-rf.model' --path_save_res './result/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-dt.model' --path_save_res './result/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-lr.model' --path_save_res './result/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-dnn.h5' --path_save_res './result/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-tabnet.pth.zip' --path_save_res './result/all-MiniLM-L6-v2/CancerDoc_MiniLM-L6-v2-tabnet_10_pfd.json'

############

python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-rf.model' --path_save_res './result/all-distilroberta-v1/EURLEX57K_distilroberta-v1-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-dt.model' --path_save_res './result/all-distilroberta-v1/EURLEX57K_distilroberta-v1-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-lr.model' --path_save_res './result/all-distilroberta-v1/EURLEX57K_distilroberta-v1-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-dnn.h5' --path_save_res './result/all-distilroberta-v1/EURLEX57K_distilroberta-v1-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-tabnet.pth.zip' --path_save_res './result/all-distilroberta-v1/EURLEX57K_distilroberta-v1-tabnet_10_pfd.json'


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-rf.model' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-dt.model' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-lr.model' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-dnn.h5' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-tabnet.pth.zip' --path_save_res './result/all-distilroberta-v1/FakeNews_distilroberta-v1-tabnet_10_pfd.json'


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-rf.model' --path_save_res './result/all-distilroberta-v1/CancerDoc_distilroberta-v1-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-dt.model' --path_save_res './result/all-distilroberta-v1/CancerDoc_distilroberta-v1-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-lr.model' --path_save_res './result/all-distilroberta-v1/CancerDoc_distilroberta-v1-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-dnn.h5' --path_save_res './result/all-distilroberta-v1/CancerDoc_distilroberta-v1-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-tabnet.pth.zip' --path_save_res './result/all-distilroberta-v1/CancerDoc_distilroberta-v1-tabnet_10_pfd.json'


##############
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-rf.model' --path_save_res './result/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-dt.model' --path_save_res './result/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-lr.model' --path_save_res './result/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-dnn.h5' --path_save_res './result/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-tabnet.pth.zip' --path_save_res './result/all-mpnet-base-v2/EURLEX57K_mpnet-base-v2-tabnet_10_pfd.json'


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-rf.model' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dt.model' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-lr.model' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dnn.h5' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/FakeNews_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/FakeNews_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/FakeNews_mpnet-base-v2-tabnet.pth.zip' --path_save_res './result/all-mpnet-base-v2/FakeNews_mpnet-base-v2-tabnet_10_pfd.json'


python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-rf.model' --path_save_res './result/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-rf_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-dt.model' --path_save_res './result/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-dt_10_pfd.json'
python largeFileTest_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-lr.model' --path_save_res './result/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-lr_10_pfd.json'
python largeFileTest_DNN_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-dnn.h5' --path_save_res './result/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-dnn_10_pfd.json'
python largeFileTest_TabNet_pfd.py --path_file_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_X.pkl' --path_file_y './data/embedding_data/all-mpnet-base-v2/CancerDoc_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-mpnet-base-v2/CancerDoc_chunk_X_10.pkl' --path_target_model './target_models/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-tabnet.pth.zip' --path_save_res './result/all-mpnet-base-v2/CancerDoc_mpnet-base-v2-tabnet_10_pfd.json'

