


python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/EURLEX57K_MiniLM-L6-v2-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/EURLEX57K_MiniLM-L6-v2-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/EURLEX57K_MiniLM-L6-v2-dt.model' --model_name 'dt'
python get_target_DNN.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/EURLEX57K_MiniLM-L6-v2-dnn.h5' --epochs 3
python get_target_tabnet.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/EURLEX57K_MiniLM-L6-v2-tabnet.pth' --epochs 10

python get_target_models.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/FakeNews_file_y.pkl' --path_save_model './target_models/FakeNews_MiniLM-L6-v2-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/FakeNews_file_y.pkl' --path_save_model './target_models/FakeNews_MiniLM-L6-v2-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/FakeNews_file_y.pkl' --path_save_model './target_models/FakeNews_MiniLM-L6-v2-dt.model' --model_name 'dt'
python get_target_DNN.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/FakeNews_file_y.pkl' --path_save_model './target_models/FakeNews_MiniLM-L6-v2-dnn.h5' --epochs 4
python get_target_tabnet.py --path_file_embedding_X './data/embedding_data/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/FakeNews_file_y.pkl' --path_save_model './target_models/FakeNews_MiniLM-L6-v2-tabnet.pth' --epochs 25


python get_target_models.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/CancerDoc_file_y.pkl' --path_save_model './target_models/CancerDoc_MiniLM-L6-v2-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/CancerDoc_file_y.pkl' --path_save_model './target_models/CancerDoc_MiniLM-L6-v2-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/CancerDoc_file_y.pkl' --path_save_model './target_models/CancerDoc_MiniLM-L6-v2-dt.model' --model_name 'dt'
python get_target_DNN.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/CancerDoc_file_y.pkl' --path_save_model './target_models/CancerDoc_MiniLM-L6-v2-dnn.h5' --epochs 40
python get_target_tabnet.py --path_file_embedding_X './data/embedding_data/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/CancerDoc_file_y.pkl' --path_save_model './target_models/CancerDoc_MiniLM-L6-v2-tabnet.pth' --epochs 250

############

python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-dt.model' --model_name 'dt'
python get_target_DNN.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-dnn.h5' --epochs 3
python get_target_tabnet.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/EURLEX57K_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/EURLEX57K_distilroberta-v1-tabnet.pth' --epochs 25

python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-dt.model' --model_name 'dt'
python get_target_DNN.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-dnn.h5' --epochs 4
python get_target_tabnet.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/FakeNews_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/FakeNews_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/FakeNews_distilroberta-v1-tabnet.pth' --epochs 25


python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-dt.model' --model_name 'dt'
python get_target_DNN.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-dnn.h5' --epochs 10
python get_target_tabnet.py --path_file_embedding_X './data/embedding_data/all-distilroberta-v1/CancerDoc_file_X.pkl' --path_file_embedding_y './data/embedding_data/all-distilroberta-v1/CancerDoc_file_y.pkl' --path_save_model './target_models/all-distilroberta-v1/CancerDoc_distilroberta-v1-tabnet.pth' --epochs 250

