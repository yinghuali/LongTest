

python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-rf.pkl' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-lr.pkl' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-xgb.pkl' --model_name 'xgb'

