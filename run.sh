

python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-rf.model' --model_name 'rf'
python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-lr.model' --model_name 'lr'
python get_target_models.py --path_file_embedding_X './data/embedding_data/EURLEX57K_file_X.pkl' --path_file_embedding_y './data/embedding_data/EURLEX57K_file_y.pkl' --path_save_model './target_models/MiniLM-L6-v2-dt.model' --model_name 'dt'



python get_chunk_embedding_EURLEX57K.py --num_chunks 5 --path_data './data/EURLEX57K/df_all_EURLEX57K.csv' --path_save_X './data/embedding_data/EURLEX57K_chunk_X.pkl' --path_save_ID './data/embedding_data/EURLEX57K_chunk_ID.pkl'
