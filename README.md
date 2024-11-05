# LongTest
LongTest: Test Prioritization for Long Text Files
## Main Requirements
    Python 3.7.2
    PyTorch 1.11.0
    TensorFlow 2.2.0
    scikit-learn 0.24.2
    pytorch-tabnet 4.1.0
    SentenceTransformer 2.2.2
##  Repository catalogue
    target_models: the evaluated models.
    result: experimental results of the paper.
    get_chunk_embedding.py: script of chunk text embedding.
    get_file_embedding.py: script of file embedding.
    get_rank_idx.py: script for uncertainty approaches.
    largeFileTest.py: script of LongTest.
    utils.py: script for tools.

## How to run LongTest
#### Step1: Download the dataset: 
XXXXX
This dataset includes all three dataset embedding vectors.   
Please unpack the dataset and place them in the LongTest directory. 

#### Step2: Run LongTest  
This is an example of running LongTest on EURLEX57K dataset.  
```
python largeFileTest.py --path_file_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_X.pkl' --path_file_y './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_file_y.pkl' --path_chunk_embedding_X './data/embedding_data/all-MiniLM-L6-v2/EURLEX57K_chunk_X_10.pkl' --path_target_model './target_models/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-rf.model' --path_save_res './result/all-MiniLM-L6-v2/EURLEX57K_MiniLM-L6-v2-rf_10.json'
```
