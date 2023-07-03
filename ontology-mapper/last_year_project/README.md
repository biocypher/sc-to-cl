# Installation
pip install -r requirements.txt

# Embedding
mkdir cwd  
python extract.py cwd/native_cells.csv (takes time)

# Run generator
python [generator.py](http://generator.py/) --numpy --prefix cwd/scibert_ncs_nn_ --embeddings sum_all_hidden_mean last_cls --bert scibert cwd\native_cells.csv

# Run app
python [app.py](http://app.py/) --embedding_words cwd\scibert_ncs_nn_sum_all_hidden_mean_synonyms.csv --embedding_file cwd\scibert_ncs_nn_sum_all_hidden_mean.npy --embedding_fn sum_all_hidden_mean --mapper nearest_neighbor --bert scibert --save cwd/scibert_ncs_nn_sum_all_hidden_mean.mdl --wordlist cwd\scA_list_with_solutions.csv
