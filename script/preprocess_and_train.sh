#! /bin/sh
python3 data_preprocess/data_preprocess_address.py --istrain 1
python3 data_preprocess/data_syntheize_tokenize_address.py 
python3 train.py --istrain 1

