# from nltk.lm.util import log_base2
# import numpy as np
from torch import load
import os
NGRAMS = 3
MIN_CHARS = 3
TRAIN_TIENG_VIET_PATH = '/home/sds/hungbnt/VietnameseOcrCorrection/dataloader/data/train_tieng_viet.txt'
BINHVQ_PATH = '/home/sds/hungbnt/VietnameseOcrCorrection/dataloader/data/corpus-full.txt'
DATA_DIR = 'corpus'

# ADDRESSES_TRAIN_PATH = 'corpus/address_train.txt'
ADDRESSES_TRAIN_PATH = '/mnt/hdd2T/AICR/Datasets/ErrorCorrector/address/address_train.txt'
# ADDRESSES_TRAIN_PATH_SYN = 'corpus/vietnam_dataset.txt'
ADDRESSES_TRAIN_PATH_SYN = '/mnt/hdd2T/AICR/Datasets/ErrorCorrector/address/vietnam_dataset.txt'
ADDRESSES_TRAIN_PATH_CLEANED = 'corpus/address_train_cleaned.txt'
ADDRESSES_TRAIN_PATH_PREPROCESSED = 'corpus/address_train_preprocessed.txt'
ADDRESSES_TRAIN_PATH_TOKENIZED = ['corpus/address_train_tokenized.pkl', 'corpus/danhba_org.pkl']
ADDRESSES_VAL_DIR = 'corpus/FWD_data/addresses'
ADDRESSES_VAL_PATH_PREPROCESSED = 'corpus/addresses_val_preprocessed.txt'

REPLACE_CODE = {'name': '<NAME>', 'address': '<ADDRESS>', 'number': '<NUMBER>', 'code': '<CODE>'}
UNKWOWN_CODE = '<UNK>'
# TOKENIZE_DICT_PATH = 'corpus/tokenizer_standarized.h5'
# VNWORD = list(load(TOKENIZE_DICT_PATH)['tone'].word_index.keys())[3:] + list(REPLACE_CODE.values()) + [UNKWOWN_CODE]
SEED = 42
# E_THRESH = log_base2(1e-5)
# E_THRESH = np.log(0.2)


CHANGE_RATIO = 0.5

E_THRESH = 1e-4

# PRESTANDARDIZE_VAL_ADDRESS_CODE = {
#     ' ' : ['✪'],
# }

MODEL_PATH = {
    'forward' : 'models/addresses_forward.pkl',
    'bothward' : 'models/addresses_bothward_danhba.pkl',
    'backward' : 'models/addresses_backward.pkl',
}

MODE_NORMALIZE = 'NFC'