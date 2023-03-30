#%%
import os
from functools import partial
from data_preprocess.data_preprocess_address import preprocess_train
import unicodedata
from tools.utils import read_data
data_path = '/home/sds/hungbnt/web_crawler/danhba_org/'
preprocess_path = 'corpus/danhba_org'

# %%
total_preprocessed_data = 0
for path, _, lfiles in os.walk(data_path):
    if lfiles:
        for file in lfiles:
            file_path = os.path.join(path, file)
            data = read_data(file_path)
            save_dir = os.path.join(preprocess_path, path.split('danhba_org/')[-1])
            os.makedirs(save_dir, exist_ok=True)       
            save_file = os.path.join(save_dir, file)
            preprocess_train(augment=True, data= data, savefile =save_file, normalize=True)
            total_preprocessed_data += len(data)
print('[INFO]: total_preprocessed_data ', total_preprocessed_data)
