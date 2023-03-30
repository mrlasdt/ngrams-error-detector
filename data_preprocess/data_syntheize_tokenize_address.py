

#%%
from tools.utils import save_pkl, write_data, flatten
import os
import config_ngram as cfg
import pandas as pd
from tqdm import tqdm
def synthetize():
    data_syn = []
    data_syn +=  [[f"quận {cfg.REPLACE_CODE['number']}"] for _ in range(10)]
    data_syn +=  [["tỉnh đắk lắk"] for _ in range(10)]
    # data_syn.extend([f"khu phố từ liêm {cfg.REPLACE_CODE['number']} thị trấn nam ban"]*100)
    return data_syn
#%%


def drop_duplicates(data):
    df = pd.DataFrame(data)
    print('Before drop_duplicate: ', len(df))
    df.drop_duplicates(inplace=True)
    print('After drop_duplicate: ', len(df))
    data = df.values.tolist()
    return data
def load_data(path):
    path = path or cfg.ADDRESSES_TRAIN_PATH_PREPROCESSED
    with open(path, 'r') as f:
        return f.read().splitlines()
def tokenize (data):
    data_tokenized = []
    for d in tqdm(data):
        data_tokenized.append(d[0].strip().split())
    return data_tokenized

def save_data(data,path):
    path = path or cfg.ADDRESSES_TRAIN_PATH_TOKENIZED
    save_pkl(data, path)

def main(data_path=None, syn = True, tokenize_=True):
    data = load_data(data_path)
    data = drop_duplicates(data)
    if syn:
        data.extend(synthetize())
    if tokenize_:
        data = tokenize(data)
    print('Tokenized ', len(data))
    return data

#%%
if __name__ == "__main__":
    # main()
    preprocess_path = 'corpus/danhba_org'
    data_full = []
    for path, _, lfiles in os.walk(preprocess_path):
        if lfiles:
            for file in lfiles:
                if file.endswith('txt'):
                    file_path = os.path.join(path, file)
                    data = load_data(file_path)
                    data_full.extend(data)
    data_full = drop_duplicates(data_full)
    # print(data_full)
    print('Tokenized ', len(data_full))
    write_data('corpus/danhba_org_drop_duplicated.txt', flatten(data_full))
    # data_full = tokenize(data_full)
    # print('Tokenized ', len(data_full))
    # save_data(data_full, 'corpus/danhba_org.pkl')


    

# %%
# %%
