# %%
import config_ngram as cfg #import BINHVQ_PATH, DATA_DIR, N_SENTENCE_MAX, SEED
from tools.utils import read_data, sub_number_email_address_code_name, remove_reduntant_space_and_newline, split_phrase_to_sents, remove_quote, write_data
from multiprocessing import Pool
from tqdm import tqdm
import os
import random
import argparse
random.seed(cfg.SEED)
NPROC = 24
# %%
def get_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--nsent', type=int, default=-1, help='use full corpus if not specified')
    parse.add_argument('--nproc', type=str, help='no. processor for multiprocessing')
    return parse.parse_args()



def preprocess(text):
    text = text.lower()
    text = sub_number_email_address_code_name(text)
    text = remove_quote(text)
    text = remove_reduntant_space_and_newline(text)
    return text


def preprocess_par(data_):
    with Pool(NPROC) as p:
        return list(tqdm(p.imap(preprocess, data_), total=len(data_)))

def preprocess_batch(data_):
    res = []
    for text in tqdm(data_):
        text = preprocess(text)
        res.append(text)
    return res  

def main():
    arg = get_arg()
    data = read_data(cfg.BINHVQ_PATH)
    nsent = arg.nsent if arg.nsent!=-1 else 'full'
    if nsent!='full':
        data = random.sample(data, nsent)
    data = preprocess_par(data)
    data = split_phrase_to_sents(data)
    write_data(f"{cfg.DATA_DIR}/{os.path.basename(cfg.BINHVQ_PATH).split('.')[0]}_{nsent}_cleaned.txt", data)
    


# %%
if __name__ == '__main__':
    main()

# %%
# data = read_data(BINHVQ_PATH)[:20]
# data_ = "\n".join(data)
# data_ = sub_number_email_address_code_name(data_)
# data_ = remove_reduntant_space_and_newline(data_)
# data, data_
# %%
