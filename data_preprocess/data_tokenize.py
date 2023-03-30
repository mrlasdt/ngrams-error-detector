# %%
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from tools.utils import save_pkl, read_data, tokenize_
# %%


def get_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--file', type=str, help='path to corpus file')
    parse.add_argument('--nproc', type=int, help='number of cpus', default=4)
    return parse.parse_args()

#%%
def tokenize_par(data, nproc):
    with Pool(nproc) as p:
        return list(tqdm(p.imap(tokenize_, data), total=len(data)))
#%%

def main():
    arg = get_arg()
    docs = read_data(arg.file)[:10]
    docs_tokenized = tokenize_par(docs, arg.nproc)
    print('Tokenized!')
    save_pkl(docs_tokenized, arg.file.split('.')[0] + '_tokenized.pkl')


# %%
if __name__ == "__main__":
    main()

# %%
# temp = load('corpus/tokenizer_standarized.h5')
# vnword = temp['tone'].word_index
# vnword, len(vnword)
# %%

# %%
# #%%
# save_corpus_pkl(docs_test_tokenized, 'corpus/train_tieng_viet_cleaned.txt'.split('.')[0] + '.pkl')

# %%
# read_pkl('corpus/train_tieng_viet_cleaned.pkl')
# %%
