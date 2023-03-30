# %%
import os
from multiprocessing import Pool
from tqdm import tqdm
from config_ngram import TRAIN_TIENG_VIET_PATH, DATA_DIR
from tools.utils import read_data, sub_number_email_address_code_name, remove_reduntant_space_and_newline, remove_quote, split_tab, write_data, split_phrase_to_sents
NPROC = 4

# %%
# def remove_special(text):
# return re.sub(r'[\w\_]+[;#$%^&*]+[\w\_]*|[\w\_]*[;#$%^&*]+[\w\_]+', '', text)


def preprocess(text):
    text = text.lower().strip()
    text = split_tab(text)  # only use when preprocess train_tieng_viet.txt
    text = sub_number_email_address_code_name(text)
    text = remove_quote(text)
    text = remove_reduntant_space_and_newline(text)
    return text


def preprocess_data(data_):
    with Pool(NPROC) as p:
        return list(tqdm(p.imap(preprocess, data_), total=len(data_)))


# %%
def main():
    data = read_data(TRAIN_TIENG_VIET_PATH)
    data = preprocess_data(data)
    data = split_phrase_to_sents(data)
    write_data(f"{DATA_DIR}/{os.path.basename(TRAIN_TIENG_VIET_PATH).split('.')[0]}_cleaned.txt", data)


# %%
if __name__ == '__main__':
    main()
# #%%
# data = read_data(TRAIN_TIENG_VIET_PATH)

# # %%
# print(data)
# data = preprocess_data(data)
# data = split_phrase_to_sents(data)
# # write_data(f'{os.path.splitext(os.path.basename(TRAIN_TIENG_VIET_PATH))}_cleaned.txt', data)
# print(data)

# # %%
