# %%
# %%
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
import config_ngram as cfg
from tools.utils import read_data, sub_number_email_address_code_name, remove_reduntant_space_and_newline, remove_quote, split_tab, write_data, split_phrase_to_sents
import re
import glob
# NPROC = 4

def get_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--istrain', type=bool, default = False, help='train or val')
    return parse.parse_args()

def load_val_data():
    txt_paths = glob.glob(f'{cfg.ADDRESSES_VAL_DIR}/0*.txt') + glob.glob(f'{cfg.ADDRESSES_VAL_DIR}/1_*.txt')
    ips = []
    gts = []
    for txt_path in txt_paths:
        ip_and_gt = read_data(txt_path)
        for i in range(0, len(ip_and_gt), 2):
            ips.append(ip_and_gt[i])
            gts.append(ip_and_gt[i+1])
    return ips, gts
    
def load_data(train:bool):
    return read_data(cfg.ADDRESSES_TRAIN_PATH) if train else load_val_data()
    
def replace_text_by_span(text, text_replace, span):
    return text[:span[0]] + text_replace + text[span[1]:]

def add_space_before_num_in_code(text):
    for i, char in enumerate(text):
        if char.isdigit():
            # return text[:i] + "_" + text[i:]
            return text[:i] + " " + text[i:]
        
def prestandardize_val_address(text):
    list_code = [v[0] for k,v in cfg.STANDARDIZE_TRAIN_ADDRESS_CODE.items()] + ['kđt', 'f', 'ph']
    regex = fr"({'|'.join(list_code)})\.?\d+\b"
    regex = re.compile(regex)
    matches = regex.finditer(text)
    for i, match in enumerate(matches):
        span = match.span()
        text_replace = add_space_before_num_in_code(match.group())
        text = replace_text_by_span(text, text_replace, (span[0]+i, span[1]+i))
    return text

def preprocess(text, train:bool):
    text = text.lower().strip()
    if not train:
        text = text.replace("✪", " ")
    text = prestandardize_val_address(text) #? should we use this for train data?
    text = sub_number_email_address_code_name(text)
    text = re.sub(r'\s?\(tp\.?\s?hcm\)\s?', '', text)
    text = text.replace("hcm tp. hồ chí minh (tphcm)", 'hcm')
    text = text.replace("tphcm tp. hồ chí minh (tphcm)", 'tphcm')
    text = remove_quote(text)
    text = remove_reduntant_space_and_newline(text)
    return text


def preprocess_data(data_, train:bool):
    # with Pool(NPROC) as p:
    #     return list(tqdm(p.imap(preprocess, data_), total=len(data_)))
    return [preprocess(text, train) for text in tqdm(data_)]




def standardize_address(text):
    for k,v in cfg.STANDARDIZE_TRAIN_ADDRESS_CODE.items():
        if text in v:
            return k;
    return text
            
def standardize_and_augment_address(text, augment:bool):
    text_splited = [ts.strip() for ts in text.split(',')]
    text_splited = [[standardize_address(t) for t in ts.split()] for ts in text_splited]
    text_splited = [' '.join(ts) for ts in text_splited]
    text_with_comma = ", ".join(text_splited)
    if not augment:
        return [text_with_comma]
    text_no_comma = " ".join(text_splited)
    text_splited.extend([text_with_comma, text_no_comma])
    return text_splited


def standardize_and_augment_data(data, augment:bool):
    data_ = []
    for text in tqdm(data):
        text_augment = standardize_and_augment_address(text, augment)
        data_.extend(text_augment)
    return data_

def write_address_data(data, train:bool):
    if train:
        write_data(f"{cfg.DATA_DIR}/{os.path.basename(cfg.ADDRESSES_TRAIN_PATH).split('.')[0]}_cleaned_no_augment.txt", data)
    else:
        combined = []
        for i in range(len(data[0])):
            if len(data[0][i])==0:
                continue;
            combined.append(data[0][i])
            combined.append(data[1][i])
        write_data(f"{cfg.DATA_DIR}/address_val_cleaned.txt", combined)
        
# %%


def main():
    arg = get_arg()
    print('TEST1')
    data = load_data(arg.istrain)
    # data = [['44 Trần TU Bình Khác Trọng P.8✪Q3', '']]
    # data = [[' 28 Lý Thương Kiệt D11, Q10\n', '']]
    # data = ['91/17 Đường Số 18, F8, Gò Vấp, Tp. Hồ Chí Minh (TPHCM), Việt Nam']
    print('TEST2')
    data = preprocess_data(data, arg.istrain) if arg.istrain==1 else (preprocess_data(data[0], arg.istrain), data[1])
    print('TEST3')
    data = standardize_and_augment_data(data, True) if arg.istrain==1 else (standardize_and_augment_data(data[0], False), data[1])
    print('TEST4')
    write_address_data(data, arg.istrain)
    print('TEST5')

# %%
if __name__ == '__main__':
    main()
