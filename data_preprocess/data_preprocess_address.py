"""

"""
# %%
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
import config_ngram as cfg
from tools.utils import read_data, sub_number_email_address_code_name, remove_reduntant_space_and_newline, remove_quote, split_tab, write_data, split_phrase_to_sents
import re
import glob

import unicodedata


STANDARDIZE_TRAIN_ADDRESS_UNITS = {  # replace value by key (not key by value)
    'đ': 'đường',
    'ql': 'quốc lộ',
    'p': 'phường',
    'f': 'phường',
    'ph': 'phường',
    'q': 'quận',
    'tp': 'thành phố',
    't.p': 'thành phố',
    "t": "tỉnh",
    'h': 'huyện',
    'p': 'phường',
    'x': 'xã',
    'tx': 'thị xã',
    't.x': 'thị xã',
    'kp': 'khu phố',
    'k.p': 'khu phố',
    'kdt': 'khu đô thị',
    'kđt': 'khu đô thị',
    'ktt': 'khu tập thể',
    'kcn': 'khu công nghiệp',
    'kcx': 'khu chế xuất',
    'kdc': 'khu dân cư',
    'tdp': 'tổ dân phố',
    'ccn': 'cụm công nghiệp',
    'kv': 'khu vực',
    'tt': 'thị trấn',
    't.t': 'thị trấn',
    'vp': 'văn phòng',
    'hn': 'hà nội',
    'hcm': 'hồ chí minh',
    'vn': 'việt nam',
}
STANDARDIZE_TRAIN_ADDRESS_SHORTKEYS = [k for k in STANDARDIZE_TRAIN_ADDRESS_UNITS if k not in ['hn', 'hcm']]


def get_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--istrain', type=bool, default=False, help='train or val')
    return parse.parse_args()


def load_val_data():
    txt_paths = glob.glob(f'{cfg.ADDRESSES_VAL_DIR}/*.txt')
    ips = []
    gts = []
    for txt_path in txt_paths:
        ip_and_gt = read_data(txt_path)
        ip_and_gt = [txt for txt in ip_and_gt if len(txt) > 1]
        for i in range(0, len(ip_and_gt), 2):
            ips.append(ip_and_gt[i])
            gts.append(ip_and_gt[i + 1])
    return ips, gts


def load_data(train: bool):
    return read_data(cfg.ADDRESSES_TRAIN_PATH_CLEANED) + read_data(cfg.ADDRESSES_TRAIN_PATH_SYN) if train else load_val_data()


def standardize_custom_for_train_data(text):
    text = text.replace("hcm tp. hồ chí minh (tphcm)", "hcm")
    text = text.replace("tphcm tp. hồ chí minh (tphcm)", "tphcm")
    text = re.sub(r"\s?\(tp\.?\s?hcm\)\s?", "", text)
    text = text.replace('tp. hà nội (hn)', 'hn')
    return text


def replace_text_by_span(text, text_replace, span):
    return text[:span[0]] + text_replace + text[span[1]:]


def add_space_before_num_in_code(text):
    for i, char in enumerate(text):
        if char.isdigit():
            # return text[:i] + "_" + text[i:]
            return text[:i] + " " + text[i:]


def standardize_shortkeys_with_number(text):
    '''
    e.g. q3 -> q 3, q.10 -> q. 10
    '''
    shortkeys = '|'.join(STANDARDIZE_TRAIN_ADDRESS_SHORTKEYS).replace('.', '\.')
    regex = "\b({s})\.?([1-9]|1[1-9])\w?\b".format(s=shortkeys)
    regex = re.compile(regex)
    matches = regex.finditer(text)
    for i, match in enumerate(matches):
        span = match.span()
        text_replace = add_space_before_num_in_code(match.group())
        # add one i to match with the additional space
        text = replace_text_by_span(text, text_replace, (span[0] + i, span[1] + i))
    # some outliers
    text = text.replace('tphcm', 'tp hcm')  # some outliers
    return text


def add_space_after_dot(text):
    for i, char in enumerate(text):
        if char == '.':
            # return text[:i] + "_" + text[i:]
            return text[:i + 1] + " " + text[i + 1:]


def standardize_shortkeys_with_text(text):
    '''
    e.g. tphcm -> tp hcm, tp.hcm -> tp. hcm
    '''
    shortkeys = '|'.join(STANDARDIZE_TRAIN_ADDRESS_SHORTKEYS).replace('.', '\.')
    regex = fr"\b({shortkeys})\.{{1}}\w+\b"
    regex = re.compile(regex)
    matches = regex.finditer(text)
    for i, match in enumerate(matches):
        span = match.span()
        text_replace = add_space_after_dot(match.group())
        # add one i to match with the additional space
        text = replace_text_by_span(text, text_replace, (span[0] + i, span[1] + i))
    # some outliers
    text = text.replace('tphcm', 'tp hcm')  # some outliers
    return text


def standardize_unit(text):
    """ 
    e.g. q 3 -> quận 3, tp hn, -> thành phố hà nội,
    """
    lts = []
    for text_split in text.split(','):
        lw = []
        for word in text_split.strip().split():
            if word in list(STANDARDIZE_TRAIN_ADDRESS_UNITS.keys()):
                lw.append(STANDARDIZE_TRAIN_ADDRESS_UNITS[word])
            elif word in [k + '.' for k in list(STANDARDIZE_TRAIN_ADDRESS_UNITS.keys())]:
                lw.append(STANDARDIZE_TRAIN_ADDRESS_UNITS[word[:-1]])
            else:
                lw.append(word)
        lts.append(lw)
    text_splited = [' '.join(ts) for ts in lts]
    text_with_comma = ", ".join(text_splited)
    return text_with_comma


def sub_address_number_custom_backup(text, train: bool):
    # address: 16A/20/11G
    text = re.sub(r'\b(\d+[a-zA-Z]?\/\d+[\w\/\-]*\/?|[a-zA-Z]?\d+\/\d+[\w\/\-]*\/?)\b',
                  cfg.REPLACE_CODE['address'], text)
    text = unicodedata.normalize('NFKD', text)
    vn_chars = unicodedata.normalize('NFKD', 'àáảãạâấầẩẫậăắằẩẫặòóỏõọôồốổỗộơờớởỡợèéẻẽẹêềếểễệuùúủũụìíỉĩịýỳỷỹỵđưừứửữự')
    # vn_chars = 'àáảãạâấầẩẫậăắằẩẫặòóỏõọôồốổỗộơờớởỡợèéẻẽẹêềếểễệuùúủũụìíỉĩịýỳỷỹỵđưừứửữự'
    text = re.sub(fr'\w+[^a-zA-Z{vn_chars}\s]+\w+', cfg.REPLACE_CODE['code'], text)
    text = re.sub(r'\b(\d{1,3}|[123]+\d{3})[a-zA-Z]+\b', cfg.REPLACE_CODE['code'], text)
    text = re.sub(r'\b[a-zA-Z]\d{1,3}\b', cfg.REPLACE_CODE['code'], text)
    text = re.sub(r'\b(\d{1,3}|[123]+\d{3})\b', cfg.REPLACE_CODE['number'], text)
    text = re.sub(r'\b([456789]+\d{3,}|\d{5,})\b', '<LONG_NUMBER>', text)
    return text


def sub_address_number_custom(text, train: bool):
    # address: 16A/20/11G
    text = re.sub(r'\b(\d+[a-zA-Z]?\/\d+[\w\/\-]*\/?|[a-zA-Z]?\d+\/\d+[\w\/\-]*\/?)\b',
                  cfg.REPLACE_CODE['address'], text)
    text = re.sub(r'\b(\d{1,3}|[123]+\d{3})\b', cfg.REPLACE_CODE['number'], text)
    text = re.sub(r'\b([456789]+\d{3,}|\d{5,})\b', '<LONG_NUMBER>', text)
    text = re.sub(r'\b(\d{1,3}|[123]+\d{3})[a-zA-Z]+\b', cfg.REPLACE_CODE['code'], text)
    text = re.sub(r'\b[a-zA-Z]\d{1,3}\b', cfg.REPLACE_CODE['code'], text)

    # text = re.sub(r'(\b\w+\d+\W+\w*\d*\b|\b\w*\d*\W+\w+\d+\b)', cfg.REPLACE_CODE['code'], text) #necessary?
    # text = re.sub(r'([^\d\s]+[\d;#$%^&*\/\-+=|]+[^\d\s]*\b)|([^\d\s]*[\d;#$%^&*\/\-+=|]+[^\d\s]+\b)', cfg.REPLACE_CODE['code'], text) #necessary?

    text = text.replace('<CODE>/<NUMBER>', '<ADDRESS>')
    if train:
        re_number_to_code = fr'{cfg.REPLACE_CODE["number"]*2}|{cfg.REPLACE_CODE["number"]}(/{cfg.REPLACE_CODE["number"]})+|[^\r\n\t\f\v \(\.\,\'\"\<\_]*{cfg.REPLACE_CODE["number"]}[^\r\n\t\f\v \)\.\,\'\"\>\_]+|[^\r\n\t\f\v \(\.\,\'\"\<\_]+{cfg.REPLACE_CODE["number"]}[^\r\n\t\f\v \)\.\,\'\"\>\_]*'
        text = re.sub(re_number_to_code, cfg.REPLACE_CODE['code'], text)

        text = text.replace('<CODE>>', '<CODE>')
        text = text.replace('<<CODE>', '<CODE>')
        text = text.replace('<CODE>.<CODE>', '<CODE>')
        text = text.replace('<NUMBER>.<CODE>', '<CODE>')
        text = text.replace('<CODE>.<NUMBER>', '<CODE>')
        text = text.replace('<CODE>.<ADDRESS>', '<CODE>')
        text = text.replace('<ADDRESS>.<CODE>', '<CODE>')
        text = text.replace('<NUMBER>.<NUMBER>', '<CODE>')
        text = text.replace('<NUMBER>.<NUMBER><NUMBER>', '<CODE>')
        text = text.replace('<NUMBER>.<NUMBER>.<NUMBER>', '<CODE>')
        text = text.replace('<NUMBER><NUMBER>.<NUMBER>', '<CODE>')
        text = text.replace('<NUMBER>.<ADDRESS>', '<CODE>')
        text = text.replace('<ADDRESS>.<NUMBER>', '<CODE>')

        text = text.replace('<NUMBER> <NUMBER> <NUMBER>', '<NUMBER>')
        text = text.replace('<NUMBER> <NUMBER>', '<NUMBER>')
        text = text.replace('<NUMBER> <ADDRESS>', '<CODE>')
        text = text.replace('<ADDRESS> <NUMBER>', '<CODE>')
        text = text.replace('<CODE> <NUMBER>', '<CODE>')
        text = text.replace('<NUMBER> <CODE>', '<CODE>')
        text = text.replace('<ADDRESS> <CODE>', '<CODE>')
        text = text.replace('<CODE> <ADDRESS>', '<CODE>')
        text = text.replace('<ADDRESS> <ADDRESS>', '<CODE>')
        text = text.replace('<CODE> <CODE>', '<CODE>')

        text = text.replace('sô <NUMBER>', 'số <NUMBER>')
    return text


def augment_train_text(text):
    text_splited = [ts.strip() for ts in text.split(',')]
    text_no_comma = " ".join(text_splited)
    for code in ['<NUMBER> <NUMBER', '<CODE> <CODE>', '<ADDRESS> <ADDRESS>',
                 '<NUMBER> <CODE>', '<CODE> <NUMBER>', '<NUMBER> <ADDRESS>', '<ADDRESS> <NUMBER>',
                 '<ADDRESS> <CODE>', '<CODE> <ADDRESS>']:
        if code in text_no_comma:
            return text_splited
    return [text_no_comma] + text_splited


def preprocess_train(augment: bool = True, data=None, savefile=None, normalize=False):
    if data == None:
        data = load_data(train=True)
    data_preprocessed = []
    for text in tqdm(data):
        # text = 'Số 30, QL.1A, TDP Viện Điều Tra Quy Hoạch Rừng, X. Vĩnh Quỳnh, H. Thanh Trì, Hà Nội, Việt Nam'
        # text = 'Số nhà RS7.SH05, Tầng 1, Thỏp RS7, Tũa nhà Richstar Residenc - Phường Hiệp Tõn - Quận Tõn Phỳ - TP Hồ Chớ Minh. Tp. Hồ Chí Minh (TPHCM), Việt Nam'
        # text = 'p.tân tạo, thành phố hồ chí minh, việt nam'
        text = text.lower().strip()
        if normalize:
            text = unicodedata.normalize(cfg.MODE_NORMALIZE, text)
        # standarize
        text = standardize_custom_for_train_data(text)
        text = standardize_shortkeys_with_number(text)
        text = standardize_shortkeys_with_text(text)  # only for train data
        text = standardize_unit(text)
        text = sub_address_number_custom(text, True)
        text = remove_reduntant_space_and_newline(text)
        text_augment = augment_train_text(text) if augment else []
        # data_preprocessed.extend([text] + text_augment)
        if augment:
            data_preprocessed.extend(text_augment)
        else:
            data_preprocessed.append(text)

    file_name = cfg.ADDRESSES_TRAIN_PATH_PREPROCESSED if augment else f"{cfg.ADDRESSES_TRAIN_PATH_PREPROCESSED}_no_augment.txt"
    file_name = savefile or file_name
    write_data(file_name, data_preprocessed)


def preprocess_val():
    ips, gts = load_data(train=False)
    data_preprocessed = []
    for i in tqdm(range(len(ips))):
        text = ips[i].lower().strip()
        # text = '8017 Phan Chu Trinh Phường Phan Chu Trinh Quận Hoàn Kiếm TP Hà Nội'
        text = text.replace("✪", " ")
        text = standardize_shortkeys_with_number(text)
        text = standardize_unit(text)
        text = sub_address_number_custom(text, False)
        text = remove_reduntant_space_and_newline(text)
        data_preprocessed.append(text)
        data_preprocessed.append(gts[i])
    write_data(cfg.ADDRESSES_VAL_PATH_PREPROCESSED, data_preprocessed)


def main():
    arg = get_arg()
    if arg.istrain:
        preprocess_train(augment=True)
    else:
        preprocess_val()


# %%
if __name__ == '__main__':
    main()
