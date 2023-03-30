
# %%
from itertools import tee, islice
import pickle
import re
import config_ngram as cfg
import sys
import string
import glob
VN_OCR_PATH = '/home/sds/hungbnt/VietnameseOcrCorrection'
sys.path.append(VN_OCR_PATH)
from .data_augmentation import random_replace_accent  # from VN_OCR
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from itertools import chain
from nltk.util import trigrams, pad_sequence
from nltk.lm.preprocessing import padded_everygram_pipeline
# %%


def flatten(t):
    return [item for sublist in t for item in sublist]


def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print('[INFO]: Saved to ', file_path)


def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def read_data(file_):
    with open(file_, 'r') as f:
        return f.readlines()


def extract_phrase(text):
    # extract phrase
    # re_replace_code = '{}|{}|{}|{}'.format(*cfg.REPLACE_CODE.values())
    # return re.findall(fr'\<?\w[\w\s\>\<]*\b|\s\W+|\W+', text)
    # return re.findall(r'\<?\w[\w\s<>/&%@#$^*?_+\=\-\(\)\\]*[^.,]', text)
    return re.findall(r'[^.,]*', text)
    


def sub_number_email_address_code_name(text):
    # https://stackoverflow.com/questions/54887282/python-regex-to-remove-urls-and-domain-names-in-string
    text = re.sub(r'[^\r\n\t\f\v\d ]+(\.\w)[^\r\n\t\f\v\d\b]*', cfg.REPLACE_CODE['address'], text)  # remove url and emails
    text = re.sub(r'-?\d*[.,]?\d+%?\b|-?\d+[.,]?\d*%?\b', cfg.REPLACE_CODE['number'], text)
    text = re.sub(r'([^\d\s]+[\d;#$%^&*\/\-+=|]+[^\d\s]*\b)|([^\d\s]*[\d;#$%^&*\/\-+=|]+[^\d\s]+\b)', cfg.REPLACE_CODE['code'], text)
    # text = re.sub(r'(\w+[\d;#$%^&*\/\-_+=|]+\w*\b)|(\w*[\d;#$%^&*\/\-_+=|]+\w+\b)', cfg.REPLACE_CODE['code'], text)
    text = re.sub(r'[^\r\n\t\f\v \.\,]+\-[^\r\n\t\f\v \.\,]+', cfg.REPLACE_CODE['name'], text)
    # text = re.sub(r'(NUMBERNUMBER|[^\r\n\t\f\v\(]*NUMBER[^\r\n\t\f\v\)]+|[^\r\n\t\f\v\(]+NUMBER[^\r\n\t\f\v\)]*)', 'CODE', text)
    # text = re.sub(r'(NUMBERNUMBER|[\S]*\(?NUMBER\)?[\S]+|[\S]+\(?NUMBER\)?[\S]*)', 'CODE', text)
    
    re_number_to_code = fr'{cfg.REPLACE_CODE["number"]*2}|{cfg.REPLACE_CODE["number"]}(/{cfg.REPLACE_CODE["number"]})+|[^\r\n\t\f\v \(\.\,\'\"\<\_]*{cfg.REPLACE_CODE["number"]}[^\r\n\t\f\v \)\.\,\'\"\>\_]+|[^\r\n\t\f\v \(\.\,\'\"\<\_]+{cfg.REPLACE_CODE["number"]}[^\r\n\t\f\v \)\.\,\'\"\>\_]*'
    text = re.sub(re_number_to_code, cfg.REPLACE_CODE['code'], text)
    text = text.replace('<CODE>>', '<CODE>')
    text = text.replace('<<CODE>', '<CODE>')
    text = text.replace('<CODE>.<CODE>', '<CODE>')
    text = text.replace('<NUMBER>.<CODE>', '<CODE>')
    text = text.replace('<CODE>.<NUMBER>', '<CODE>')
    return text


def remove_reduntant_space_and_newline(text):
    text = re.sub(r"[\r\n\t]", " ", text)  # remove newline and enter char
    text = re.sub(r"\s{2,}", " ", text)  # replace two or more sequential spaces with 1 space
    return text.strip()


def remove_quote(text):
    return re.sub(r'[\"\']', '', text)


def split_tab(text):
    return text.split("\t")[1]


def write_data(file_, data_):
    with open(file_, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_))
    print('[INFO]: Saved to ', file_)

def split_phrase_to_sents(data_):
    sents = []
    for p in data_:
        if p is None:
            continue
        lp = extract_phrase(p)
        for p_ in lp:
            if len(p_) < cfg.MIN_CHARS and p_ not in string.punctuation:
                continue
            sents.append(p_)
    return sents


def random_replace_accent_with_mask(text):
    lwords = text.split()
    code_indices = [i for i, w in enumerate(lwords) if w in [cfg.UNKWOWN_CODE] + list(cfg.REPLACE_CODE.values())]
    text_without_code = ' '.join([w for w in lwords if w not in [cfg.UNKWOWN_CODE] + list(cfg.REPLACE_CODE.values())])
    text_without_code_changed = random_replace_accent(text_without_code, cfg.CHANGE_RATIO)
    # ltext_changed = []
    # code_cnt = 0
    # failed in this case corpus = ' '.join(['quy', 'ước', 'này', 'được', 'nhóm', 'NAME', 'NAME', 'working', 'group'])
    # for t in text_without_code_changed.split():
    #     if code_cnt < len(code_indices) and len(ltext_changed) == code_indices[code_cnt]:
    #         ltext_changed.append(lwords[code_indices[code_cnt]])
    #         code_cnt += 1
    #     ltext_changed.append(t)
    # if code_cnt < len(code_indices) and len(ltext_changed) == code_indices[code_cnt]:
    #     ltext_changed.append(lwords[code_indices[code_cnt]])
    ltext_changed = text_without_code_changed.split()
    _ = [ltext_changed.insert(code_index, lwords[code_index]) for code_index in code_indices]
    mask = [0 if i in code_indices else 1 for i in range(len(ltext_changed))]
    return ltext_changed, mask


def tokenize_(sent):
    # if isinstance(sent, str):
    sent = sent.strip().split()
    for idx, word in enumerate(sent):
        if word not in cfg.VNWORD:
            sent[idx] = cfg.UNKWOWN_CODE
    return sent
# def transform_to_val_data(corpus):
#     gts = []
#     ips = []
#     masks = []
#     for i in range(len(corpus)):
#         ltext_changed, mask = random_replace_accent_with_mask(corpus[i])
#         gt = [0 if t == ltext_changed[j] else 1 for j, t in enumerate(corpus[i].split())]
#         ip, mask, gt = pad_val_data(ltext_changed, mask, gt)
#         ips.append(ip)
#         masks.append(mask)
#         gts.append(gt)
#     return ips, masks, gts


def pad_val_data(ip, mask, gt):
    len_pad = cfg.NGRAMS - 1
    ip = ['<s>'] * len_pad + ip + ['</s>'] * len_pad
    mask = [0] * len_pad + mask + [0] * len_pad
    gt = [0] * len_pad + gt + [0] * len_pad
    return ip, mask, gt


def transform_corpus_to_val_data(text):
    text = text.strip()
    ltext_changed, mask = random_replace_accent_with_mask(text)
    # ltext_changed = tokenize_(ltext_changed)
    # if len(ltext_changed)!= len(text.split()):
    #     print(len(ltext_changed),len(text.split()))
    #     print(ltext_changed, text.split())
    gt = [0 if t == ltext_changed[i] else 1 for i, t in enumerate(text.split())]
    ip, mask, gt = pad_val_data(ltext_changed, mask, gt)
    return ip, mask, gt


def transform_corpus_to_val_data_par(corpus, nproc=4):
    '''
    use to transform corpus to validation data, just for inference
    '''
    with Pool(nproc) as p:
        return list(
            tqdm(
                p.imap(transform_corpus_to_val_data, corpus),
                total=len(corpus),
                desc='Transforming corpus to validation data...'))


def trigrams_bothward(sequence: list, mode: str):
    '''
    https://github.com/nltk/nltk/blob/6f18391a82ce20218f84bb4e8524a5fbaca354e9/nltk/lm/counter.py#L18
    context, word = ngram[:-1], ngram[-1]

    mode = 'forward' -> return w0 w1 w2
    mode = 'backward -> return w2 w1 w0
    mode = 'bothward'-> return w0 w2 w1
    customized from nltk.utils.trigrams
    '''
    iterables = tee(sequence, 3)
    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window
    if mode == 'forward':
        yield from zip(iterables[0], iterables[1], iterables[2])
    elif mode == 'backward':
        yield from zip(iterables[2], iterables[1], iterables[0])
    elif mode == 'bothward':
        yield from zip(iterables[2], iterables[0], iterables[1])
    else:
        raise KeyError('Invalid mode')


def padded_trigrams_pipeline(text: list, mode: str):
    '''
    padding for preprocessing before train
    https://www.nltk.org/_modules/nltk/lm/preprocessing.html#padded_everygram_pipeline
    '''
    flatten_ = chain.from_iterable
    pad_both_ends = partial(pad_sequence, pad_left=True, left_pad_symbol="<s>",
                            pad_right=True, right_pad_symbol="</s>")
    padding_fn_trigrams = partial(pad_both_ends, n=3)
    return (
        (trigrams_bothward(list(padding_fn_trigrams(sent)), mode) for sent in text),
        flatten_(map(padding_fn_trigrams, text))
    )


def every_trigrams_bothward(sequence, mode: str):
    # Pad if indicated using max_len.
    sequence = pad_sequence(sequence, 3)
    # Sliding window to store grams.
    history = list(islice(sequence, 3))

    # Yield ngrams from sequence.
    while history:
        # print('hist', history)
        if mode == 'forward':
            if len(history) == 3:
                yield tuple([history[0], history[1], history[2]])
            if len(history) >= 2:
                yield tuple([history[0], history[1]])
            if len(history) >= 1:
                yield tuple([history[0]])
        elif mode == 'backward':
            if len(history) == 3:
                yield tuple([history[2], history[1], history[0]])
            if len(history) >= 2:
                yield tuple([history[1], history[0]])
            if len(history) >= 1:
                yield tuple([history[0]])
        elif mode == 'bothward':
            if len(history) == 3:
                yield tuple([history[2], history[0], history[1]])
            if len(history) >= 2:
                yield tuple([history[0], history[1]])
                yield tuple([history[1], history[0]])
            if len(history) >= 1:
                yield tuple([history[0]])
        # Append element to history if sequence has more items.
        try:
            history.append(next(sequence))
        except StopIteration:
            pass
        # print(history)
        del history[0]


def padded_every_trigrams_pipeline_with_mode(text: list, mode: str):
    
    '''
    padding for preprocessing before train
    https://www.nltk.org/_modules/nltk/lm/preprocessing.html#padded_everygram_pipeline
    '''
    flatten_ = chain.from_iterable
    pad_both_ends = partial(pad_sequence, pad_left=True, left_pad_symbol="<s>",
                            pad_right=True, right_pad_symbol="</s>")
    padding_fn_trigrams = partial(pad_both_ends, n=3)
    return (
        (every_trigrams_bothward(list(padding_fn_trigrams(sent)), mode) for sent in text),
        flatten_(map(padding_fn_trigrams, text))
    )

def load_address_val_data(path):
    data = []
    text = read_data(path)
    # text = [' \<NUMBER> hàng Lước ? Quảng đường CỰ', ' 1   0   1    1   1    1    1']
    text = [t for t in text if len(t)>1]
    for i in range(0, len(text), 2):
        # if i == 102:
        #     continue #TODO
        if len(text[i])>1:
            ips = [p.strip().split() for p in extract_phrase(text[i]) if len(p.strip())>1]
            # gts = [int(label) for label in text[i + 1].split()]
            gts = [plabel.strip().split() for plabel in extract_phrase(text[i+1]) if plabel.strip().isdigit() or len(plabel.strip())>1]
            gts = [[int(label) for label in plabel] for plabel in gts]
            for j in range(len(ips)):

                assert len(ips[j]) == len(gts[j]), ' label mismatched'
                # mask = [1 if len(ips[i]>1) or ips[i].isdigit()]
                mask = [1]*len(ips[j])
                ip, mask, gt = pad_val_data(ips[j], mask, gts[j])
                data.append([ip, mask, gt])
    return data

def transform_address_to_data(text):
    ips = [p.strip().split() for p in extract_phrase(text) if len(p.strip())>1]
    data = []
    for j in range(len(ips)):
        mask = [1]*len(ips[j])
        ip, mask, _ = pad_val_data(ips[j], mask, mask)
        data.append((ip,mask))
    return data
#customize metric
# def flexible_metric(preds, gts):
#     #pred centric
#     tp, fp, fn,tn = 0,0,0,0
#     for pred, gt in zip(preds,gts):
#         for i in range(len(pred)):
#             prev = max(i-1,0)
#             next = min(i+1,len(pred))
#             if pred[i] ==1:
#                 if gt[i]==0 and gt[next]==0 and gt[prev]==0:
#                     fp+=1
#                 else:
#                     tp+=1
#             else: #if preds[i]==0
#                 if gt[i]==1 and gt[next]== 1 and gt[prev]== 1: #and to flexible but not really reasonable
#                     fn+=1
#                 else:
#                     tn+=1
#     precision = tp / (tp+fp)                
#     recall = tp / (tp+fn)
#     accuracy = (tp+tn) / (tp+fp+tn+fn)             
#     return precision, recall, accuracy

# def flexible_metric(preds, gts):
#     '''
#     gt centric intuitively makes sense more than pred centric
#     '''
#     tp, fp, fn,tn = 0,0,0,0
#     for pred, gt in zip(preds,gts):
#         for i in range(len(pred)):
#             prev = max(i-1,0)
#             next = min(i+1,len(pred)-1)
#             if gt[i] ==1:
#                 if pred[i]==1 or pred[next]==1 or pred[prev]==1:
#                     tp+=1 
#                 else:
#                     fn+=1
#             else: #if gt[i]==0
#                 if pred[i]==1 and pred[next]== 1 and pred[prev]== 1: #and to flexible but not really reasonable
#                     fp+=1
#                 else:
#                     tn+=1
#     precision = tp / (tp+fp)                
#     recall = tp / (tp+fn)
#     accuracy = (tp+tn) / (tp+fp+tn+fn)             
#     print(tp,fp,fn,tn)
#     return precision, recall, accuracy
# def flexible_metric(preds, gts):
#     '''
#     gt centric intuitively makes sense more than pred centric
#     '''
#     tp, fp, fn,tn = 0,0,0,0
#     for pred, gt in zip(preds,gts):
#         for i in range(len(pred)):
#             prev = max(i-1,0)
#             next = min(i+1,len(pred)-1)
#             if gt[i] ==1:
#                 if pred[i]==1 or pred[next]==1 or pred[prev]==1:
#                     tp+=1 
#                 else:
#                     fn+=1
#             else: #if gt[i]==0
#                 if gt[prev]==0 and gt[next]==0:
#                     if pred[i]==1:
#                         fp+=1
#                     else:
#                         tn+=1
#                 else: #[1,0,1][1,0,0],[0,0,1]
#                     pass #flexible
#     precision = tp / (tp+fp)                
#     recall = tp / (tp+fn)
#     accuracy = (tp+tn) / (tp+fp+tn+fn)
#     f1 = 2*precision*recall/(precision+recall)
#     print(cfg.E_THRESH, tp,fp,fn,tn)
#     return accuracy, precision, recall, f1

def flexible_metric(preds, gts):
    '''
    serate precision and recall
    '''
    tp_recall, fn_recall = 0,0
    tp_precision, fp_precision = 0,0
    err_list = set()
    for i, pred_and_gt in enumerate(zip(preds,gts)):
        pred, gt = pred_and_gt
        for j in range(len(pred)):
            prev = max(j-1,0)
            next = min(j+1,len(pred)-1)
            if gt[j] ==1: #recall
                if pred[j]==1 or pred[next]==1 or pred[prev]==1:
                    tp_recall+=1 
                else:
                    fn_recall+=1
                    err_list.add(i)                
    
            if pred[j] ==1: #precision
                if gt[j]==1 or gt[prev] ==1 or gt[next]==1:
                    tp_precision+=1
                else:   
                    fp_precision+=1
                    err_list.add(i)
    precision = tp_precision / (tp_precision+fp_precision)                
    recall = tp_recall / (tp_recall+fn_recall)
    # accuracy = (tp+tn) / (tp+fp+tn+fn)
    f1 = 2*precision*recall/(precision+recall)
    print(cfg.E_THRESH)
    return (precision, recall, f1), err_list

# def flexible_metric(preds, gts):
#     tp, fp, fn,tn = 0,0,0,0
#     for pred, gt in zip(preds,gts):
#         for i in range(len(pred)):
#             prev = max(i-1,0)
#             next = min(i+1,len(pred)-1)
#             if gt[i] ==1: #[0,1,0],[1,1,0],[0,1,1], [1,1,1]
#                 if pred[i]==1 or pred[next]==1 or pred[prev]==1:
#                     tp+=1 
#                 else:
#                     fn+=1
#             else: #if gt[i]==0
#                 if gt[prev] ==1 or gt[next]==1: #[1,0,0], [1,0,1],[0,0,1]
#                     if pred[i]==1:
#                         tp+=1
                    
#                 if pred[i]==0 and pred[next]==0 and pred[prev]==0:
#                     tn+=1
#                 else:
#                     fn+=1
#     precision = tp / (tp+fp)                
#     recall = tp / (tp+fn)
#     accuracy = (tp+tn) / (tp+fp+tn+fn)             
#     print(tp,fp,fn,tn)
#     return precision, recall, accuracy

# def flexible_metric(preds, gts):
#     tp, fp, fn,tn = 0,0,0,0
#     for pred, gt in zip(preds,gts):
#         for i in range(len(pred)):
#             prev = max(i-1,0)
#             next = min(i+1,len(pred)-1)
#             if gt[i] == 0 and gt[prev]==0 and gt[next]==0: #[0,0,0]
#                 if pred[i]==1: #[0,1,0],[1,1,0][1,0,1],[0,1,1], [1,1,1]
#                     fp+=1 
#                 else: #[0,0,0],[1,0,0],[0,0,1]
#                     tn+=1
#             elif gt[i] == 0 and gt[prev]==1 and gt[next]==0: #[1,0,0] or [1,0]
#                 if pred[i]==1 or pred[prev]==1:
#                     tp+=1
#                 else: #[0,0,0], [0,0,1]
#                     fn+=1
#             elif gt[i] == 0 and gt[prev]==0 and gt[next]==1: #[0,0,1] or [0,1]
#                 if pred[i]==1 or pred[next]==1:
#                     tp+=1
#                 else:
#                     fn+=1
#             else: #[0,1,0],[1,1,0][1,0,1],[0,1,1], [1,1,1]
#                 if pred[i]==1 or pred[prev]==1 or pred[next]==1:
#                     tp+=1
#                 else:
#                     fn+=1
#     precision = tp / (tp+fp)                
#     recall = tp / (tp+fn)
#     accuracy = (tp+tn) / (tp+fp+tn+fn)
#     f1 = 2*precision*recall/(precision + recall)         
#     print(tp,fp,fn,tn)
#     return accuracy, precision, recall, f1
# print(flexible_metric([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]], [[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]]))
#%%
# data_path = 'corpus/FWD_data/addresses'
# data = load_address_val_data(data_path)
# len(data)
# %%
# if __name__ == '__main__':
#     text = """Tiếng Việt được chính thức ghi nhận trong Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam 2013, tại Chương I Điều 5 Mục 3, là ngôn ngữ quốc gia của Việt Nam . Tiếng Việt bao gồm cách phát âm tiếng Việt và chữ Quốc ngữ để viết. Tuy nhiên, hiện chưa có bất kỳ văn bản nào ở cấp nhà nước quy định "giọng chuẩn" và "quốc tự" của tiế
# ng Việt . Hiện nay phần lớn các văn bản trong nước được viết theo những ""Quy định về chính tả tiếng Việt và về thuật ngữ tiếng Việt" áp dụng cho các sách giáo khoa, báo và văn bản của ngành giáo dục" nêu tại Quyết định của Bộ Giáo dục số 240/QĐ ngày 5 tháng 3 năm 1984 do những người thụ hưởng giáo dục đó sau này ra làm việc trong mọi lĩnh vực xã hội.
# """
    # text = sub_number_email_address_code_name(text)
    # print(text)
    # pass
    # text = "Trên cơ sở kết quả kiểm tra NAME hiện trạng CODE tồi UNK"
    # outs = transform_corpus_to_val_data_par([text] * 5)
    # print(outs)
    # # %%
    # import numpy as np
    # np.array(outs)[:, -1, :].tolist()

    # # # %%
    # # print(np.array(outs)[:, :-1, :].tolist())

    # # %%
    # train_data, padded_sent = padded_trigrams_pipeline([text], 'forward')
    # print([list(d) for d in train_data])
    # # %%
    # train_data, padded_sent = padded_trigrams_pipeline([text], 'backward')
    # print([list(d) for d in train_data])
    # # %%
    # train_data, padded_sent = padded_trigrams_pipeline([text], 'bothward')
    # print([list(d) for d in train_data])
    # # corpus = read_data(self.val_data_path)[:100]
    # # train_data, padded_sent = padded_trigrams_pipeline(corpus)

    # # %%

    # # %%
    # trigrams(list(padding_fn_trigrams(text))[::-1])
    # # %%

    # corpus = read_data('corpus/train_tieng_viet_cleaned.txt')[:5000]
    # val_data = transform_corpus_to_val_data_par(corpus)
    # print(val_data)
#     train_data, padded_sent = padded_every_trigrams_pipeline([text],'forward')
#     # train_data, padded_sent = padded_every_trigrams_pipeline([text],'backward')
#     # train_data, padded_sent = padded_every_trigrams_pipeline([text],'bothward')
#     print([list(d) for d in train_data], list(padded_sent))
# # %%
