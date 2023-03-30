# %%
import re
from tools.utils import read_data, transform_corpus_to_val_data_par, flatten, padded_trigrams_pipeline, read_pkl, padded_every_trigrams_pipeline_with_mode, load_address_val_data, flexible_metric, transform_address_to_data
import argparse
import config_ngram as cfg  # import NGRAMS, DATA_DIR
import os
from ngrams_model import NgramsModel
from sklearn.metrics import classification_report
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import nltk
import time
# %%
def get_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--istrain', type=bool, default=False, help='train if specified, otherwise do validate')
    parse.add_argument('--mode', type=str, default='bothward', help='mode of trigrams use for training', choices=['forward','bothward','backward','enhance']) #enhance = combine 3 models
    return parse.parse_args()


class Runner():
    def __init__(self, istrain, mode):
        self.ngrams_model = NgramsModel(istrain,mode)
        self.istrain = istrain
        self.mode = mode
        # self.train_data_path = os.path.join(cfg.DATA_DIR, 'train_tieng_viet_cleaned_tokenized.pkl')
        self.train_data_path = cfg.ADDRESSES_TRAIN_PATH_TOKENIZED
        # self.val_data_path = os.path.join(cfg.DATA_DIR, 'train_tieng_viet_cleaned.txt')
        self.val_data_path = cfg.ADDRESSES_VAL_PATH_PREPROCESSED

    def load_data(self):
        if self.istrain:
            train_data = []
            for data_path in self.train_data_path:
                train_data.extend(read_pkl(data_path))
            return train_data
        else:
            return read_data(self.val_data_path)

    def save_model(self):
        self.ngrams_model.save_model()

    def train_every_trigrams(self):
        start = time.time()
        # read train data
        corpus = self.load_data()
        print('[INFO]: Data loaded')
        train_data, padded_sent = padded_every_trigrams_pipeline_with_mode(corpus, self.mode)
        print('[INFO]: Data preprocessed')
        del corpus
        # train _model
        print(f'[INFO]: Training model {self.mode}...')
        self.ngrams_model.fit(self.mode, train_data, padded_sent)
        print('[INFO]: Model trained, number of vocab: ', len(self.ngrams_model.vocab))
        print(f'Took {time.time() - start} s')

    # def train_trigrams(self, mode: str):
    #     start = time.time()
    #     # read train data
    #     corpus = self.load_data()
    #     print('[INFO]: Data loaded')
    #     train_data, padded_sent = padded_trigrams_pipeline(corpus, mode)
    #     del corpus
    #     print('[INFO]: Data preprocessed')
    #     # train _model
    #     print('[INFO]: Training model...')
    #     # self.ngrams_model.fit_par(train_data, padded_sent, nproc=24)
    #     self.ngrams_model.fit(train_data, padded_sent)
    #     print('[INFO]: Model trained, number of vocab: ', len(self.ngrams_model.model.vocab))
    #     print(f'Cost {time.time() - start} s')

    def predict(self, ip_and_mask,val_mode):
        ip, mask = ip_and_mask
        # ip = ['<s>', '<s>', '<NUMBER>', 'bình', 'thạnh', 'hcm', '</s>', '</s>']#test
        # ip = ['<s>', '<s>', '<NUMBER>', 'lý', 'thương', 'kiệt', '<CODE>', '</s>', '</s>']
        # ip = ['<s>', '<s>', 'số', '<NUMBER>', 'hoàng', 'văn', 'thị', 'phường', 'sin', 'liên', 'quận', 'đống', 'đà', '</s>', '</s>']
        # mask = [1]*len(ip)
        pred = self.ngrams_model.predict(ip, mask, val_mode)
        # for i, word in enumerate(ip): #post process #TODO
        #     if word == '<LONG_NUMBER>':
        #         pred[i]=1
        return pred

    def predict_par(self, ips_and_masks, nproc=4):
        with Pool(nproc) as p:
            return list(tqdm(p.imap(self.predict, ips_and_masks), total=len(ips_and_masks), desc='Predicting...'))

    def predict_batch(self, ips_and_masks, val_mode):
        # return [self.predict(ip_and_mask, val_mode) for ip_and_mask in tqdm(ips_and_masks)]
        return [self.predict(ip_and_mask, val_mode) for ip_and_mask in ips_and_masks]

    def validate(self):
        corpus = self.load_data(train=False)
        val_data = transform_corpus_to_val_data_par(corpus, nproc=4)
        del corpus
        # preds, gts = [], []
        # for ip, mask, gt in val_data:
        #     preds.extend(self.model.predict(ip, mask))
        #     gts.extend(gt)
        #     print('-' * 100)
        #     # print(ip)
        #     print('predict: \t', self.model.predict(ip, mask))
        #     print('groundtruth: \t', gt)
        # print(classification_report(gts, preds))
        val_data = np.asarray(val_data, dtype=object)
        ips_and_masks = val_data[:, :-1].tolist()
        gts = val_data[:, -1].tolist()
        # preds = self.predict_par(ips_and_masks, nproc=24)
        preds = self.predict_batch(ips_and_masks)
        for i in range(len(preds)):
            print('-' * 100)
            print(ips_and_masks[i])
            print(preds[i])
            print(gts[i])
        print(classification_report(flatten(gts), flatten(preds)))

    def validate_address(self):
        val_data = load_address_val_data(self.val_data_path)
        val_data = np.asarray(val_data, dtype=object)
        ips_and_masks = val_data[:, :-1].tolist()
        gts = val_data[:, -1].tolist()
        # preds = self.predict_par(ips_and_masks, nproc=24)
        preds = self.predict_batch(ips_and_masks, self.mode)
        gts_filtered = []
        preds_filtered = []
        for i in range(len(gts)):
            gt_filtered = []
            pred_filtered = []
            for j in range(len(gts[i])):
                if gts[i][j]!=2:
                    gt_filtered.append(gts[i][j])
                    pred_filtered.append(preds[i][j])
            gts_filtered.append(gt_filtered)
            preds_filtered.append(pred_filtered)
        # preds = [[0]*len(gt) for gt in gts]
        
        metric, err_list = flexible_metric(preds_filtered,gts_filtered)
        for i in err_list:
            print('-'*100)
            print(ips_and_masks[i][0])
            print(preds[i])
            print(gts[i])
        
        # print(classification_report(flatten(gts), flatten(preds)))
        # print(metric)
        # print('Undetectable')
        print(classification_report(flatten(gts_filtered), flatten(preds_filtered)))
        print(metric)
        
    def inference_address(self, text:str):
        ips_and_masks = transform_address_to_data(text)
        preds = self.predict_batch(ips_and_masks, self.mode)
        preds = [pred[2:-2] for pred in preds]
        return flatten(preds)
    # def inference_address_batch(self, texts):
        
def main():
    # trainer = Trainer('models/kneserney_train_tieng_viet_model.pkl')
    # trainer.validate()

    # assert nltk.__version__ == '3.5', 'Please select the right version for nltk!'
    # trainer = Trainer()
    # trainer.train_trigrams()
    # trainer.save_model('models/kneserney_train_tieng_viet_trigrams_nltk35.pkl')

    # assert nltk.__version__ == '3.7', 'Please select the right version for nltk!'
    # trainer = Trainer()
    # trainer.train_everygrams()
    # trainer.save_model('models/kneserney_train_tieng_viet_trigrams_nltk37.pkl')
    assert nltk.__version__ == '3.5', 'Please select the right version for nltk!'
    arg = get_arg()
    trainer = Runner(arg.istrain, arg.mode)
    if arg.istrain:
        # trainer.train_trigrams(arg.train_mode)
        trainer.train_every_trigrams()
        trainer.save_model()
        # trainer.validate()
    else:
        # trainer.validate()
        trainer.validate_address()


# %%
if __name__ == "__main__":
    main()
    pass
# %%
# # %%
# corpus = read_pkl(os.path.join(DATA_DIR, 'train_tieng_viet_cleaned.pkl'))
# #%%
# train_data, padded_sent = padded_everygram_pipeline(NGRAMS, corpus[:5])

# #%%
# for i in next(train_data):
#     print(i)
# # %%
# vi_model = KneserNeyInterpolated(NGRAMS)
# vi_model.fit(train_data, padded_sent)
# #%%
# len(vi_model.vocab)
# # %%
# vi_model.vocab
# # %%
# #save model
# model_file = os.path.join('models', 'kneserney_train_tieng_viet_model.pkl')
# save_pkl(vi_model, model_file)
# %%
# trainer = Trainer()
# trainer.load_model('models/kneserney_train_tieng_viet_trigrams_forward_nltk35.pkl')
# # trainer.load_model('models/backup/kneserney_train_tieng_viet_nltk37_everygrams.pkl')
# trainer.validate()
# %%
