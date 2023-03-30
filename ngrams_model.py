# %%
import config_ngram as cfg
import os
from nltk.lm import KneserNeyInterpolated
from tools.utils import read_pkl, save_pkl
import time
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats.mstats import gmean
from nltk.lm.vocabulary import Vocabulary
# %%


class NgramsModel():
    def __init__(self, istrain :bool = False, mode: str='bothward', ngrams=cfg.NGRAMS):
        self.ngrams = ngrams
        self.istrain = istrain
        self.mode = mode
        self._dmodels = self.load_model()
        
    @property
    def model(self):
        return self._dmodels[self.mode] if self.mode != 'enhance' else self._dmodels
    
    @property
    def vocab(self):
        return self._dmodels[self.mode].vocab if self.mode!='enhance' else self._dmodels['forward'].vocab
            


    def load_model(self):
        dmodels = {}
        if self.mode =='enhance':
            for k,v in cfg.MODEL_PATH.items():
                dmodels[k] = read_pkl(v) if not self.istrain else KneserNeyInterpolated(self.ngrams)
        else:
            dmodels[self.mode] = read_pkl(cfg.MODEL_PATH[self.mode]) if not self.istrain else KneserNeyInterpolated(self.ngrams)
        return dmodels

    def save_model(self):
        for mode in self._dmodels:
            save_pkl(self._dmodels[mode], cfg.MODEL_PATH[mode])

    def fit(self, mode, train_data, padded_sent):
        assert mode != 'enhance', 'Training supports only 1 type of model at a time'
        self._dmodels[mode].fit(train_data, padded_sent)

    def score(self, mode, word, context):
        # try:
        assert mode !='enhance', 'Scoring supports only 1 type of model at a time'
        return self._dmodels[mode].score(word, context)
        # except ZeroDivisionError:
        #     return None

    def predict(self, lwords, lmasks, mode):
        '''
        0: not error
        1: error
        '''
        mode = mode or self.mode
        # start = time.time()fff
        lres = [0] * len(lwords)
        # start = self.ngrams if self.enhance_mode else self.ngrams * 2 - 1
        # end = len(lwords) - self.ngrams + 1 if self.enhance_mode else len(lwords) - self.ngrams * 2 + 1
        start = self.ngrams -1
        end = len(lwords) - self.ngrams + 1
        for i in range(start, end):
            # print(i, 'loop', time.time() - start, 's')
            if lmasks[i]:
                scores = []
                if mode in ['forward', 'enhance']:
                    # scores.append(self.score('forward', lwords[i+1], (lwords[i - 1], lwords[i]))) #w0 w-2 w-1
                    scores.append(self.score('forward', lwords[i], (lwords[i - 2], lwords[i-1]))) #w0 w-2 w-1
                if mode in ['bothward', 'enhance']:
                    scores.append(self.score('bothward', lwords[i], (lwords[i + 1], lwords[i - 1]))) #w0 w-2 w-1
                if mode in ['backward', 'enhance']:
                    # scores.append(self.score('backward', lwords[i-1], (lwords[i+1], lwords[i]))) #w0 w-2 w-1
                    scores.append(self.score('backward', lwords[i], (lwords[i+2], lwords[i+1]))) #w0 w-2 w-1
                # p = sum(scores)/len(scores)
                p = gmean([s for s in scores if s is not None])
                # print('score', time.time() - start, 's')
                if p < cfg.E_THRESH :
                    lres[i] = 1
        return lres
    # def predict_old(self, lwords, lmasks):
    #     lres = [0] * len(lwords)
    #     if NgramsModel.score(self._model, lwords[1], [lwords[0]]) > E_THRESH:
    #         lres[1] = 1
    #     for i in range(2, len(lwords) - 1):
    #         if lmasks[i]:
    #             p = NgramsModel.score(self._model,lwords[i], [lwords[i - 1], lwords[i - 2]])
    #             if p < E_THRESH:
    #                 lres[i] = 1
    #     return lres
    
    # def _fit_one(self, sent):
    #     try:
    #         self._model.counts.update(self._model.vocab.lookup(sent))
    #     except TypeError:
    #         pass

    # def fit_par(self, text, vocabulary_text=None, nproc=4):
    #     ''' not worked'''
    #     # https://github.com/nltk/nltk/blob/6f18391a82ce20218f84bb4e8524a5fbaca354e9/nltk/lm/api.py#L72
    #     if not self._model.vocab:
    #         if vocabulary_text is None:
    #             raise ValueError(
    #                 "Cannot fit without a vocabulary or text to create it from."
    #             )
    #         self._model.vocab.update(vocabulary_text)
    #     text = [list(t) for t in text]
    #     with Pool(nproc) as p:
    #         _ = list(tqdm(p.imap(self._fit_one, text), total=len(list(text)), desc='Training in paralell...'))
    
    


# %% unit test
# if __name__ == '__main__':
#     model = NgramModel('models/kneserney_train_tieng_viet_model.pkl', 3)
# # %%
# if __name__ == '__main__':
#     model = NgramModel()
#     print(len(model.model.vocab))
# %%
