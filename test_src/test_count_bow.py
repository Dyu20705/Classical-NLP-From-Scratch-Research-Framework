import numpy as np
from scipy.sparse import crs_matrix
from collections import Counter
from test_src.test_text_processor import text_processor

class count_bow:
    def __init__(self, lowercase=True, ngram=(1,1), max_features=None):
        self.lowercase = lowercase
        self.ngram = ngram
        self.max_features = max_features
        self.vocabulary = None

    def get_ngrams(self, tokens):
        min_n, max_n = self.ngram
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i + n])
                ngrams.append(ngram)
        return ngrams
    
    def fit(self, documents):
        for doc in documents:
            tokens = text_processor().process(doc)
            ngrams = self.get_ngrams(tokens)
            tk_cnter = Counter(ngrams)
        
        if self.max_features is None:
            sort_tk = sorted(tk_cnter.items(), key=lambda x: (x[1], x[0]))
        else:
            sort_tk = sorted(tk_cnter.items(), key=lambda x: (x[1], x[0]))[:self.max_features]
        
        self.vocabulary = {tk: idx for idx, (tk, _) in enumerate(sort_tk)}
        return self
    
    def transform(self, documents):
        if self.vocabulary is None:
            raise ValueError("Chua fit du lieu")
        
        ndocs = len(documents)
        nfeat = len(self.vocabulary)

        rows, cols, data = [], [], []

        for i, doc in enumerate(documents):
            tokens = text_processor().process(doc)
            ngrams = self.get_ngrams(tokens)
            tk_cnter = Counter(ngrams)

            for tk, cnt in tk_cnter.items():
                if tk in self.vocabulary:
                    idx = self.vocabulary[tk]
                    rows.append(i)
                    cols.append(idx)
                    data.append(cnt)
        X = crs_matrix((data, (rows, cols)), shape=(ndocs, nfeat), dtype=np.float32)
        return X
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
