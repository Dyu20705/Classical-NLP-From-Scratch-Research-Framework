import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter


class CountVectorizer:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    
    def __init__(self, lowercase=True, ngram_range=(1, 1), max_features=None):
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary_ = None
    
    def _get_ngrams(self, tokens):
\
\
\
\
\
\
\
\
\
\
\
\
           
        min_n, max_n = self.ngram_range
        ngrams = []
        
                                           
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i + n])
                ngrams.append(ngram)
        
        return ngrams
    
    def fit(self, documents):
\
\
\
\
\
\
\
\
\
\
\
\
           
                                               
        token_counts = Counter()
        
        for doc in documents:
            if doc is None:
                doc = ''
            if not isinstance(doc, str):
                doc = str(doc)

                             
            if self.lowercase:
                doc = doc.lower()
            
                                    
            tokens = doc.split()
            
                              
            ngrams = self._get_ngrams(tokens)
            
                          
            token_counts.update(ngrams)
        
                                                                      
        if self.max_features is None:
            sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
        else:
            sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))[:self.max_features]
        
                                                    
        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
        
        return self
    
    def transform(self, documents):
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if self.vocabulary_ is None:
            raise ValueError("CountVectorizer must be fitted before transform. Call fit() first.")
        
        n_documents = len(documents)
        n_features = len(self.vocabulary_)
        
                                                                       
        row_indices = []
        col_indices = []
        values = []
        
        for doc_idx, doc in enumerate(documents):
            if doc is None:
                doc = ''
            if not isinstance(doc, str):
                doc = str(doc)

                             
            if self.lowercase:
                doc = doc.lower()
            
                                    
            tokens = doc.split()
            
                              
            ngrams = self._get_ngrams(tokens)
            
                                                               
            ngram_counts = Counter(ngrams)
            
                                  
            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary_:
                    row_indices.append(doc_idx)
                    col_indices.append(self.vocabulary_[ngram])
                    values.append(count)
        
                                  
        X = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_documents, n_features),
            dtype=np.float32
        )
        
        return X
    
    def fit_transform(self, documents):
\
\
\
\
\
\
\
\
\
\
\
\
           
        self.fit(documents)
        return self.transform(documents)