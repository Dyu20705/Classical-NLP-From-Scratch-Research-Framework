import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter


class CountVectorizer:
    """
    Convert a collection of text documents to a sparse matrix of token counts.
    
    This implementation:
    - Tokenizes by whitespace
    - Supports n-grams (unigrams, bigrams, etc.)
    - Builds vocabulary only from training data
    - Produces sparse CSR matrices to save memory
    
    Parameters:
    -----------
    lowercase : bool, default=True
        If True, convert all text to lowercase before tokenization.
    ngram_range : tuple, default=(1, 1)
        The lower and upper boundary of n-gram range (min_n, max_n).
        Example: (1, 2) for unigrams and bigrams.
    max_features : int or None, default=None
        If not None, keep top-k most frequent tokens.
        If None, keep all tokens.
    
    Attributes:
    -----------
    vocabulary_ : dict
        Mapping from token string to feature index (built during fit).
    """
    
    def __init__(self, lowercase=True, ngram_range=(1, 1), max_features=None):
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary_ = None
    
    def _get_ngrams(self, tokens):
        """
        Generate n-grams from a list of tokens.
        
        Parameters:
        -----------
        tokens : list of str
            Tokenized words.
        
        Returns:
        --------
        ngrams : list of str
            Generated n-grams with format "word1_word2" for bigrams, etc.
        """
        min_n, max_n = self.ngram_range
        ngrams = []
        
        # Add n-grams of all sizes in range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i + n])
                ngrams.append(ngram)
        
        return ngrams
    
    def fit(self, documents):
        """
        Build vocabulary from documents.
        
        Parameters:
        -----------
        documents : list of str
            List of text documents.
        
        Returns:
        --------
        self : CountVectorizer
            Fitted vectorizer instance.
        """
        # Count all tokens across all documents
        token_counts = Counter()
        
        for doc in documents:
            # Preprocess text
            if self.lowercase:
                doc = doc.lower()
            
            # Tokenize by whitespace
            tokens = doc.split()
            
            # Generate n-grams
            ngrams = self._get_ngrams(tokens)
            
            # Count tokens
            token_counts.update(ngrams)
        
        # Sort by frequency and select top-k if max_features specified
        if self.max_features is None:
            sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
        else:
            sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:self.max_features]
        
        # Create vocabulary mapping (token -> index)
        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
        
        return self
    
    def transform(self, documents):
        """
        Transform documents to sparse count matrix.
        
        Parameters:
        -----------
        documents : list of str
            List of text documents to transform.
        
        Returns:
        --------
        X : scipy.sparse.csr_matrix of shape (n_documents, n_features)
            Sparse document-term matrix with raw token counts.
            Each row represents a document, each column a token from vocabulary.
        """
        if self.vocabulary_ is None:
            raise ValueError("CountVectorizer must be fitted before transform. Call fit() first.")
        
        n_documents = len(documents)
        n_features = len(self.vocabulary_)
        
        # Store sparse matrix data: row indices, column indices, values
        row_indices = []
        col_indices = []
        values = []
        
        for doc_idx, doc in enumerate(documents):
            # Preprocess text
            if self.lowercase:
                doc = doc.lower()
            
            # Tokenize by whitespace
            tokens = doc.split()
            
            # Generate n-grams
            ngrams = self._get_ngrams(tokens)
            
            # Count occurrences of each n-gram in this document
            ngram_counts = Counter(ngrams)
            
            # Add to sparse matrix
            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary_:
                    row_indices.append(doc_idx)
                    col_indices.append(self.vocabulary_[ngram])
                    values.append(count)
        
        # Create sparse CSR matrix
        X = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_documents, n_features),
            dtype=np.float32
        )
        
        return X
    
    def fit_transform(self, documents):
        """
        Fit vectorizer and transform documents in one step.
        
        Parameters:
        -----------
        documents : list of str
            List of text documents.
        
        Returns:
        --------
        X : scipy.sparse.csr_matrix
            Sparse document-term count matrix.
        """
        self.fit(documents)
        return self.transform(documents)