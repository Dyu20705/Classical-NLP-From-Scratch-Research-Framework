import numpy as np
from scipy.sparse import csr_matrix


class TfidfTransformer:
    """
    Transform sparse count matrix to TF-IDF weighted matrix.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) is a numerical statistic
    that reflects how important a term is to a document in a collection.
    
    Formula:
    --------
    idf(t) = log((N + 1) / (df(t) + 1)) + 1
    where:
    - N = total number of documents
    - df(t) = number of documents containing term t
    
    tfidf(d, t) = tf(d, t) * idf(t)
    
    Then L2 normalization is applied to each row (document).
    
    Parameters:
    -----------
    smooth_idf : bool, default=True
        If True, add 1 to document frequencies to avoid zero divisions.
    
    Attributes:
    -----------
    idf_ : np.ndarray of shape (n_features,)
        IDF weight for each feature (computed during fit).
    """
    
    def __init__(self, smooth_idf=True):
        self.smooth_idf = smooth_idf
        self.idf_ = None
    
    def fit(self, X):
        """
        Learn IDF weights from sparse count matrix.
        
        Parameters:
        -----------
        X : scipy.sparse.csr_matrix of shape (n_samples, n_features)
            Sparse count matrix, typically output from CountVectorizer.
        
        Returns:
        --------
        self : TfidfTransformer
            Fitted transformer instance.
        """
        if not hasattr(X, 'toarray'):
            raise TypeError("X must be a sparse matrix (scipy.sparse)")
        
        n_samples, n_features = X.shape
        
        # Count document frequency: how many documents contain each feature
        # Convert to binary (>0) and sum across documents
        df = np.array(X.astype(bool).sum(axis=0)).ravel()
        
        # Compute IDF weights with smoothing
        if self.smooth_idf:
            idf = np.log((n_samples + 1.0) / (df + 1.0)) + 1.0
        else:
            idf = np.log(n_samples / df) + 1.0
        
        self.idf_ = idf
        
        return self
    
    def transform(self, X):
        """
        Transform sparse count matrix to TF-IDF weighted matrix.
        
        Parameters:
        -----------
        X : scipy.sparse.csr_matrix of shape (n_samples, n_features)
            Sparse count matrix to transform.
        
        Returns:
        --------
        X_tfidf : scipy.sparse.csr_matrix of shape (n_samples, n_features)
            TF-IDF weighted sparse matrix with L2 normalization applied.
        """
        if self.idf_ is None:
            raise ValueError("TfidfTransformer must be fitted before transform. Call fit() first.")
        
        if not hasattr(X, 'toarray'):
            raise TypeError("X must be a sparse matrix (scipy.sparse)")
        
        # Make a copy to avoid modifying original
        X_tfidf = X.copy().astype(np.float32)
        
        # Apply IDF weighting to each feature
        # Multiply each column by its IDF weight
        X_tfidf = X_tfidf.multiply(self.idf_)
        
        # L2 normalization: normalize each row to have unit L2 norm
        # norm = sqrt(sum(x^2)) for each row
        norms = np.sqrt(np.array(X_tfidf.power(2).sum(axis=1)).ravel())
        
        # Avoid division by zero
        norms[norms == 0.0] = 1.0
        
        # Normalize rows by dividing each row by its norm
        # Convert norms to diagonal matrix for efficient sparse multiplication
        norms_inv = 1.0 / norms
        norms_diag = csr_matrix((norms_inv, (np.arange(len(norms_inv)), np.arange(len(norms_inv)))),
                                shape=(len(norms_inv), len(norms_inv)))
        X_tfidf = norms_diag @ X_tfidf
        
        return X_tfidf.tocsr()
    
    def fit_transform(self, X):
        """
        Fit transformer and transform data in one step.
        
        Parameters:
        -----------
        X : scipy.sparse.csr_matrix
            Sparse count matrix.
        
        Returns:
        --------
        X_tfidf : scipy.sparse.csr_matrix
            TF-IDF weighted sparse matrix with L2 normalization.
        """
        self.fit(X)
        return self.transform(X)