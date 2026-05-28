import numpy as np
from src.models.base import BaseModel


class NaiveBayes(BaseModel):
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
       
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = None
        self.feature_log_probs = None
    
    def fit(self, X, y):
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
           
                                                                                
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).flatten()
        
        n_samples, n_features = X.shape
        
                            
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
                               
        self.class_priors = np.zeros(n_classes)
        self.feature_log_probs = np.zeros((n_classes, n_features))
        
                                            
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            
                                                              
            self.class_priors[idx] = len(X_cls) / n_samples
            
                                                                                  
                                                                                                              
            feature_counts = X_cls.sum(axis=0)                                                 
            total_count = feature_counts.sum()
            
                                               
            self.feature_log_probs[idx, :] = np.log(
                (feature_counts + self.alpha) / (total_count + self.alpha * n_features)
            )
        
                                           
        self.class_priors = np.log(self.class_priors)
        
        return self
    
    def predict(self, X):
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
           
                                           
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float32)
        
                                                             
        log_posteriors = self._compute_log_posteriors(X)
        
                                                      
        predictions = self.classes[np.argmax(log_posteriors, axis=1)]
        
        return np.array(predictions)
    
    def _compute_log_posteriors(self, X):
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
           
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_posteriors = np.zeros((n_samples, n_classes))
        
                                                                                                       
        for idx in range(n_classes):
                                   
            log_prior = self.class_priors[idx]
            
                                                                       
                                                                       
            log_likelihood = np.dot(X, self.feature_log_probs[idx, :])
            
            log_posteriors[:, idx] = log_prior + log_likelihood
        
        return log_posteriors
    
    def predict_proba(self, X):
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
           
                                           
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float32)
        
                                
        log_posteriors = self._compute_log_posteriors(X)
        
                                                                  
                                                       
        log_posteriors_stable = log_posteriors - log_posteriors.max(axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors_stable)
        
                               
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        
        return posteriors