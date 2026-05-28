import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
\
\
\
\
\
\
       
    
    @abstractmethod
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
           
        pass
    
    @abstractmethod
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
           
        pass

    def predict_proba(self, X):
                                                                        
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba()")

    def score(self, X, y):
\
\
\
\
\
           
        y_true = np.asarray(y).flatten()
        y_pred = np.asarray(self.predict(X)).flatten()
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y and predictions must have the same length")
        return float(np.mean(y_true == y_pred))
