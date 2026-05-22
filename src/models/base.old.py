import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Defines the interface that all models must implement:
    - fit: Train the model on training data
    - predict: Make predictions on new data
    """
    
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on training data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training target labels.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Features to predict on.
        
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        pass
