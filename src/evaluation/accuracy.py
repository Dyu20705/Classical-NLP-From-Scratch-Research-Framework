import numpy as np


def accuracy(y_true, y_pred):
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
       
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total
