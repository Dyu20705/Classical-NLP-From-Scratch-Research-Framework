import numpy as np


def confusion_matrix(y_true, y_pred):
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
       
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
                        
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
                                   
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
                                 
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
                           
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm, classes
