import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
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
       
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0, 1)")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    split_index = int(n_samples * (1 - test_size))

                                                   
    if stratify:
        classes, class_counts = np.unique(y, return_counts=True)
        
                                              
        indices_per_class = {cls: np.where(y == cls)[0] for cls in classes}
        train_indices = []
        test_indices = []
        
        for cls in classes:
            cls_indices = indices_per_class[cls]
            np.random.shuffle(cls_indices)
            
            n_train = int(len(cls_indices) * (1 - test_size))
                                                                                     
            if len(cls_indices) >= 2:
                n_train = max(1, min(n_train, len(cls_indices) - 1))
            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
                                        
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    else:
                             
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test