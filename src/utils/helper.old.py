import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split the dataset into training and testing sets with optional stratification.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features dataset.
    y : array-like of shape (n_samples,)
        Target variable.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Controls the randomness of the split for reproducibility.
    stratify : bool, default=True
        If True, maintain class distribution in train/test splits.
        If False, perform simple random split.

    Returns:
    --------
    X_train, X_test, y_train, y_test : array-like
        Split datasets maintaining class distribution if stratify=True.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    split_index = int(n_samples * (1 - test_size))

    # Stratified split: maintain class distribution
    if stratify:
        classes, class_counts = np.unique(y, return_counts=True)
        
        # For each class, split proportionally
        indices_per_class = {cls: np.where(y == cls)[0] for cls in classes}
        train_indices = []
        test_indices = []
        
        for cls in classes:
            cls_indices = indices_per_class[cls]
            np.random.shuffle(cls_indices)
            
            n_train = int(len(cls_indices) * (1 - test_size))
            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Shuffle train and test indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    else:
        # Simple random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }