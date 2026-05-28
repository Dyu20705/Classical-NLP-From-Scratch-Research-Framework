import numpy as np

class naivebayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_prior = None
        self.classes = None
        self.feature_log_prob = None

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        n_samples, n_features = X.shape
        X = np.asarray(X, dtype=np.float32)
        y = np.assarray(y, dtype=np.int32).flatten()

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_prior = np.zeros(n_classes, dtype=np.float32)
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.class_prior[idx] = len(X_cls) / n_samples
        self.class_prior = np.log(self.class_prior)
        
        self.feature_log_prob = np.zeros((n_classes, n_features), dtype=np.float32)
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            feature_count = X_cls.sum(axis=0)
            total_count = feature_count.sum()
            #Laplace smoothing to avoid log(0)
            self.feature_log_prob[idx, :] = np.log(
                (feature_count + self.alpha) / (total_count + self.alpha * n_features)
            )
        return self

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        log_prob = np.zeros((n_samples, n_classes), dtype=np.float32)
        for idx in range(n_classes):
            log_prob[:, idx] = self.class_prior[idx] + X @ self.feature_log_prob[idx, :].T
        prediction = self.classes[np.argmax(log_prob, axis=1)]
        return np.array(prediction)
    
    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        log_prob = np.zeros((n_samples, n_classes), dtype=np.float32)
        for idx in range(n_classes):
            log_prob[:, idx] = self.class_prior[idx] + X @ self.feature_log_prob[idx, :].T
        
        prob = np.exp(log_prob - np.max(log_prob, axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        return prob
    
        