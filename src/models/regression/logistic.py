import numpy as np
from src.models.base import BaseModel


class LogisticRegression(BaseModel):
    """Binary logistic regression (from-scratch) for sentiment classification."""

    def __init__(self, learning_rate=0.1, n_iters=1000, threshold=0.5, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).flatten().astype(np.float32)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        for i in range(self.n_iters):
            logits = X.dot(self.w) + self.b
            probs = self._sigmoid(logits)

            dw = (1.0 / n_samples) * X.T.dot(probs - y)
            db = float((1.0 / n_samples) * np.sum(probs - y))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if self.verbose and i % 100 == 0:
                eps = 1e-12
                p = np.clip(probs, eps, 1 - eps)
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                print(f"[LogReg] iter={i:04d}, loss={loss:.6f}")

        return self

    def predict_proba(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        probs = self._sigmoid(X.dot(self.w) + self.b)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
