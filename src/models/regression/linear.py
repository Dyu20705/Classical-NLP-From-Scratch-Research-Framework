import numpy as np
from src.models.base import BaseModel


class LinearRegression(BaseModel):
    """Gradient-descent linear regression for numeric targets."""

    def __init__(self, learning_rate=0.01, n_iters=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        for i in range(self.n_iters):
            y_hat = X.dot(self.w) + self.b
            error = y_hat - y

            dw = (1.0 / n_samples) * X.T.dot(error)
            db = float((1.0 / n_samples) * np.sum(error))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if self.verbose and i % 100 == 0:
                loss = float((0.5 / n_samples) * np.sum(error ** 2))
                print(f"[LinReg] iter={i:04d}, loss={loss:.6f}")

        return self

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        return X.dot(self.w) + self.b