"""
Simple feed-forward neural network (from-scratch, NumPy only).

Adapted from the MNIST scratch notebook into a reusable model class
for this repository's sentiment experiments.

Notes:
- Input shape expected by public API: (n_samples, n_features)
- Internally we keep matrix as (n_features, n_samples) to match
  the original scratch math.
- Supports binary and multi-class classification via softmax output.
"""

import numpy as np
from src.models.base import BaseModel


class NeuralNetworkClassifier(BaseModel):
    def __init__(
        self,
        hidden_size=32,
        learning_rate=0.05,
        n_iters=500,
        random_state=42,
        verbose=False,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.verbose = verbose

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.classes_ = None
        self.loss_history_ = []

    @staticmethod
    def _relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def _softmax(z):
        z_stable = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def _cross_entropy(y_one_hot, probs):
        eps = 1e-12
        clipped = np.clip(probs, eps, 1.0)
        return -np.mean(np.sum(y_one_hot * np.log(clipped), axis=0))

    def _init_params(self, n_features, n_classes):
        rng = np.random.default_rng(self.random_state)
        # He-like small random init for hidden layer
        self.W1 = (rng.standard_normal((self.hidden_size, n_features)).astype(np.float32)
                   * np.sqrt(2.0 / max(1, n_features)))
        self.b1 = np.zeros((self.hidden_size, 1), dtype=np.float32)
        self.W2 = (rng.standard_normal((n_classes, self.hidden_size)).astype(np.float32)
                   * np.sqrt(2.0 / max(1, self.hidden_size)))
        self.b2 = np.zeros((n_classes, 1), dtype=np.float32)

    def _forward(self, x_t):
        z1 = self.W1.dot(x_t) + self.b1
        a1 = self._relu(z1)
        z2 = self.W2.dot(a1) + self.b2
        a2 = self._softmax(z2)
        return z1, a1, z2, a2

    def fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).flatten()

        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self._init_params(n_features, n_classes)

        # Convert to (features, samples) for math consistency with scratch derivation
        x_t = X.T

        y_one_hot = np.zeros((n_classes, n_samples), dtype=np.float32)
        y_one_hot[y_idx, np.arange(n_samples)] = 1.0

        for i in range(self.n_iters):
            z1, a1, z2, a2 = self._forward(x_t)

            # Backward pass
            dz2 = a2 - y_one_hot
            dW2 = (1.0 / n_samples) * dz2.dot(a1.T)
            db2 = (1.0 / n_samples) * np.sum(dz2, axis=1, keepdims=True)
            dz1 = self.W2.T.dot(dz2) * self._relu_deriv(z1)
            dW1 = (1.0 / n_samples) * dz1.dot(x_t.T)
            db1 = (1.0 / n_samples) * np.sum(dz1, axis=1, keepdims=True)

            # Update
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if self.verbose and (i % 50 == 0 or i == self.n_iters - 1):
                loss = self._cross_entropy(y_one_hot, a2)
                self.loss_history_.append(loss)
                print(f"[NN] iter={i:04d}, loss={loss:.6f}")

        return self

    def predict_proba(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        _, _, _, a2 = self._forward(X.T)
        return a2.T

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
