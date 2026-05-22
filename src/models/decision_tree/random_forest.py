from src.models.decision_tree.decision_tree import DecisionTree
import numpy as np
from collections import Counter
from src.models.base import BaseModel


class RandomForest(BaseModel):
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        np.random.seed(self.random_state)
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X)

        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions