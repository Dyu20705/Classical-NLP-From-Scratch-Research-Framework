import numpy as np
from collections import Counter
from src.models.base import BaseModel

def euclidean_distance(x1, x2):
    return sum((x1 - x2) ** 2) ** 0.5

class KNN(BaseModel):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y).flatten()
        return self

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        prediction = [self.predict_class(row) for row in X]
        return np.array(prediction)

    def predict_class(self, new_point):
        # Compute distances between x and all examples in the training set
        distances = np.array([euclidean_distance(new_point, x_train) for x_train in self.X_train])

        # Sort by distance and return indices of the first k neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Return the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common