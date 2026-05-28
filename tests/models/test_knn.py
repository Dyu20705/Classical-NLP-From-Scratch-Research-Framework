import numpy as np

from src.models.knn.knn import KNN


def test_knn_predicts_expected_labels():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    knn = KNN(k=1).fit(X, y)
    pred = knn.predict(X)
    assert np.array_equal(pred, y)
