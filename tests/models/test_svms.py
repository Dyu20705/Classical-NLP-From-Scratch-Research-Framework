import numpy as np

from src.models.svms.svm import SVM


def test_svm_binary_output_shape():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = SVM(learning_rate=0.01, n_iters=200)
    clf.fit(X, y)
    pred = clf.predict(X)

    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})
