import numpy as np

from src.models.decision_tree.decision_tree import DecisionTree


def test_decision_tree_fit_predict_shape():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    dt = DecisionTree(max_depth=3, min_samples_split=2)
    dt.fit(X, y)
    pred = dt.predict(X)

    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})


def test_decision_tree_handles_sparse_like_input():
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
    y = np.array([1, 0, 1])

    dt = DecisionTree(max_depth=2)
    dt.fit(X, y)
    pred = dt.predict(X)

    assert pred.shape == y.shape
