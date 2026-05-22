import numpy as np

from src.models.decision_tree.random_forest import RandomForest


def test_random_forest_fit_predict_shape():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    rf = RandomForest(n_trees=3, max_depth=3, random_state=42)
    rf.fit(X, y)
    pred = rf.predict(X)

    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})
