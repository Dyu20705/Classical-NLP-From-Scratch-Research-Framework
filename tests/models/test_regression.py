import numpy as np

from src.models.regression.linear import LinearRegression
from src.models.regression.logistic import LogisticRegression


def test_linear_regression_fit_predict_shape():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = LinearRegression(learning_rate=0.05, n_iters=500)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.mean((pred - y) ** 2) < 2.0


def test_logistic_regression_binary_predictions():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iters=300)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})
