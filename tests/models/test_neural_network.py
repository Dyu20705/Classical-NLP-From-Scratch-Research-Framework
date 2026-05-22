import numpy as np

from src.models.neural_network.neural_network import NeuralNetworkClassifier


def test_neural_network_fit_predict_shapes():
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0])

    model = NeuralNetworkClassifier(hidden_size=8, learning_rate=0.1, n_iters=200, random_state=1)
    model.fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert pred.shape == y.shape
    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
