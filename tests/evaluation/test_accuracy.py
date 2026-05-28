import numpy as np

from src.evaluation.accuracy import accuracy


def test_accuracy_correct_value():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.75
