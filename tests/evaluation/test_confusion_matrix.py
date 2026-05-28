import numpy as np

from src.evaluation.confusion_matrix import confusion_matrix


def test_confusion_matrix_shape_and_counts():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cm, classes = confusion_matrix(y_true, y_pred)

    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_true)
    assert np.array_equal(classes, np.array([0, 1]))
