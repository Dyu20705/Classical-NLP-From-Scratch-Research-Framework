import numpy as np

from src.utils.helper import train_test_split


def test_train_test_split_stratified_preserves_classes():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=True
    )

    assert len(X_train) + len(X_test) == len(X)
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}
