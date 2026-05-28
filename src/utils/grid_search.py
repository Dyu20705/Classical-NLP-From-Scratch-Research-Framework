from __future__ import annotations

from itertools import product

import numpy as np

from src.evaluation.accuracy import accuracy
from src.models.decision_tree.decision_tree import DecisionTree
from src.models.naive_bayes.naive_bayes import NaiveBayes


param_grid_decision_tree = {
    "max_depth": [3, 5, 8, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 15, 20, 30, 50],
}

param_grid_naive_bayes = {
    "alpha": [0.1, 0.5, 1.0, 1.5, 2.0],
}


def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _make_folds(n_samples, cv=5, random_state=42):
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to run cross validation")

    n_splits = min(cv, n_samples)
    if n_splits < 2:
        raise ValueError("cv must be at least 2")

    indices = np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    return np.array_split(indices, n_splits)


def _cross_val_accuracy(model_factory, X, y, cv=5, random_state=42):
    X = _to_dense(X)
    y = np.asarray(y).flatten()

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")

    folds = _make_folds(len(X), cv=cv, random_state=random_state)
    scores = []

    for fold_idx in range(len(folds)):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(len(folds)) if i != fold_idx])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        scores.append(accuracy(y_val, predictions))

    return float(np.mean(scores))


def grid_search_naive_bayes(X, y, param_grid=None, cv=5, random_state=42, verbose=True):
    """Manual grid search for the local NaiveBayes implementation."""
    grid = param_grid or param_grid_naive_bayes
    alpha_values = grid.get("alpha", [1.0])

    best_score = -1.0
    best_params = None
    results = []

    for alpha in alpha_values:
        def factory(current_alpha=alpha):
            return NaiveBayes(alpha=current_alpha)

        score = _cross_val_accuracy(factory, X, y, cv=cv, random_state=random_state)
        params = {"alpha": alpha}
        results.append({"params": params, "score": score})

        if verbose:
            print(f"Testing NaiveBayes alpha={alpha} -> Mean Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params

    if verbose:
        print("\n--- NAIVE BAYES BEST RESULT ---")
        print("Best parameters:", best_params)
        print("Best score:", best_score)

    return {"best_params": best_params, "best_score": best_score, "results": results}


def grid_search_decision_tree(X, y, param_grid=None, cv=5, random_state=42, verbose=True):
    """Manual grid search for the local DecisionTree implementation."""
    grid = param_grid or param_grid_decision_tree
    max_depth_values = grid.get("max_depth", [3])
    min_samples_split_values = grid.get("min_samples_split", [2])

    best_score = -1.0
    best_params = None
    results = []

    for max_depth, min_samples_split in product(max_depth_values, min_samples_split_values):
        def factory(current_max_depth=max_depth, current_min_samples_split=min_samples_split):
            return DecisionTree(
                max_depth=current_max_depth,
                min_samples_split=current_min_samples_split,
            )

        score = _cross_val_accuracy(factory, X, y, cv=cv, random_state=random_state)
        params = {"max_depth": max_depth, "min_samples_split": min_samples_split}
        results.append({"params": params, "score": score})

        if verbose:
            print(
                "Testing DecisionTree "
                f"max_depth={max_depth}, min_samples_split={min_samples_split} -> "
                f"Mean Accuracy: {score:.4f}"
            )

        if score > best_score:
            best_score = score
            best_params = params

    if verbose:
        print("\n--- DECISION TREE BEST RESULT ---")
        print("Best parameters:", best_params)
        print("Best score:", best_score)

    return {"best_params": best_params, "best_score": best_score, "results": results}


def main():
    raise SystemExit(
        "Import grid_search_naive_bayes or grid_search_decision_tree and call them from your pipeline."
    )


if __name__ == "__main__":
    main()
