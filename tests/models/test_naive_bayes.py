import numpy as np

from src.feature_extraction.count_vectorizer import CountVectorizer
from src.models.naive_bayes.naive_bayes import NaiveBayes


def test_naive_bayes_fit_predict_binary():
    docs = ["good movie", "excellent film", "bad movie", "terrible film"]
    y = np.array([1, 1, 0, 0])
    cv = CountVectorizer()
    X = cv.fit_transform(docs)

    nb = NaiveBayes(alpha=1.0)
    nb.fit(X, y)
    pred = nb.predict(X)

    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})


def test_naive_bayes_predict_proba_valid():
    docs = ["good movie", "bad movie"]
    y = np.array([1, 0])
    cv = CountVectorizer()
    X = cv.fit_transform(docs)

    nb = NaiveBayes(alpha=1.0).fit(X, y)
    proba = nb.predict_proba(X)

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
