import numpy as np

from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer


def test_tfidf_shape_matches_input():
    docs = ["good movie", "bad movie", "good film"]
    cv = CountVectorizer()
    X = cv.fit_transform(docs)
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)
    assert X_tfidf.shape == X.shape


def test_tfidf_row_l2_norm_close_to_one():
    docs = ["good good bad", "bad bad", "good bad"]
    cv = CountVectorizer()
    X = cv.fit_transform(docs)
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)
    norms = np.sqrt(np.array(X_tfidf.power(2).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)


def test_transform_before_fit_raises():
    tfidf = TfidfTransformer()
    cv = CountVectorizer()
    X = cv.fit_transform(["a b"])
    try:
        tfidf.transform(X)
    except ValueError:
        return
    assert False, "Expected ValueError when transform called before fit"
