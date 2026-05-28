import numpy as np

from src.feature_extraction.count_vectorizer import CountVectorizer


def test_fit_transform_shape_and_vocab():
    docs = ["good movie", "bad movie", "good film"]
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_features=None)
    X = cv.fit_transform(docs)
    assert X.shape[0] == 3
    assert X.shape[1] == len(cv.vocabulary_)


def test_deterministic_vocabulary_on_ties():
    docs = ["a b", "b a"]
    cv1 = CountVectorizer(max_features=None)
    cv2 = CountVectorizer(max_features=None)
    cv1.fit(docs)
    cv2.fit(docs)
    assert cv1.vocabulary_ == cv2.vocabulary_


def test_oov_ignored_in_transform():
    train = ["good movie"]
    test = ["good unknown unknown"]
    cv = CountVectorizer()
    cv.fit(train)
    X = cv.transform(test)
    # only 'good' contributes
    assert X.nnz == 1


def test_repeated_tokens_counted():
    docs = ["good good good"]
    cv = CountVectorizer()
    X = cv.fit_transform(docs)
    idx = cv.vocabulary_["good"]
    assert X.toarray()[0, idx] == 3
