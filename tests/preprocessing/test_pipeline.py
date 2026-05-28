import numpy as np

from src.preprocessing.text_processor import TextProcessor
from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.models.decision_tree.decision_tree import DecisionTree
from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.utils.helper import train_test_split


def test_end_to_end_pipeline_smoke():
    docs = np.array([
        "good movie great",
        "excellent film good",
        "bad movie terrible",
        "awful film bad",
        "nice and enjoyable",
        "boring and poor",
    ])
    y = np.array([1, 1, 0, 0, 1, 0])

    p = TextProcessor(remove_stopwords=True)
    cleaned = np.array([' '.join(p.process(t)) for t in docs])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned, y, test_size=0.33, random_state=42, stratify=True
    )

    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_features=50)
    X_train = cv.fit_transform(X_train_text)
    X_test = cv.transform(X_test_text)

    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    nb = NaiveBayes(alpha=1.0).fit(X_train, y_train)
    dt = DecisionTree(max_depth=4, min_samples_split=2).fit(X_train_tfidf, y_train)

    y_pred_nb = nb.predict(X_test)
    y_pred_dt = dt.predict(X_test_tfidf)

    acc_nb = accuracy(y_test, y_pred_nb)
    acc_dt = accuracy(y_test, y_pred_dt)

    cm_nb, classes_nb = confusion_matrix(y_test, y_pred_nb)
    cm_dt, classes_dt = confusion_matrix(y_test, y_pred_dt)

    assert 0.0 <= acc_nb <= 1.0
    assert 0.0 <= acc_dt <= 1.0
    assert cm_nb.sum() == len(y_test)
    assert cm_dt.sum() == len(y_test)
    assert len(classes_nb) >= 1
    assert len(classes_dt) >= 1
