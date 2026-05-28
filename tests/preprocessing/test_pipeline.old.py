"""
Comprehensive test suite for the ML pipeline.

This module validates:
1. CountVectorizer - sparse matrix generation
2. TfidfTransformer - TF-IDF weighting
3. BaseModel interface
4. Metrics: accuracy, confusion_matrix
5. train_test_split with stratification
6. Decision Tree classifier
7. Naive Bayes classifier
8. End-to-end pipeline
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\Users\Admin\Src\Cod\ml-lab\a-sentiment')

from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer
from src.models.base import BaseModel
from src.models.decision_tree.decision_tree import DecisionTree
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.utils.helper import train_test_split
from src.preprocessing.text_processor import TextProcessor


def test_count_vectorizer():
    """Test CountVectorizer with toy dataset."""
    print("\n" + "="*60)
    print("TEST 1: CountVectorizer (Unigrams)")
    print("="*60)
    
    # Toy dataset
    docs = [
        "good movie",
        "bad movie",
        "good film",
        "bad film"
    ]
    
    # Fit and transform
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_features=None)
    X_train = cv.fit_transform(docs[:3])
    X_test = cv.transform([docs[3]])
    
    print(f"Vocabulary: {cv.vocabulary_}")
    print(f"Training matrix shape: {X_train.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    print(f"Sparsity (train): {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}")
    print(f"Training matrix (dense):\n{X_train.toarray()}")
    print(f"Test matrix (dense):\n{X_test.toarray()}")
    
    assert X_train.shape[0] == 3, "Training size mismatch"
    assert X_train.shape[1] == len(cv.vocabulary_), "Vocabulary size mismatch"
    assert X_train.nnz > 0, "Sparse matrix is empty"
    
    print("✓ CountVectorizer test PASSED")


def test_count_vectorizer_bigrams():
    """Test CountVectorizer with bigrams."""
    print("\n" + "="*60)
    print("TEST 2: CountVectorizer (Unigrams + Bigrams)")
    print("="*60)
    
    docs = [
        "not good not bad",
        "very good very good",
        "not bad at all"
    ]
    
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 2), max_features=10)
    X = cv.fit_transform(docs)
    
    print(f"Vocabulary size: {len(cv.vocabulary_)}")
    print(f"Vocabulary: {cv.vocabulary_}")
    print(f"Matrix shape: {X.shape}")
    print(f"Matrix (dense):\n{X.toarray()}")
    
    assert len(cv.vocabulary_) <= 10, "max_features not respected"
    assert X.nnz > 0, "Sparse matrix is empty"
    
    print("✓ CountVectorizer bigrams test PASSED")


def test_tfidf_transformer():
    """Test TfidfTransformer."""
    print("\n" + "="*60)
    print("TEST 3: TfidfTransformer")
    print("="*60)
    
    # Create toy count matrix
    docs = [
        "good good bad",
        "bad bad",
        "good bad"
    ]
    
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    X_count = cv.fit_transform(docs)
    
    print(f"Count matrix (dense):\n{X_count.toarray()}")
    print(f"Vocabulary: {cv.vocabulary_}")
    
    # Apply TF-IDF
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_count)
    
    print(f"IDF weights: {tfidf.idf_}")
    print(f"TF-IDF matrix (dense):\n{X_tfidf.toarray()}")
    
    # Check L2 normalization
    norms = np.sqrt(np.array(X_tfidf.power(2).sum(axis=1)).ravel())
    print(f"Row norms (should be ~1.0): {norms}")
    
    assert X_tfidf.nnz > 0, "TF-IDF matrix is empty"
    assert np.allclose(norms, 1.0), "L2 normalization failed"
    
    print("✓ TfidfTransformer test PASSED")


def test_train_test_split():
    """Test train_test_split with stratification."""
    print("\n" + "="*60)
    print("TEST 4: train_test_split (with stratification)")
    print("="*60)
    
    # Toy dataset with imbalanced classes
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=True)
    
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Check stratification
    train_ratio = np.bincount(y_train)[0] / len(y_train)
    test_ratio = np.bincount(y_test)[0] / len(y_test)
    
    print(f"Class 0 ratio - Train: {train_ratio:.2%}, Test: {test_ratio:.2%}")
    
    assert len(X_train) + len(X_test) == len(X), "Split size mismatch"
    assert abs(train_ratio - test_ratio) <= 0.25, "Stratification failed"
    
    print("✓ train_test_split test PASSED")


def test_metrics():
    """Test accuracy and confusion_matrix."""
    print("\n" + "="*60)
    print("TEST 5: Metrics (accuracy & confusion_matrix)")
    print("="*60)
    
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    
    # Test accuracy
    acc = accuracy(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Test confusion matrix
    cm, classes = confusion_matrix(y_true, y_pred)
    print(f"Classes: {classes}")
    print(f"Confusion Matrix:\n{cm}")
    
    assert 0 <= acc <= 1, "Invalid accuracy"
    assert cm.shape == (2, 2), "Wrong confusion matrix shape"
    
    print("✓ Metrics test PASSED")


def test_decision_tree():
    """Test Decision Tree classifier."""
    print("\n" + "="*60)
    print("TEST 6: Decision Tree Classifier")
    print("="*60)
    
    # Toy dataset
    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0]
    ])
    y_train = np.array([0, 0, 1, 1])
    
    X_test = np.array([[1.5, 2.5], [3.5, 1.5]])
    
    # Train
    dt = DecisionTree(min_samples_split=2, max_depth=3)
    dt.fit(X_train, y_train)
    
    # Predict
    y_pred = dt.predict(X_test)
    print(f"Predictions: {y_pred}")
    print(f"Prediction shape: {y_pred.shape}")
    
    # Check training accuracy
    y_pred_train = dt.predict(X_train)
    train_acc = accuracy(y_train, y_pred_train)
    print(f"Training accuracy: {train_acc:.4f}")
    
    assert y_pred.shape[0] == X_test.shape[0], "Prediction count mismatch"
    assert isinstance(y_pred, np.ndarray), "Prediction should be numpy array"
    
    print("✓ Decision Tree test PASSED")


def test_decision_tree_sparse():
    """Test Decision Tree with sparse matrices."""
    print("\n" + "="*60)
    print("TEST 7: Decision Tree with Sparse Matrices")
    print("="*60)
    
    # Create sparse matrix
    docs_train = ["good movie", "bad movie", "good film", "bad film"]
    docs_test = ["good bad"]
    
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    X_train = cv.fit_transform(docs_train)
    X_test = cv.transform(docs_test)
    y_train = np.array([1, 0, 1, 0])
    
    print(f"Training matrix shape: {X_train.shape}")
    print(f"Training matrix sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}")
    
    # Train on sparse matrix (should be converted to dense internally)
    dt = DecisionTree(max_depth=2)
    dt.fit(X_train, y_train)
    
    # Predict with sparse matrix
    y_pred = dt.predict(X_test)
    print(f"Predictions: {y_pred}")
    
    assert y_pred.shape[0] == X_test.shape[0], "Prediction count mismatch"
    
    print("✓ Decision Tree sparse test PASSED")


def test_naive_bayes():
    """Test Naive Bayes classifier."""
    print("\n" + "="*60)
    print("TEST 8: Naive Bayes Classifier")
    print("="*60)
    
    # Toy dataset
    X_train = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 1.0]
    ])
    y_train = np.array([0, 0, 1, 1])
    
    X_test = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
    
    # Train
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    
    # Predict
    y_pred = nb.predict(X_test)
    print(f"Predictions: {y_pred}")
    
    # Check training accuracy
    y_pred_train = nb.predict(X_train)
    train_acc = accuracy(y_train, y_pred_train)
    print(f"Training accuracy: {train_acc:.4f}")
    
    # Test probability prediction
    proba = nb.predict_proba(X_test)
    print(f"Prediction probabilities:\n{proba}")
    
    assert y_pred.shape[0] == X_test.shape[0], "Prediction count mismatch"
    assert proba.shape == (X_test.shape[0], 2), "Probability shape mismatch"
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    print("✓ Naive Bayes test PASSED")


def test_naive_bayes_sparse():
    """Test Naive Bayes with sparse matrices."""
    print("\n" + "="*60)
    print("TEST 9: Naive Bayes with Sparse Matrices (BoW)")
    print("="*60)
    
    # Create sparse BoW matrices
    docs_train = ["good good good", "bad bad", "good bad", "bad good good"]
    docs_test = ["good"]
    
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    X_train = cv.fit_transform(docs_train)
    X_test = cv.transform(docs_test)
    y_train = np.array([1, 0, 0, 1])
    
    print(f"Training matrix shape: {X_train.shape}")
    print(f"Training matrix (dense):\n{X_train.toarray()}")
    
    # Train (should convert sparse to dense internally)
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    
    # Predict
    y_pred = nb.predict(X_test)
    print(f"Predictions: {y_pred}")
    
    assert y_pred.shape[0] == X_test.shape[0], "Prediction count mismatch"
    
    print("✓ Naive Bayes sparse test PASSED")


def test_end_to_end_pipeline():
    """Test complete ML pipeline."""
    print("\n" + "="*60)
    print("TEST 10: End-to-End ML Pipeline")
    print("="*60)
    
    # Toy sentiment dataset
    docs = [
        "good movie very good",
        "bad movie very bad",
        "good film excellent",
        "bad film terrible",
        "amazing movie loved it",
        "worst movie hated it"
    ]
    labels = np.array([1, 0, 1, 0, 1, 0])
    
    print(f"Dataset: {len(docs)} documents, 2 classes")
    
    # Step 1: Train-test split
    docs_array = np.array(docs)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        docs_array, labels, test_size=0.33, random_state=42, stratify=True
    )
    
    print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Step 2: Vectorization
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_features=20)
    X_train = cv.fit_transform(X_train_text)
    X_test = cv.transform(X_test_text)
    
    print(f"Feature matrix shape - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 3: Optional TF-IDF transformation
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"After TF-IDF - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")
    
    # Step 4: Train with raw counts (Multinomial NB prefers counts)
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy(y_test, y_pred_nb)
    
    print(f"Naive Bayes accuracy: {acc_nb:.4f}")
    
    # Step 5: Train Decision Tree on TF-IDF
    dt = DecisionTree(max_depth=3)
    dt.fit(X_train_tfidf, y_train)
    y_pred_dt = dt.predict(X_test_tfidf)
    acc_dt = accuracy(y_test, y_pred_dt)
    
    print(f"Decision Tree accuracy: {acc_dt:.4f}")
    
    # Step 6: Confusion matrices
    cm_nb, classes_nb = confusion_matrix(y_test, y_pred_nb)
    cm_dt, classes_dt = confusion_matrix(y_test, y_pred_dt)
    
    print(f"Naive Bayes confusion matrix:\n{cm_nb}")
    print(f"Decision Tree confusion matrix:\n{cm_dt}")
    
    assert acc_nb >= 0, "Invalid NB accuracy"
    assert acc_dt >= 0, "Invalid DT accuracy"
    
    print("✓ End-to-end pipeline test PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE ML PIPELINE TEST SUITE")
    print("="*60)
    
    try:
        test_count_vectorizer()
        test_count_vectorizer_bigrams()
        test_tfidf_transformer()
        test_train_test_split()
        test_metrics()
        test_decision_tree()
        test_decision_tree_sparse()
        test_naive_bayes()
        test_naive_bayes_sparse()
        test_end_to_end_pipeline()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
