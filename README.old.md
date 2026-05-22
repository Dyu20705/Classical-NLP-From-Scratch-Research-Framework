# Sentiment Analysis ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

A **custom-implemented** sentiment analysis machine learning pipeline featuring hand-crafted feature extraction, text preprocessing, and classification models built from scratch—without reliance on scikit-learn for core algorithms.

## 🎯 Project Overview

This project demonstrates:
- **Custom feature extraction**: Bag-of-Words (CountVectorizer) and TF-IDF transformations using sparse matrices
- **From-scratch classifiers**: Decision Tree and Multinomial Naive Bayes implementations
- **Memory efficiency**: Sparse matrix operations for high-dimensional text data
- **Reproducibility**: Stratified train/test splits with fixed random seed (42)
- **Production-ready evaluation**: Comprehensive metrics including accuracy, confusion matrices, and visualizations

### Key Differentiators

✅ **No sklearn for core algorithms** — All feature extraction and base classifiers implemented from first principles  
✅ **Sparse matrix optimization** — Efficient storage and computation for text features  
✅ **Numerical stability** — Log-space probability calculations in Naive Bayes to prevent underflow  
✅ **Data integrity** — Stratified splitting ensures class distribution preservation (no data leakage)  
✅ **Professional structure** — Modular design with concise documentation and test suite  

## 📋 Quick Links

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Experimental Results](#-experimental-results)
- [Testing](#-testing)

## 🚀 Quick Start

### Run the Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python main.py
```

**Expected Output:** Complete sentiment analysis pipeline with model training, evaluation, and visualizations.

## 📦 Installation

### Requirements

- **Python 3.10+**
- **NumPy** ≥ 1.21.0
- **SciPy** ≥ 1.7.0
- **Pandas** ≥ 1.3.0
- **Matplotlib** ≥ 3.4.0
- **Seaborn** ≥ 0.11.0

### Setup

```bash
# Clone repository
git clone <repository-url>
cd a-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Pipeline Example

```python
# New: run experiments runner
# python experiments/run_baseline.py

# Use data/processed/*_clean.csv produced by src.preprocessing.pipeline
from src import (
    CountVectorizer, TfidfTransformer,
    NaiveBayes, DecisionTree,
    train_test_split, accuracy, confusion_matrix
)
from src import TextProcessor

# 1. Load and preprocess data
processor = TextProcessor(remove_stopwords=False)
texts = np.array(["Great movie!", "Terrible film", ...])
labels = np.array([1, 0, ...])

processed = [' '.join(processor.process(text)) for text in texts]

# 2. Split data (stratified)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(processed), labels,
    test_size=0.2, random_state=42, stratify=True
)

# 3. Extract features
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 4. Train model
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)

# 5. Evaluate
y_pred = nb.predict(X_test)
acc = accuracy(y_test, y_pred)
cm, classes = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

### Feature Extraction

#### Bag-of-Words

```python
from src import CountVectorizer

cv = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 1),  # Unigrams only
    max_features=1000
)
X = cv.fit_transform(texts)  # Sparse matrix (n_samples, 1000)
```

#### TF-IDF

```python
from src import TfidfTransformer

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X)
```

### Classification Models

#### Naive Bayes

```python
from src import NaiveBayes

nb = NaiveBayes(alpha=1.0)  # Laplace smoothing
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)  # Probability estimates
```

#### Decision Tree

```python
from src import DecisionTree

dt = DecisionTree(max_depth=4, min_samples_split=2)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
```

### Evaluation & Visualization

```python
from src.evaluation import accuracy, confusion_matrix
from src.evaluation.visualization import MetricsVisualizer

# Metrics
acc = accuracy(y_test, y_pred)
cm, classes = confusion_matrix(y_test, y_pred)

# Visualizations
MetricsVisualizer.plot_confusion_matrix(cm, classes)
MetricsVisualizer.plot_roc_style_metrics(y_test, y_pred)

import matplotlib.pyplot as plt
plt.show()
```

## 📁 Project Structure

```
a-sentiment/
├── main.py                              # Full pipeline demo
├── requirements.txt                     # Dependencies
├── README.md                            # This file
├── .gitignore                           # Git ignore rules
│
├── src/                                 # Main package
│   ├── __init__.py                      # Module exports
│   ├── feature_extraction/              # Feature extraction
│   │   ├── count_vectorizer.py          # Bag-of-words
│   │   ├── tfidf_vectorizer.py          # TF-IDF transformer
│   │   └── vocabulary.py                # Word↔index mappings
│   │
│   ├── models/                          # Classification models
│   │   ├── base.py                      # Abstract interface
│   │   ├── decision_tree/
│   │   │   ├── decision_tree.py         # Decision tree classifier
│   │   │   └── decision_tree_old.py     # Backup
│   │   └── naive_bayes/
│   │       ├── naive_bayes.py           # Naive Bayes classifier
│   │       └── naive_bayes_old.py       # Backup
│   │
│   ├── preprocessing/
│   │   └── text_processor.py            # Text cleaning & tokenization
│   │
│   ├── evaluation/
│   │   ├── accuracy.py                  # Accuracy metric
│   │   ├── confusion_matrix.py          # Confusion matrix
│   │   └── visualization.py             # Matplotlib utilities
│   │
│   └── utils/
│       └── helper.py                    # train_test_split, etc.
│
├── tests/
│   └── test_pipeline.py                 # Test suite
│
├── data/
│   ├── raw/                             # Raw datasets
│   └── processed/                       # Processed data
│
├── experiments/
│   ├── config.yaml                      # Experiment config
│   └── run_baseline.py                  # Baseline runner
│
└── notebook/
    └── exploration.ipynb                # Data exploration
```

## 🔌 API Reference

### CountVectorizer

```python
cv = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    max_features=None
)
X = cv.fit_transform(texts)              # Sparse matrix
X_test = cv.transform(texts_test)        # Transform new data
vocab_size = len(cv.vocabulary_)         # Vocabulary size
```

### TfidfTransformer

```python
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)
X_test_tfidf = tfidf.transform(X_test_counts)
```

### NaiveBayes

```python
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)
```

### DecisionTree

```python
dt = DecisionTree(max_depth=4, min_samples_split=2)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
```

### TextProcessor

```python
processor = TextProcessor(
    remove_stopwords=False,
    remove_numbers=True
)
clean_text = processor.clean(text)
tokens = processor.tokenize(text)
processed = processor.process(text)
```

### Utilities

```python
from src.utils import train_test_split
from src.evaluation import accuracy, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=True
)
acc = accuracy(y_test, y_pred)
cm, classes = confusion_matrix(y_test, y_pred)
```

## 📊 Experimental Results

| Model | Features | Accuracy |
|-------|----------|----------|
| Naive Bayes | Raw Counts | 1.0000 |
| Decision Tree | Raw Counts | 1.0000 |
| Decision Tree | TF-IDF | 1.0000 |

**Dataset**: 12 sentiment-labeled reviews (toy dataset)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Experiment Pipeline

```bash
# Baseline run from YAML config
python experiments/run_baseline.py
```

Artifacts are saved under `results/experiments/` in organized subfolders (`metrics`, `logs`, `predictions`, `figures`).

## 🔬 Methodology

### Feature Extraction
1. Text preprocessing (HTML removal, lowercasing)
2. Tokenization (whitespace-based)
3. Vocabulary building (word→index mapping)
4. Bag-of-Words (sparse count matrix)
5. TF-IDF weighting (optional)

### Classification
- **Naive Bayes**: Multinomial with Laplace smoothing
- **Decision Tree**: Information gain-based splitting

### Data Handling
- Stratified train/test split (80/20)
- Sparse matrix format (CSR)
- Fixed random seed (42) for reproducibility

## 📝 Key Features

✅ **No sklearn for core algorithms** - Pure NumPy/SciPy  
✅ **Sparse matrices** - Memory efficient text representation  
✅ **Log-space computation** - Numerical stability  
✅ **Stratified splitting** - No data leakage  
✅ **Comprehensive tests** - 10+ test cases  
✅ **Visualization tools** - matplotlib/seaborn integration  
✅ **Professional documentation** - Docstrings and examples  

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 📚 Documentation

- `docs/IMPLEMENTATION_SUMMARY.md` - Current implementation snapshot
- `docs/ENGINEERING_REVIEW.md` - Current engineering notes
- `docs/VERIFICATION_CHECKLIST.md` - Current constraints and validation
- `report/BaoCaoBaiTap.md` - Vietnamese academic report

---

**Status**: Current snapshot  
**Last Updated**: 2026-05-22
