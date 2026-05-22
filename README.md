# Sentiment Analysis From Scratch

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-31%20passed-brightgreen.svg)](#testing)
[![Focus](https://img.shields.io/badge/Focus-NLP%20%7C%20Classical%20ML-informational.svg)](#overview)

**A research-oriented sentiment analysis repository built around from-scratch machine learning implementations, sparse NLP feature engineering, and reproducible experimentation.**

This project studies binary sentiment classification on IMDB and Stanford Sentiment Treebank style data using custom implementations of text preprocessing, Bag-of-Words, TF-IDF, Naive Bayes, and Decision Tree. The repository is designed both as an engineering artifact and as a learning environment for understanding how classical NLP pipelines behave beyond high-level library abstractions.

## Overview

The repository serves three aligned goals:

- **Educational goal**: expose the internal mechanics of classical NLP pipelines instead of hiding them behind black-box APIs.
- **Engineering goal**: organize preprocessing, vectorization, modeling, evaluation, experiments, tests, and reporting as separate modules.
- **Research goal**: provide a baseline framework for studying sparse text classification, especially the tradeoffs of Naive Bayes and Decision Tree on sentiment data.

The core emphasis of the report and repository is on **Multinomial Naive Bayes** and **Decision Tree**. Additional models such as **KNN**, **Logistic Regression**, **SVM**, **Perceptron**, **Random Forest**, and a simple **Neural Network** are present as secondary experimental implementations.

## Key Features

- **From-scratch ML implementations** for Naive Bayes, Decision Tree, KNN, SVM, Logistic Regression, Perceptron, Random Forest, and a feed-forward Neural Network.
- **Custom NLP preprocessing** with HTML cleaning, lowercasing, URL/email removal, tokenization, stopword filtering, and optional numeric removal.
- **Sparse vectorization stack** with a custom `CountVectorizer` and `TfidfTransformer`.
- **Evaluation utilities** for accuracy, confusion matrix computation, and visualization.
- **Experiment entrypoints** through `main.py` and a YAML-configured baseline runner under `experiments/`.
- **Notebook workflow** for exploratory analysis and teaching-oriented experimentation in `notebook/exploration.ipynb`.
- **Testing support** across preprocessing, feature extraction, models, evaluation, and utilities.
- **Academic reporting workflow** centered on `report/BaoCaoBaiTap.md` and the planned figure workflow in `report/images/`.
- **Reproducibility primitives** including deterministic train/test splitting and structured output directories.

## Why This Project Matters

Classical sentiment analysis remains valuable because sparse text classification still teaches the core engineering pressures of NLP:

- **Understanding internals matters**: Laplace smoothing, entropy, information gain, TF-IDF normalization, and stratified splitting become much clearer when implemented directly.
- **Sparse NLP is not trivial**: large vocabularies, document sparsity, and feature dimensionality shape both performance and memory use.
- **Black-box abstraction hides tradeoffs**: this repository makes those tradeoffs visible, especially where current implementations still convert sparse matrices to dense arrays inside some models.
- **Educational value is high**: the project is suitable for coursework, debugging practice, algorithm study, and baseline experimentation before moving to embedding-based or transformer-based systems.
- **Engineering decisions remain visible**: modular code, explicit preprocessing, configurable experiments, and tests make the repository easier to audit and extend.

## Architecture Overview

The repository is organized around a standard NLP pipeline:

**Raw text -> preprocessing -> vectorization -> model training -> evaluation -> reporting**

### Component Roles

- **Preprocessing**: normalize raw text and enforce consistent token-level inputs.
- **Feature extraction**: convert cleaned text into sparse document-term representations.
- **Models**: train classical ML classifiers implemented in NumPy/SciPy style code.
- **Evaluation**: compute metrics and generate plots for error analysis.
- **Experiments**: run scripted baselines and write reproducible artifacts.
- **Reports and notebooks**: support exploratory work and academic writeups.

### Repository Structure

```text
ml-models-sentiment/
├── main.py
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/
│   │   ├── imdb/
│   │   └── sst/
│   └── processed/
├── experiments/
│   ├── config.yaml
│   ├── config.old.yaml
│   ├── run_baseline.py
│   └── run_baseline.old.py
├── notebook/
│   ├── exploration.ipynb
│   ├── exploration.main.ipynb
│   └── simple-mnist-nn-from-scratch-numpy-no-tf-keras.ipynb
├── report/
│   ├── BaoCaoBaiTap.md
│   ├── IMAGES_GUIDE.md
│   └── images/
├── src/
│   ├── evaluation/
│   ├── feature_extraction/
│   ├── models/
│   │   ├── decision_tree/
│   │   ├── knn/
│   │   ├── naive_bayes/
│   │   ├── neural_network/
│   │   ├── pca/
│   │   ├── perceptron/
│   │   ├── regression/
│   │   └── svms/
│   ├── preprocessing/
│   └── utils/
└── tests/
    ├── evaluation/
    ├── feature_extraction/
    ├── models/
    ├── preprocessing/
    └── utils/
```

Some folders also keep `*.old.*` snapshots as reference material. They are preserved in the repository but excluded from the active test collection.

## Datasets

The repository works with two sentiment datasets already present under `data/raw/`:

- **IMDB**: movie review sentiment data stored in `data/raw/imdb/IMDB_dataset.csv`.
- **SST-style dataset**: sentence-level sentiment data stored in `data/raw/sst/train.csv`, with companion `test.csv` and `sample_submission.csv`.

### Task Framing

- Primary framing: **binary sentiment classification**
- Input type: raw review or sentence text
- Output type: binary polarity labels

### Dataset Setup

There is **no automated dataset download script** in the repository. The current workflow assumes the CSV files already exist under `data/raw/`, which is the case for this codebase snapshot.

### Preprocessing Goals

The preprocessing layer is designed to:

- remove HTML and markup artifacts
- reduce casing noise
- strip URLs and emails
- optionally remove numbers
- tokenize consistently
- optionally remove stopwords

These choices are suitable for classical sparse models, although they also surface the usual tradeoff between vocabulary compression and loss of nuanced sentiment cues.

## ML Pipeline

The implemented workflow is:

**Raw Text -> Cleaning -> Tokenization -> Stopword Removal -> Vectorization -> Model Training -> Evaluation**

### Count Vectorizer

`src/feature_extraction/count_vectorizer.py` builds a vocabulary from training data and returns a sparse CSR matrix of token counts. It supports:

- lowercasing
- configurable `ngram_range`
- configurable `max_features`

### TF-IDF

`src/feature_extraction/tfidf_vectorizer.py` transforms sparse count matrices into TF-IDF representations with:

- smoothed IDF
- column-wise weighting
- row-wise L2 normalization

### Sparse Feature Space Challenges

This repository makes the classical NLP tradeoff explicit:

- vectorizers produce **memory-efficient sparse matrices**
- some downstream custom models still call `toarray()`
- the code is therefore educationally transparent, but not yet fully optimized for very large vocabularies or corpora

That tension is central to the project and is discussed directly in the report.

## Model Implementations

### Primary Models

#### Naive Bayes

`src/models/naive_bayes/naive_bayes.py` implements a **Multinomial Naive Bayes** classifier from scratch using:

- class prior estimation
- per-class feature likelihoods
- **Laplace smoothing** via `alpha`
- **log-space posterior computation** for numerical stability

This is the main probabilistic baseline in the repository and one of the two central models in the report.

#### Decision Tree

`src/models/decision_tree/decision_tree.py` implements a classification tree using:

- **entropy** as the impurity function
- **information gain** for split selection
- `max_depth`
- `min_samples_split`

This is the main rule-based baseline in the repository and the second central model in the report.

### Secondary Implementations

The codebase also includes additional experimental models:

- `src/models/knn/knn.py`
- `src/models/regression/logistic.py`
- `src/models/regression/linear.py`
- `src/models/svms/svm.py`
- `src/models/perceptron/perceptron.py`
- `src/models/decision_tree/random_forest.py`
- `src/models/neural_network/neural_network.py`
- `src/models/pca/pca.py`

These broaden the repository technically, but they are not the primary focus of the report.

## Experiments

The repository contains two practical experiment entrypoints:

- **`main.py`**: a command-line baseline runner for IMDB or SST-style CSV data
- **`experiments/run_baseline.py`**: a YAML-configured experiment script paired with `experiments/config.yaml`

### Experiment Workflow

The code supports the following baseline workflow:

1. load raw CSV data
2. infer or override text/label columns
3. preprocess text with `TextProcessor`
4. split train/test with stratification
5. build Count or TF-IDF features
6. train one or more models
7. compute metrics
8. save artifacts under `results/`

### Metrics

The evaluation stack explicitly works with:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The CLI baseline in `main.py` computes macro precision/recall/F1 and writes structured outputs such as:

- `metrics.json`
- `predictions.csv`
- optional figure exports

## Visualization

`src/evaluation/visualization.py` provides plotting helpers for:

- confusion matrices
- model accuracy comparison
- per-class metric charts
- class distribution views
- prediction distribution views

### Report Images

The repository includes a reporting workflow in `report/IMAGES_GUIDE.md` that standardizes:

- figure naming
- figure storage under `report/images/`
- markdown integration into the report

At the moment, `report/images/` is present but empty, which is consistent with an in-progress reporting workflow rather than a fully curated final artifact set.

### Notebook Visualizations

The notebook adds a more exploratory layer with:

- class distribution plots
- token-length histograms
- sparsity views
- confusion matrix visualizations
- feature inspection for teaching and debugging

## Notebook

The main notebook is `notebook/exploration.ipynb`.

Its role is not only to show outputs, but to document reasoning:

- dataset inspection for IMDB and SST-style data
- preprocessing examples before and after cleaning
- feature extraction diagnostics
- model workflow demonstrations
- guided sections for manual practice on Naive Bayes and Decision Tree

This makes the notebook useful for:

- exploratory analysis
- debugging preprocessing
- classroom demonstration
- research-style experimentation

## Testing

The repository includes a meaningful `pytest` suite and currently passes:

- **31 tests passed**

### Test Coverage Areas

- **Preprocessing tests**: text cleaning and CSV preprocessing pipeline behavior
- **Feature extraction tests**: CountVectorizer, TF-IDF, and vocabulary logic
- **Model tests**: Naive Bayes, Decision Tree, KNN, SVM, regression, Random Forest, and Neural Network smoke coverage
- **Evaluation tests**: accuracy, confusion matrix, and visualization helpers
- **Utility tests**: stratified `train_test_split`

### Test Layout

```bash
tests/
├── evaluation/
├── feature_extraction/
├── models/
├── preprocessing/
└── utils/
```

This test organization materially improves the engineering quality of the repository: models are not treated as notebook-only experiments, but as modules with executable expectations.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ml-models-sentiment
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main baseline

```bash
python main.py --dataset sst --limit 300 --models naive_bayes decision_tree
```

### 5. Run tests

```bash
pytest -q
```

### 6. Launch the notebook

```bash
jupyter notebook notebook/exploration.ipynb
```

## Results

The repository does not yet present a single canonical benchmark table for the full datasets. However, executable baseline outputs do exist.

### Example Output

The following numbers are from a **small smoke run** of:

```bash
python main.py --dataset sst --limit 300 --models naive_bayes decision_tree --no-plots
```

| Run Type | Dataset Slice | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---|---:|---:|---:|---:|
| Example smoke run | SST, first 300 rows | Naive Bayes | 0.6066 | 0.5924 | 0.5731 | 0.5643 |
| Example smoke run | SST, first 300 rows | Decision Tree | 0.6066 | 0.6144 | 0.5533 | 0.5107 |

These numbers should be interpreted as **sanity-check outputs**, not as final benchmark claims.

## Report

The academic report lives in:

- `report/BaoCaoBaiTap.md`

Its current emphasis is aligned with the repository's primary research focus:

- Naive Bayes
- Decision Tree
- sparse NLP feature engineering
- engineering tradeoffs in from-scratch ML

### Reporting Workflow

- `report/BaoCaoBaiTap.md` holds the narrative report
- `report/IMAGES_GUIDE.md` defines the figure-generation convention
- `report/images/` is the intended destination for report-ready visual assets

This separation between code, experiment artifacts, and academic writeup is a good sign of maintainability: the repository is not only trying to run models, but also to support documentation and communication of results.

## Engineering Quality

Several qualities make this repository stronger than a minimal classroom script:

- **Modularity**: preprocessing, vectorization, models, evaluation, and utilities are separated into dedicated modules.
- **Reproducibility**: deterministic train/test split settings and artifact output directories are built into the workflow.
- **Separation of concerns**: notebooks, scripts, source modules, tests, and report materials are not collapsed into a single file.
- **Maintainability**: tests exist across multiple subsystems, and backup/reference files are preserved without being part of active test collection.
- **Extensibility**: the repository already accommodates multiple model families and both script-based and notebook-based experimentation.

This structure reads well for recruiters and maintainers because it shows that the project is not only about model accuracy, but about how machine learning code is organized, tested, and communicated.

## Future Work

Several next steps follow naturally from the current codebase:

- keep sparse representations all the way through model training where possible
- strengthen the experiment runner and benchmark reporting workflow
- add more formal hyperparameter sweeps
- integrate figure export directly into the report pipeline
- extend ensemble experiments such as **Random Forest**
- explore dense representations and embeddings
- add modern baselines such as **Transformer** or **BERT**-based sentiment classifiers
- improve scaling and optimization for larger corpora

## License

This repository is released under the **MIT License**. See [LICENSE](LICENSE).

## Repository Status

At the current snapshot, the repository is best understood as:

- a **working from-scratch NLP/ML project**
- a **serious educational baseline**
- a **research-style sentiment analysis codebase**
- an **engineering portfolio artifact in active refinement**

It is already strong as a study repository for classical NLP pipelines, and it becomes even more valuable because its limitations are visible rather than hidden.
