# Classical NLP Sentiment Research Framework (From Scratch)

[![CI](https://github.com/Dyu20705/processing-sentiment-with-ml-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Dyu20705/processing-sentiment-with-ml-model/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Abstract

This repository investigates binary sentiment classification through a classical NLP lens, with implementations written from scratch and organized as a reproducible experiment framework. The project focuses on sparse text representations, probabilistic baselines, and model behavior under high-dimensional feature spaces. While modern deep models dominate many benchmarks, classical pipelines remain scientifically useful for studying inductive bias, data efficiency, interpretability, and systems-level tradeoffs. The framework is designed to support disciplined experimentation rather than ad-hoc model collection.

## Design Philosophy

The repository is built around four principles:

1. Scientific framing before implementation volume.
2. Reproducibility over one-off results.
3. Sparse-first engineering for text data.
4. Honest reporting of limitations and failure modes.

## Research Motivation

Classical NLP remains valuable for both research training and engineering practice:

- Multinomial Naive Bayes is often a strong baseline on sparse text features despite strong independence assumptions.
- Sparse representations expose computational tradeoffs more transparently than many end-to-end black-box workflows.
- From-scratch implementations force explicit reasoning about numerical stability, feature statistics, and evaluation boundaries.
- Controlled classical pipelines are useful for methodological ablations (preprocessing, vocabulary size, n-gram range, weighting schemes).

## Research Questions

1. Why does Multinomial Naive Bayes perform strongly on sparse sentiment features?
2. How does TF-IDF weighting affect generalization relative to raw counts?
3. Why do decision trees often degrade in high-dimensional sparse text spaces?
4. What are the computational and memory tradeoffs between sparse and dense operations?
5. How sensitive are models to preprocessing choices (stopwords, numbers, n-grams, text normalization)?

## Repository Features

- From-scratch implementations of core NLP and ML components.
- Sparse matrix support in feature extraction and baseline workflows.
- Config-driven experiment execution.
- Modular preprocessing and feature extraction layers.
- Stratified evaluation utilities and metrics.
- Artifact-oriented outputs for downstream analysis.
- Unit tests and CI validation.

## System Architecture

The framework is layered to keep data flow explicit and composable:

- Preprocessing layer: text normalization, tokenization, stopword handling.
- Feature extraction layer: count-based and TF-IDF sparse vectors.
- Model layer: baseline and experimental estimators.
- Evaluation layer: metrics and diagnostic visualizations.
- Experiment layer: CLI and YAML-driven orchestration.
- Utility layer: split helpers, notebook setup, and search tools.

```text
ml-models-sentiment/
|- main.py
|- experiments/
|  |- config.yaml
|  |- run_baseline.py
|- src/
|  |- preprocessing/
|  |  |- text_processor.py
|  |  |- pipeline.py
|  |- feature_extraction/
|  |  |- count_vectorizer.py
|  |  |- tfidf_vectorizer.py
|  |  |- vocabulary.py
|  |- models/
|  |  |- base.py
|  |  |- naive_bayes/naive_bayes.py
|  |  |- decision_tree/decision_tree.py
|  |  |- decision_tree/random_forest.py
|  |  |- knn/knn.py
|  |  |- neural_network/neural_network.py
|  |  |- perceptron/perceptron.py
|  |  |- regression/
|  |  |- svms/svm.py
|  |  |- pca/pca.py
|  |- evaluation/
|  |  |- accuracy.py
|  |  |- confusion_matrix.py
|  |  |- visualization.py
|  |- utils/
|     |- helper.py
|     |- grid_search.py
|     |- notebook_setup.py
|- tests/
|- test_src/
|- data/raw/
|- results/
```

## Experimental Methodology

### Datasets

- IMDB: `data/raw/imdb/IMDB_dataset.csv`
- SST train: `data/raw/sst/train.csv`
- SST test: `data/raw/sst/test.csv`

### Data Protocol

- Automatic text/label column inference with optional explicit override.
- Label normalization to a stable numeric representation.
- Train/test split with stratification and fixed seed.
- Feature fitting only on train partitions to avoid leakage.

### Metrics

- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1
- Macro-average summary
- Confusion matrix

### Vectorization Approaches

- CountVectorizer (baseline sparse representation)
- TF-IDF transformer (ablation and weighting analysis)

### Reproducibility Strategy

- Default deterministic seed (`random_state=42`).
- Configurable experiment parameters (`experiments/config.yaml`).
- Persistent run artifacts for metrics and predictions.
- CI execution for tests and baseline experiment path.

## Implemented Models

### Baseline Models

1. Multinomial Naive Bayes (`src/models/naive_bayes/naive_bayes.py`)
Reason: a principled probabilistic baseline that performs competitively on sparse text due to additive token evidence and robust smoothing behavior.

2. Decision Tree (`src/models/decision_tree/decision_tree.py`)
Reason: a contrastive non-linear baseline for studying fragmentation and overfitting behavior in high-dimensional sparse NLP spaces.

### Experimental Models

- Random Forest, KNN, SVM, Logistic Regression, Linear Regression, Perceptron, Neural Network, PCA.
Reason: controlled extensions for comparative experiments, algorithmic study, and educational systems analysis. These are not treated as canonical production baselines in the current sentiment pipeline.

## Feature Extraction

- CountVectorizer: produces sparse bag-of-words matrices, supports n-gram ranges and feature caps, and serves as the baseline representation.
- TF-IDF Transformer: applies inverse document frequency reweighting with L2 normalization to evaluate frequency-weighted generalization.
- Vocabulary Management: keeps explicit token-index mappings with unknown-token handling for controlled feature-space growth.

Rationale for sparse representation:

- Text feature spaces are high-dimensional and mostly zero.
- Sparse operations reduce memory usage and improve tractability.
- Sparse pipelines expose systems-level bottlenecks that are hidden in dense abstractions.

## Reproducibility

The project follows a deterministic experiment philosophy:

- Fixed random seeds in split and model-related procedures.
- YAML-driven experiment parameters.
- CI workflow (`.github/workflows/ci.yml`) for test and baseline execution.
- Artifact saving in `results/cli` and `results/experiments`:

  - `metrics.json`
  - `classification_report.json`
  - `predictions.csv`
  - `label_lookup.json`
  - `metadata.json`
  - `git_commit.txt`
  - `environment.txt`

## Evaluation Strategy

Evaluation goes beyond scalar accuracy:

- Class-wise precision, recall, and F1 for imbalance awareness.
- Confusion-matrix inspection for error topology.
- Macro metrics for class-agnostic comparison.
- Failure analysis as a first-class activity (negation, rare words, long dependencies, label ambiguity).

## Key Findings (Current Snapshot)

- Naive Bayes is a stable sparse baseline and generally stronger than decision trees under the same lightweight setup.
- Decision trees are sensitive to sparse, high-dimensional splits and require careful regularization.
- The framework is suitable for ablation studies and reproducibility checks, but current reported values are baseline-level, not claim-level research conclusions.

## Example Experimental Results

The table below shows representative baseline-style results from constrained runs. These values are included to illustrate reporting format and expected magnitude, not to claim state-of-the-art performance.

### Model and Vectorizer Comparison

| Dataset Slice | Model | Vectorizer | Accuracy | Macro F1 |
| --- | --- | --- | ---: | ---: |
| SST (300 rows, smoke run) | Naive Bayes | Count | 0.590 | 0.579 |
| SST (300 rows, smoke run) | Decision Tree | Count | 0.574 | 0.488 |
| SST (small baseline run) | Naive Bayes | TF-IDF | 0.60-0.64 | 0.59-0.63 |
| SST (small baseline run) | Decision Tree | TF-IDF | 0.55-0.60 | 0.50-0.57 |

### Training Cost Observations

| Model | Vectorizer | Relative Train Cost | Relative Inference Cost | Sparse Compatibility |
| --- | --- | --- | --- | --- |
| Naive Bayes | Count / TF-IDF | Low | Low | Strong |
| Decision Tree | Count / TF-IDF | Medium to High | Low to Medium | Partial |
| Neural Network (experimental) | Dense input path | Medium to High | Medium | Limited in current implementation |

## Engineering Notes

- Sparse-first design is used in feature extraction and evaluation paths.
- Some model implementations still densify sparse inputs internally, which impacts memory behavior on larger vocabularies.
- Modular decomposition supports controlled substitutions across preprocessing, vectorization, and model layers.
- Grid search utilities are included for baseline hyperparameter sweeps, but exhaustive tuning is intentionally limited in the current scope.

## Current Limitations

- Some experimental models are not fully sparse-aware.
- `main.py` still couples orchestration, reporting, and artifact writing.
- Hyperparameter search depth is modest.
- Experiment metadata tracking (commit hash, environment snapshot, dataset checksums) is not yet fully automated.
- Current emphasis is educational/research interpretability over production optimization.

## Future Work

- Sparse-aware tree training and pruning strategies for text feature spaces.
- Structured experiment tracking with immutable run metadata.
- Advanced tokenization and linguistic preprocessing variants.
- Calibrated probability analysis and confidence reliability diagnostics.
- Comparative baseline against compact transformer classifiers under controlled compute budgets.
- Runtime and memory profiling suite for sparse vs dense paths.
- Distributed or chunked preprocessing for larger corpora.

## Repository Structure

```text
.
|- main.py
|- requirements.txt
|- experiments/
|- src/
|- tests/
|- test_src/
|- data/raw/
|- results/
|- notebook/
|- docs/
|- .github/workflows/ci.yml
```

## Quickstart

### Installation

```bash
git clone https://github.com/Dyu20705/processing-sentiment-with-ml-model.git
cd ml-models-sentiment
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Example Commands

Run baseline CLI:

```bash
python main.py --dataset sst --limit 300
```

Run config-driven experiment:

```bash
python experiments/run_baseline.py
```

Aggregate benchmark across experiment runs:

```bash
python scripts/aggregate_results.py
```

Run controlled ablations (stopwords, n-gram, max_features):

```bash
python scripts/run_ablations.py --dataset sst --limit 2000
```

Override dataset path and columns:

```bash
python main.py --dataset-path data/raw/imdb/IMDB_dataset.csv --text-col review --label-col sentiment --limit 5000
```

## Testing

Run all tests:

```bash
python -m pytest -q
```

CI validates:

- test suite execution
- baseline experiment run
- artifact upload for inspection

## License

This repository is released under the MIT License. See `LICENSE`.
