"""Run legacy sentiment experiments across all classification models in src/."""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
    _has_yaml = True
except Exception:
    _has_yaml = False

from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.evaluation.visualization import MetricsVisualizer
from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer
from src.models.decision_tree.decision_tree import DecisionTree
from src.models.decision_tree.random_forest import RandomForest
from src.models.knn.knn import KNN
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.models.neural_network.neural_network import NeuralNetworkClassifier
from src.models.perceptron.perceptron import Perceptron
from src.models.regression.linear import LinearRegression
from src.models.regression.logistic import LogisticRegression
from src.models.svms.svm import SVM
from src.preprocessing.pipeline import process_csv_to_clean, validate_labels
from src.utils.helper import train_test_split


def _minimal_yaml_parse(path):
    config = {}
    stack = [config]
    indent_stack = [0]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.rstrip('\n')
            if not raw.strip() or raw.strip().startswith('#'):
                continue
            indent = len(raw) - len(raw.lstrip(' '))
            keyval = raw.strip()
            if ':' not in keyval:
                continue
            key, val = keyval.split(':', 1)
            key = key.strip()
            val = val.strip()
            if val == '':
                node = {}
                while indent_stack and indent <= indent_stack[-1]:
                    stack.pop()
                    indent_stack.pop()
                stack[-1][key] = node
                stack.append(node)
                indent_stack.append(indent)
                continue
            parsed = val
            if parsed.startswith('[') and parsed.endswith(']'):
                parsed = [item.strip().strip('"\'') for item in parsed[1:-1].split(',') if item.strip()]
            elif parsed.lower() in ('true', 'false'):
                parsed = parsed.lower() == 'true'
            else:
                try:
                    parsed = float(parsed) if '.' in parsed else int(parsed)
                except Exception:
                    parsed = parsed.strip('"\'')
            stack[-1][key] = parsed
    return config


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if _has_yaml:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return _minimal_yaml_parse(path)


def ensure_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_path(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _detect_columns(df, text_col, label_col):
    cols = set(df.columns)
    if text_col not in cols:
        for candidate in ('sentence', 'text', 'review'):
            if candidate in cols:
                text_col = candidate
                break
    if label_col not in cols:
        for candidate in ('target', 'label', 'sentiment'):
            if candidate in cols:
                label_col = candidate
                break
    if text_col not in cols or label_col not in cols:
        raise ValueError(f"Could not detect text/label columns. Available columns: {list(df.columns)}")
    return text_col, label_col


def _prepare_features(X_train_text, X_test_text, vec_cfg):
    vectorizer = CountVectorizer(
        lowercase=vec_cfg.get('lowercase', True),
        ngram_range=tuple(vec_cfg.get('ngram_range', [1, 1])),
        max_features=vec_cfg.get('max_features', None),
    )
    X_train_counts = vectorizer.fit_transform(X_train_text)
    X_test_counts = vectorizer.transform(X_test_text)

    if vec_cfg.get('type', 'count') == 'tfidf':
        tfidf = TfidfTransformer()
        X_train = tfidf.fit_transform(X_train_counts)
        X_test = tfidf.transform(X_test_counts)
    else:
        X_train = X_train_counts
        X_test = X_test_counts

    return vectorizer, X_train, X_test


def _predict_linear_regression(model, X, threshold=0.5):
    predictions = model.predict(X)
    return (np.asarray(predictions).flatten() >= threshold).astype(int)


def _build_models(models_cfg):
    models = []

    nb_cfg = models_cfg.get('naive_bayes', {})
    models.append(('naive_bayes', NaiveBayes(alpha=nb_cfg.get('alpha', 1.0))))

    dt_cfg = models_cfg.get('decision_tree', {})
    models.append((
        'decision_tree',
        DecisionTree(
            max_depth=dt_cfg.get('max_depth', 4),
            min_samples_split=dt_cfg.get('min_samples_split', 2),
        ),
    ))

    rf_cfg = models_cfg.get('random_forest', {})
    models.append((
        'random_forest',
        RandomForest(
            n_trees=rf_cfg.get('n_trees', 10),
            max_depth=rf_cfg.get('max_depth', 10),
            min_samples_split=rf_cfg.get('min_samples_split', 2),
            random_state=rf_cfg.get('random_state', 42),
        ),
    ))

    knn_cfg = models_cfg.get('knn', {})
    models.append(('knn', KNN(k=knn_cfg.get('k', 3))))

    perceptron_cfg = models_cfg.get('perceptron', {})
    models.append((
        'perceptron',
        Perceptron(
            learning_rate=perceptron_cfg.get('learning_rate', 0.01),
            n_iters=perceptron_cfg.get('n_iters', 1000),
        ),
    ))

    logreg_cfg = models_cfg.get('logistic_regression', {})
    models.append((
        'logistic_regression',
        LogisticRegression(
            learning_rate=logreg_cfg.get('learning_rate', 0.1),
            n_iters=logreg_cfg.get('n_iters', 1000),
            threshold=logreg_cfg.get('threshold', 0.5),
            verbose=logreg_cfg.get('verbose', False),
        ),
    ))

    svm_cfg = models_cfg.get('svm', {})
    models.append((
        'svm',
        SVM(
            learning_rate=svm_cfg.get('learning_rate', 0.001),
            lambda_param=svm_cfg.get('lambda_param', 0.01),
            n_iters=svm_cfg.get('n_iters', 1000),
        ),
    ))

    nn_cfg = models_cfg.get('neural_network', {})
    models.append((
        'neural_network',
        NeuralNetworkClassifier(
            hidden_size=nn_cfg.get('hidden_size', 32),
            learning_rate=nn_cfg.get('learning_rate', 0.05),
            n_iters=nn_cfg.get('n_iters', 500),
            random_state=nn_cfg.get('random_state', 42),
            verbose=nn_cfg.get('verbose', False),
        ),
    ))

    linreg_cfg = models_cfg.get('linear_regression', {})
    models.append((
        'linear_regression',
        LinearRegression(
            learning_rate=linreg_cfg.get('learning_rate', 0.01),
            n_iters=linreg_cfg.get('n_iters', 1000),
            verbose=linreg_cfg.get('verbose', False),
        ),
    ))

    return models


def main(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / 'experiments' / 'config.old.yaml'
    cfg = load_config(config_path)

    ds_cfg = cfg.get('dataset', {})
    preprocess_cfg = cfg.get('preprocessing', {})
    vec_cfg = cfg.get('vectorizer', {})
    models_cfg = cfg.get('models', {})
    eval_cfg = cfg.get('evaluation', {})
    out_cfg = cfg.get('output', {})

    raw_path = _resolve_path(ds_cfg.get('path', 'data/raw/sst/train.csv'))
    text_col = ds_cfg.get('text_col', 'sentence')
    label_col = ds_cfg.get('label_col', 'target')
    max_rows = ds_cfg.get('max_rows', None)

    results_dir = _resolve_path(out_cfg.get('results_dir', 'results/experiments'))
    metrics_dir = _resolve_path(out_cfg.get('metrics_dir', str(results_dir / 'metrics')))
    logs_dir = _resolve_path(out_cfg.get('logs_dir', str(results_dir / 'logs')))
    predictions_dir = _resolve_path(out_cfg.get('predictions_dir', str(results_dir / 'predictions')))
    figures_dir = _resolve_path(out_cfg.get('figures_dir', str(results_dir / 'figures')))

    for directory in (results_dir, metrics_dir, logs_dir, predictions_dir, figures_dir):
        ensure_dirs(directory)

    run_ts = int(time.time())
    with open(logs_dir / f'run_{run_ts}.log', 'w', encoding='utf-8') as lf:
        lf.write(f'Run timestamp: {run_ts}\n')
        lf.write(f'Raw dataset path: {raw_path}\n')

    print('Preprocessing...')
    sample_df = pd.read_csv(raw_path, nrows=100)
    text_col, label_col = _detect_columns(sample_df, text_col, label_col)

    cleaned_csv, _ = process_csv_to_clean(
        raw_path,
        output_dir=PROJECT_ROOT / 'data' / 'processed',
        text_col=text_col,
        label_col=label_col,
        remove_stopwords=preprocess_cfg.get('remove_stopwords', True),
        stem_func=None,
        remove_numbers=preprocess_cfg.get('remove_numbers', True),
    )

    print('Validating labels...')
    label_counts = validate_labels(cleaned_csv, label_col=label_col)
    print('Label distribution (sample):', label_counts)

    df = pd.read_csv(cleaned_csv)
    if max_rows is not None:
        df = df.head(int(max_rows))

    texts = df[text_col].astype(str).values
    labels = df[label_col].values

    test_size = eval_cfg.get('test_size', 0.2)
    random_seed = eval_cfg.get('random_seed', 42)
    stratify = eval_cfg.get('stratify', True)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        np.array(texts),
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    print('Vectorizing...')
    _, X_train, X_test = _prepare_features(X_train_text, X_test_text, vec_cfg)

    results = {}
    predictions = {'text': X_test_text, 'y_true': y_test}
    model_predictions = {}

    for model_name, model in _build_models(models_cfg):
        print(f'Training {model_name}...')
        t0 = time.time()
        model.fit(X_train, y_train)

        if model_name == 'linear_regression':
            threshold = models_cfg.get('linear_regression', {}).get('threshold', 0.5)
            y_pred = _predict_linear_regression(model, X_test, threshold=threshold)
        else:
            y_pred = model.predict(X_test)

        y_pred = np.asarray(y_pred).flatten()
        acc = accuracy(y_test, y_pred)
        cm, classes = confusion_matrix(y_test, y_pred)
        results[model_name] = {
            'accuracy': acc,
            'confusion': cm.tolist(),
            'train_seconds': time.time() - t0,
        }
        model_predictions[f'{model_name}_pred'] = y_pred
        print(f'{model_name} accuracy: {acc:.4f}')

    predictions.update(model_predictions)

    with open(metrics_dir / f'results_{run_ts}.json', 'w', encoding='utf-8') as f:
        json.dump({'config': cfg, 'results': results}, f, ensure_ascii=False, indent=2)

    if out_cfg.get('save_csv_predictions', True):
        pd.DataFrame(predictions).to_csv(predictions_dir / f'predictions_{run_ts}.csv', index=False, encoding='utf-8')

    if out_cfg.get('save_confusion', True):
        for model_name, model_result in results.items():
            cm = np.array(model_result['confusion'])
            classes = np.unique(y_test)
            fig, _ = MetricsVisualizer.plot_confusion_matrix(cm, classes, title=f'Confusion Matrix - {model_name.replace("_", " ").title()}')
            fig.savefig(figures_dir / f'cm_{model_name}_{run_ts}.png', dpi=120, bbox_inches='tight')
            plt.close(fig)

        accuracy_map = {model_name: model_result['accuracy'] for model_name, model_result in results.items()}
        fig, _ = MetricsVisualizer.plot_accuracy_comparison(accuracy_map)
        fig.savefig(figures_dir / f'accuracy_comparison_{run_ts}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

    print('Run complete')


if __name__ == '__main__':
    main()
