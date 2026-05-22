"""
Command-line entry point for the sentiment analysis project.

This script provides a reproducible baseline workflow:
1. load a dataset from ``data/raw``
2. preprocess text with the project text processor
3. split train/test with optional stratification
4. vectorize with CountVectorizer or TF-IDF
5. train one or more repository models
6. report metrics and optionally save visualizations
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.evaluation.visualization import MetricsVisualizer
from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer
from src.models.decision_tree.decision_tree import DecisionTree
from src.models.knn.knn import KNN
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.models.regression.logistic import LogisticRegression
from src.models.svms.svm import SVM
from src.preprocessing.text_processor import TextProcessor
from src.utils.helper import train_test_split

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a sentiment analysis baseline.")
    parser.add_argument(
        "--dataset",
        choices=["imdb", "sst"],
        default="sst",
        help="Dataset under data/raw to use.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional explicit CSV path. Overrides --dataset.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Optional text column override.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Optional label column override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of rows to use. Use 0 or a negative value for all rows.",
    )
    parser.add_argument(
        "--vectorizer",
        choices=["count", "tfidf"],
        default="tfidf",
        help="Feature representation.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=1,
        help="Upper bound of the n-gram range. Lower bound is fixed at 1.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum vocabulary size.",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords during preprocessing.",
    )
    parser.add_argument(
        "--keep-numbers",
        action="store_true",
        help="Keep numeric characters during preprocessing.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and split.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["naive_bayes", "decision_tree", "knn", "logistic_regression", "svm"],
        default=["naive_bayes", "decision_tree"],
        help="Models to train.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(RESULTS_ROOT / "cli"),
        help="Directory for metrics and figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def decode_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    if text.startswith(("b'", 'b"')):
        try:
            decoded = ast.literal_eval(text)
            if isinstance(decoded, bytes):
                return decoded.decode("utf-8", errors="ignore")
            return str(decoded)
        except Exception:
            return text
    return text


def infer_text_label_columns(df):
    lower_map = {column.lower(): column for column in df.columns}
    text_candidates = ["sentence", "review", "text", "content", "comment"]
    label_candidates = ["target", "label", "sentiment", "polarity", "class"]

    text_col = next((lower_map[name] for name in text_candidates if name in lower_map), df.columns[0])
    label_col = next((lower_map[name] for name in label_candidates if name in lower_map), df.columns[-1])
    return text_col, label_col


def resolve_dataset_path(args):
    if args.dataset_path:
        return Path(args.dataset_path).resolve()

    if args.dataset == "imdb":
        candidates = sorted((DATA_ROOT / "imdb").glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV file found in {DATA_ROOT / 'imdb'}")
        return candidates[0]

    return DATA_ROOT / "sst" / "train.csv"


def normalize_labels(series):
    raw = series.copy()

    if pd.api.types.is_numeric_dtype(raw):
        unique_values = sorted(pd.unique(raw.dropna()))
        if set(unique_values).issubset({0, 1}):
            return raw.astype(int).to_numpy(), {0: 0, 1: 1}

    normalized = raw.astype(str).str.strip().str.lower()
    positive_aliases = {"1", "positive", "pos", "true", "yes"}
    negative_aliases = {"0", "negative", "neg", "false", "no"}

    if normalized.isin(positive_aliases | negative_aliases).all():
        encoded = normalized.map(lambda value: 1 if value in positive_aliases else 0)
        return encoded.astype(int).to_numpy(), {0: "negative", 1: "positive"}

    codes, uniques = pd.factorize(raw)
    return codes.astype(int), {idx: label for idx, label in enumerate(uniques.tolist())}


def load_dataset(args):
    dataset_path = resolve_dataset_path(args)
    df = pd.read_csv(dataset_path)

    text_col, label_col = infer_text_label_columns(df)
    if args.text_col:
        text_col = args.text_col
    if args.label_col:
        label_col = args.label_col

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns not found. Available columns: {list(df.columns)}")

    work = df[[text_col, label_col]].copy()
    work.columns = ["text", "label"]
    work["text"] = work["text"].map(decode_text).astype(str)
    work = work.dropna(subset=["text", "label"])
    work = work[work["text"].str.strip() != ""]

    if args.limit and args.limit > 0:
        work = work.head(args.limit).copy()

    labels, label_lookup = normalize_labels(work["label"])
    work["label_id"] = labels

    return {
        "path": dataset_path,
        "frame": work,
        "text_col": text_col,
        "label_col": label_col,
        "label_lookup": label_lookup,
    }


def preprocess_texts(texts, remove_stopwords=False, remove_numbers=True):
    processor = TextProcessor(
        remove_stopwords=remove_stopwords,
        remove_numbers=remove_numbers,
    )
    processed = [" ".join(processor.process(text)) for text in texts]
    return processed, processor


def build_features(train_texts, test_texts, vectorizer_type, ngram_max, max_features):
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, ngram_max),
        max_features=max_features,
    )

    X_train_counts = vectorizer.fit_transform(train_texts)
    X_test_counts = vectorizer.transform(test_texts)

    features = {
        "vectorizer": vectorizer,
        "X_train_counts": X_train_counts,
        "X_test_counts": X_test_counts,
        "representation": "count",
    }

    if vectorizer_type == "tfidf":
        transformer = TfidfTransformer()
        features["tfidf_transformer"] = transformer
        features["X_train"] = transformer.fit_transform(X_train_counts)
        features["X_test"] = transformer.transform(X_test_counts)
        features["representation"] = "tfidf"
    else:
        features["X_train"] = X_train_counts
        features["X_test"] = X_test_counts

    return features


def make_model(name):
    if name == "naive_bayes":
        return NaiveBayes(alpha=1.0)
    if name == "decision_tree":
        return DecisionTree(max_depth=6, min_samples_split=2)
    if name == "knn":
        return KNN(k=5)
    if name == "logistic_regression":
        return LogisticRegression(learning_rate=0.1, n_iters=300, threshold=0.5, verbose=False)
    if name == "svm":
        return SVM(learning_rate=0.001, lambda_param=0.01, n_iters=300)
    raise ValueError(f"Unsupported model: {name}")


def compute_metrics(y_true, y_pred):
    cm, classes = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    return {
        "accuracy": float(accuracy(y_true, y_pred)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "confusion_matrix": cm,
        "classes": classes,
    }


def train_models(model_names, X_train, X_test, y_train, y_test):
    results = {}

    for model_name in model_names:
        model = make_model(model_name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = compute_metrics(y_test, predictions)
        results[model_name] = {
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
        }

    return results


def save_outputs(save_dir, results, y_test, label_lookup, make_plots):
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {}
    for model_name, bundle in results.items():
        metrics = bundle["metrics"]
        metrics_payload[model_name] = {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "classes": metrics["classes"].tolist(),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)

    prediction_frame = pd.DataFrame({"y_true": y_test})
    for model_name, bundle in results.items():
        prediction_frame[f"{model_name}_pred"] = bundle["predictions"]
    prediction_frame.to_csv(save_dir / "predictions.csv", index=False)

    with open(save_dir / "label_lookup.json", "w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in label_lookup.items()}, handle, indent=2, ensure_ascii=False)

    if not make_plots:
        return

    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    accuracy_map = {name: bundle["metrics"]["accuracy"] for name, bundle in results.items()}
    fig, _ = MetricsVisualizer.plot_accuracy_comparison(accuracy_map)
    fig.savefig(figures_dir / "accuracy_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    best_model_name = max(results.items(), key=lambda item: item[1]["metrics"]["f1_macro"])[0]
    for model_name, bundle in results.items():
        metrics = bundle["metrics"]
        fig, _ = MetricsVisualizer.plot_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["classes"],
            title=f"Confusion Matrix - {model_name}",
            normalize=True,
        )
        fig.savefig(figures_dir / f"confusion_matrix_{model_name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    fig, _ = MetricsVisualizer.plot_roc_style_metrics(y_test, results[best_model_name]["predictions"])
    fig.savefig(figures_dir / f"metrics_{best_model_name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_summary(dataset_bundle, features, results):
    data = dataset_bundle["frame"]

    print("\n" + "=" * 72)
    print("SENTIMENT ANALYSIS BASELINE")
    print("=" * 72)
    print(f"Dataset path      : {dataset_bundle['path']}")
    print(f"Rows used         : {len(data)}")
    print(f"Text column       : {dataset_bundle['text_col']}")
    print(f"Label column      : {dataset_bundle['label_col']}")
    print(f"Representation    : {features['representation']}")
    print(f"Vocabulary size   : {len(features['vectorizer'].vocabulary_)}")
    print(f"Train matrix shape: {features['X_train'].shape}")
    print(f"Test matrix shape : {features['X_test'].shape}")
    print("-" * 72)

    for model_name, bundle in sorted(results.items(), key=lambda item: item[1]["metrics"]["f1_macro"], reverse=True):
        metrics = bundle["metrics"]
        print(
            f"{model_name:20s} "
            f"acc={metrics['accuracy']:.4f} "
            f"precision={metrics['precision_macro']:.4f} "
            f"recall={metrics['recall_macro']:.4f} "
            f"f1={metrics['f1_macro']:.4f}"
        )


def main():
    args = parse_args()
    dataset_bundle = load_dataset(args)

    processed_texts, _ = preprocess_texts(
        dataset_bundle["frame"]["text"].tolist(),
        remove_stopwords=args.remove_stopwords,
        remove_numbers=not args.keep_numbers,
    )
    labels = dataset_bundle["frame"]["label_id"].to_numpy()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        np.array(processed_texts),
        np.array(labels),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )

    features = build_features(
        X_train_text,
        X_test_text,
        vectorizer_type=args.vectorizer,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )
    results = train_models(args.models, features["X_train"], features["X_test"], y_train, y_test)

    save_dir = Path(args.save_dir).resolve()
    save_outputs(
        save_dir=save_dir,
        results=results,
        y_test=y_test,
        label_lookup=dataset_bundle["label_lookup"],
        make_plots=not args.no_plots,
    )
    print_summary(dataset_bundle, features, results)
    print(f"Artifacts saved   : {save_dir}")


if __name__ == "__main__":
    main()
