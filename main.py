from __future__ import annotations

import argparse
import ast
import json
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.feature_extraction.count_vectorizer import CountVectorizer
from src.models.decision_tree.decision_tree import DecisionTree
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.preprocessing.text_processor import TextProcessor
from src.utils.helper import train_test_split
from src.utils.experiment_tracking import write_experiment_tracking

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the sentiment baseline.")
    parser.add_argument("--dataset", choices=["imdb", "sst"], default="sst")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument("--keep-numbers", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=str(RESULTS_ROOT / "cli"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    return parser.parse_args()


def decode_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
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
        if candidates:
            return candidates[0]
        fallback = DATA_ROOT / "sst" / "train.csv"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No CSV file found in {DATA_ROOT / 'imdb'}")
    return DATA_ROOT / "sst" / "train.csv"


def normalize_labels(series):
    raw = series.copy()
    if pd.api.types.is_numeric_dtype(raw):
        unique_values = sorted(pd.unique(raw.dropna()))
        if set(unique_values).issubset({0, 1}):
            return raw.astype(int).to_numpy(), {0: "negative", 1: "positive"}

    normalized = raw.astype(str).str.strip().str.lower()
    positive_aliases = {"1", "positive", "pos", "true", "yes"}
    negative_aliases = {"0", "negative", "neg", "false", "no"}

    if normalized.isin(positive_aliases | negative_aliases).all():
        encoded = normalized.map(lambda value: 1 if value in positive_aliases else 0)
        return encoded.astype(int).to_numpy(), {0: "negative", 1: "positive"}

    codes, uniques = pd.factorize(raw)
    lookup = {idx: str(label) for idx, label in enumerate(uniques.tolist())}
    return codes.astype(int), lookup


def load_dataset(args):
    dataset_path = resolve_dataset_path(args)
    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset at {dataset_path}: {exc}") from exc

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
    processor = TextProcessor(remove_stopwords=remove_stopwords, remove_numbers=remove_numbers)
    processed = [" ".join(processor.process(text)) for text in texts]
    return processed, processor


def build_features(train_texts, test_texts, ngram_max, max_features):
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, ngram_max), max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return {"vectorizer": vectorizer, "X_train": X_train, "X_test": X_test}


def compute_metrics(y_true, y_pred):
    cm, classes = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    report_rows = []
    for idx, cls in enumerate(classes):
        report_rows.append(
            {
                "label": str(cls),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(cm[idx].sum()),
            }
        )

    report_rows.append(
        {
            "label": "macro_avg",
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "support": int(len(y_true)),
        }
    )

    report = pd.DataFrame(report_rows)

    return {
        "accuracy": float(accuracy(y_true, y_pred)),
        "confusion_matrix": cm,
        "classes": classes,
        "report": report,
    }


def train_models(X_train, X_test, y_train, y_test):
    results = {}

    nb_model = NaiveBayes(alpha=1.0)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    results["naive_bayes"] = {
        "model": nb_model,
        "predictions": nb_pred,
        "metrics": compute_metrics(y_test, nb_pred),
    }

    dt_model = DecisionTree(max_depth=8, min_samples_split=5)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    results["decision_tree"] = {
        "model": dt_model,
        "predictions": dt_pred,
        "metrics": compute_metrics(y_test, dt_pred),
    }

    return results


def save_outputs(save_dir, results, y_test, label_lookup):
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {}
    report_payload = {}
    for model_name, bundle in results.items():
        metrics = bundle["metrics"]
        metrics_payload[model_name] = {
            "accuracy": metrics["accuracy"],
            "classes": [str(value) for value in metrics["classes"].tolist()],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }
        report_payload[model_name] = bundle["metrics"]["report"].to_dict(orient="records")

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)

    with open(save_dir / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    prediction_frame = pd.DataFrame({"y_true": y_test})
    for model_name, bundle in results.items():
        prediction_frame[f"{model_name}_pred"] = bundle["predictions"]
    prediction_frame.to_csv(save_dir / "predictions.csv", index=False)

    with open(save_dir / "label_lookup.json", "w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in label_lookup.items()}, handle, indent=2, ensure_ascii=False)


def print_summary(dataset_bundle, features, results):
    data = dataset_bundle["frame"]
    print("\n" + "=" * 72)
    print("SENTIMENT ANALYSIS BASELINE")
    print("=" * 72)
    print(f"Dataset path      : {dataset_bundle['path']}")
    print(f"Rows used         : {len(data)}")
    print(f"Text column       : {dataset_bundle['text_col']}")
    print(f"Label column      : {dataset_bundle['label_col']}")
    print(f"Vocabulary size   : {len(features['vectorizer'].vocabulary_)}")
    print(f"Train matrix shape: {features['X_train'].shape}")
    print(f"Test matrix shape : {features['X_test'].shape}")
    print("-" * 72)

    for model_name in ["naive_bayes", "decision_tree"]:
        metrics = results[model_name]["metrics"]
        print(f"{model_name:20s} acc={metrics['accuracy']:.4f}")
        print(metrics["report"].to_string(index=False))
        print("-" * 72)


def run_pipeline(args):
    started = perf_counter()
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
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )
    results = train_models(features["X_train"], features["X_test"], y_train, y_test)
    resolved_save_dir = Path(args.save_dir).resolve()
    save_outputs(resolved_save_dir, results, y_test, dataset_bundle["label_lookup"])

    write_experiment_tracking(
        save_dir=resolved_save_dir,
        args=args,
        dataset_bundle=dataset_bundle,
        model_results=results,
        elapsed_seconds=perf_counter() - started,
    )

    print_summary(dataset_bundle, features, results)
    print(f"Artifacts saved   : {resolved_save_dir}")


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
