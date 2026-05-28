from __future__ import annotations

from pathlib import Path
import ast

import pandas as pd


def resolve_project_root(start: Path | None = None) -> Path:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    current = start or Path.cwd()
    candidates = [current, current.parent]

    for candidate in candidates:
        if (candidate / "data" / "raw").exists() and (candidate / "src").exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve project root. Expected a folder containing both 'data/raw' and 'src'."
    )


def find_csv_file(folder: Path) -> Path:
\
\
\
\
       
    candidates = sorted(folder.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in {folder}")
    return candidates[0]


def decode_text(value) -> str:
\
\
\
\
\
\
       
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
            pass

    return text


def infer_text_label_columns(df: pd.DataFrame) -> tuple[str, str]:
\
\
\
\
\
\
       
    lower_map = {column.lower(): column for column in df.columns}
    text_candidates = ["sentence", "review", "text", "content", "comment"]
    label_candidates = ["target", "label", "sentiment", "polarity", "class"]

    text_col = next((lower_map[candidate] for candidate in text_candidates if candidate in lower_map), df.columns[0])
    label_col = next((lower_map[candidate] for candidate in label_candidates if candidate in lower_map), df.columns[-1])
    return text_col, label_col


def load_sentiment_dataset(path: Path, name: str, limit: int | None = None) -> dict:
\
\
\
\
\
       
    df = pd.read_csv(path)
    text_col, label_col = infer_text_label_columns(df)
    work = df[[text_col, label_col]].copy()
    work.columns = ["text", "label"]
    work["text"] = work["text"].map(decode_text).astype(str)
    work["label"] = work["label"].copy()

    if limit is not None:
        work = work.head(limit)

    return {
        "name": name,
        "path": path,
        "raw": df,
        "data": work,
        "text_col": text_col,
        "label_col": label_col,
    }


def text_length_stats(text_series: pd.Series) -> pd.Series:
                                                                 
    return text_series.fillna("").astype(str).map(lambda value: len(value.split()))


def summarize_dataset(bundle: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
\
\
\
\
\
       
    data = bundle["data"]
    lengths = text_length_stats(data["text"])
    label_counts = data["label"].value_counts(dropna=False).sort_index()
    summary = pd.DataFrame(
        {
            "dataset": [bundle["name"]],
            "rows": [len(data)],
            "unique_labels": [label_counts.shape[0]],
            "avg_tokens": [lengths.mean()],
            "median_tokens": [lengths.median()],
            "min_tokens": [lengths.min()],
            "max_tokens": [lengths.max()],
        }
    )
    return summary, label_counts, lengths
