from __future__ import annotations

from pathlib import Path
import ast

import pandas as pd


def resolve_project_root(start: Path | None = None) -> Path:
    """Return the repository root for notebook usage.

    Why this exists:
    - In a notebook, `Path.cwd()` is not guaranteed to be the repository root.
    - The notebook may be started from the workspace root, from the notebook
      folder, or from some other current working directory depending on the
      editor session.
    - The helper checks a small set of likely locations and returns the first
      one that clearly looks like this project.

    The function is intentionally conservative:
    - It only accepts a directory as the project root if both `data/raw` and
      `src` exist there.
    - If neither the current folder nor its parent match, it raises an error
      instead of guessing silently.
    """
    current = start or Path.cwd()
    candidates = [current, current.parent]

    for candidate in candidates:
        if (candidate / "data" / "raw").exists() and (candidate / "src").exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve project root. Expected a folder containing both 'data/raw' and 'src'."
    )


def find_csv_file(folder: Path) -> Path:
    """Return the first CSV file found in a folder.

    The notebook uses this because some dataset folders may contain a single CSV
    file whose exact filename is not important for the experiment.
    """
    candidates = sorted(folder.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in {folder}")
    return candidates[0]


def decode_text(value) -> str:
    """Convert a raw cell value into plain text.

    Some dataset columns store text as byte-string-like literals such as
    "b'example text'". This helper decodes those values so downstream cleaning
    receives normal Unicode text. If decoding fails, the original string form is
    returned so the notebook can keep running instead of crashing on one bad row.
    """
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
    """Guess which columns in a dataset contain text and labels.

    The project works with multiple sentiment datasets that do not always share
    identical column names. This function uses a small priority list so the
    notebook can normalize each dataset into a shared schema without hard-coding
    one dataset's layout.
    """
    lower_map = {column.lower(): column for column in df.columns}
    text_candidates = ["sentence", "review", "text", "content", "comment"]
    label_candidates = ["target", "label", "sentiment", "polarity", "class"]

    text_col = next((lower_map[candidate] for candidate in text_candidates if candidate in lower_map), df.columns[0])
    label_col = next((lower_map[candidate] for candidate in label_candidates if candidate in lower_map), df.columns[-1])
    return text_col, label_col


def load_sentiment_dataset(path: Path, name: str, limit: int | None = None) -> dict:
    """Load a sentiment CSV and normalize it into a shared structure.

    The returned dictionary keeps both the raw frame and the cleaned two-column
    frame so the notebook can show the original data, summarize it, and then use
    the normalized version for model experiments.
    """
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
    """Count whitespace-separated tokens for each text sample."""
    return text_series.fillna("").astype(str).map(lambda value: len(value.split()))


def summarize_dataset(bundle: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the compact summary table used in the notebook.

    The summary is intentionally small: row count, label count, and basic token
    length statistics are enough to spot obvious dataset issues without turning
    the notebook into a reporting dashboard.
    """
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
