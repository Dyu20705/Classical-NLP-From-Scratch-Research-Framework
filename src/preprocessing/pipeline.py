"""
Preprocessing pipeline utilities for the project.

Provides a reproducible, chunked CSV processing flow that:
- reads raw CSVs (streaming when large)
- cleans text (lowercase, remove html, urls, emails, numbers, special chars)
- tokenizes (whitespace)
- removes stopwords
- optionally stems or lemmatizes if external stemmer is provided
- handles empty samples and validates labels
- writes cleaned outputs to data/processed/ in CSV and JSONL

Design notes:
- This module avoids converting sparse matrices to dense during preprocessing.
- Stemming/lemmatization is optional and delegated to an external function to avoid adding heavy dependencies.
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from src.preprocessing.text_processor import TextProcessor

CSV_CHUNKSIZE = 10000


def _normalize_text_input(text):
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='ignore')
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r"^b\s*(['\"])", r"\1", text)
        text = re.sub(r"^b\s+", "", text)
        if (text.startswith("b'") and text.endswith("'")) or (text.startswith('b"') and text.endswith('"')):
            return text[2:-1]
    return text


def _clean_text_basic(text, remove_numbers=True):
    text = _normalize_text_input(text)
    if not isinstance(text, str):
        return ''

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers if requested
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Keep only letters, spaces and apostrophes
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def process_csv_to_clean(input_path,
                         output_dir='data/processed',
                         text_col='sentence',
                         label_col='target',
                         remove_stopwords=True,
                         stem_func=None,
                         remove_numbers=True,
                         force=False):
    """
    Process a CSV file (possibly large) and save cleaned outputs.

    Parameters
    ----------
    input_path : str or Path
        Path to raw CSV file.
    output_dir : str
        Directory where processed outputs are saved.
    text_col, label_col : str
        Column names for text and label in the input CSV.
    remove_stopwords : bool
        Whether to remove stopwords.
    stem_func : callable or None
        Optional function token -> stem(token). If None, no stemming done.
    remove_numbers : bool
        Whether to strip numeric characters.
    force : bool
        If True, overwrite existing outputs.

    Outputs
    -------
    - CSV saved at data/processed/<basename>_clean.csv
    - JSONL saved at data/processed/<basename>_clean.jsonl
    """
    input_path = Path(input_path)
    assert input_path.exists(), f"Input file does not exist: {input_path}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = input_path.stem
    csv_out = output_dir / f"{base}_clean.csv"
    jsonl_out = output_dir / f"{base}_clean.jsonl"

    if csv_out.exists() and jsonl_out.exists():
        if not force:
            print(f"Processed files already exist for {input_path}. Use force=True to overwrite.")
            return str(csv_out), str(jsonl_out)
        csv_out.unlink()
        jsonl_out.unlink()

    processor = TextProcessor(remove_stopwords=remove_stopwords, remove_numbers=remove_numbers)

    rows_written = 0
    # Stream CSV in chunks to support large files
    for chunk in pd.read_csv(input_path, chunksize=CSV_CHUNKSIZE, iterator=True, encoding='utf-8', low_memory=True):
        # Validate expected columns
        if text_col not in chunk.columns:
            raise ValueError(f"Text column '{text_col}' not found in input CSV")

        if label_col not in chunk.columns:
            # allow missing label_col for unlabeled datasets; create placeholder
            chunk[label_col] = None

        # Clean and tokenize
        processed_texts = []
        valid_labels = []
        for text, label in zip(chunk[text_col].fillna(''), chunk[label_col]):
            cleaned = _clean_text_basic(text, remove_numbers=remove_numbers)
            tokens = processor.tokenize(cleaned)
            if remove_stopwords:
                tokens = processor.remove_stopwords_tokens(tokens)
            if stem_func is not None:
                try:
                    tokens = [stem_func(t) for t in tokens]
                except Exception:
                    # if stemming fails, fallback to tokens
                    pass

            # Handle empty samples by replacing with a single placeholder token
            if len(tokens) == 0:
                tokens = ['']

            processed_texts.append(' '.join(tokens))
            valid_labels.append(label)

        out_df = pd.DataFrame({text_col: processed_texts, label_col: valid_labels})

        # Append to CSV (write header the first time)
        if not csv_out.exists():
            out_df.to_csv(csv_out, index=False, encoding='utf-8')
        else:
            out_df.to_csv(csv_out, mode='a', index=False, header=False, encoding='utf-8')

        # Append to JSONL
        with open(jsonl_out, 'a', encoding='utf-8') as jf:
            for _, row in out_df.iterrows():
                jf.write(json.dumps(row.dropna().to_dict(), ensure_ascii=False) + '\n')

        rows_written += len(out_df)
        print(f"Processed {rows_written} rows so far")

    print(f"Finished processing. Saved CSV: {csv_out}, JSONL: {jsonl_out}")
    return str(csv_out), str(jsonl_out)


def validate_labels(csv_path, label_col='target'):
    """
    Quick label validation: returns observed classes and counts.
    """
    df = pd.read_csv(csv_path, usecols=[label_col])
    counts = df[label_col].value_counts(dropna=False)
    return counts.to_dict()


if __name__ == '__main__':
    # Example usage (do not run on import)
    pass
