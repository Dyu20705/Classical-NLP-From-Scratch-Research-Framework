from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_macro(report_records: list[dict]) -> dict:
    macro = next((row for row in report_records if row.get("label") == "macro_avg"), None)
    if macro is None:
        return {"macro_precision": None, "macro_recall": None, "macro_f1": None}
    return {
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1"),
    }


def aggregate(results_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        run_dir = metrics_path.parent

        report_path = run_dir / "classification_report.json"
        meta_path = run_dir / "metadata.json"

        if not report_path.exists():
            continue

        metrics_payload = _load_json(metrics_path)
        report_payload = _load_json(report_path)
        meta_payload = _load_json(meta_path) if meta_path.exists() else {}

        for model_name, model_metrics in metrics_payload.items():
            macro_metrics = _extract_macro(report_payload.get(model_name, []))
            rows.append(
                {
                    "experiment_id": run_dir.name,
                    "model": model_name,
                    "accuracy": model_metrics.get("accuracy"),
                    "macro_precision": macro_metrics["macro_precision"],
                    "macro_recall": macro_metrics["macro_recall"],
                    "macro_f1": macro_metrics["macro_f1"],
                    "dataset": meta_payload.get("dataset"),
                    "rows_used": meta_payload.get("rows_used"),
                    "random_seed": meta_payload.get("random_seed"),
                    "ngram_max": meta_payload.get("ngram_max"),
                    "max_features": meta_payload.get("max_features"),
                    "remove_stopwords": meta_payload.get("remove_stopwords"),
                    "keep_numbers": meta_payload.get("keep_numbers"),
                    "elapsed_seconds": meta_payload.get("elapsed_seconds"),
                    "timestamp_utc": meta_payload.get("timestamp_utc"),
                    "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by=["timestamp_utc", "experiment_id", "model"], ascending=[True, True, True])
    return frame


def to_markdown_table(frame: pd.DataFrame) -> str:
    display_cols = [
        "experiment_id",
        "model",
        "accuracy",
        "macro_f1",
        "dataset",
        "rows_used",
        "ngram_max",
        "max_features",
        "remove_stopwords",
        "elapsed_seconds",
    ]

    table = frame[display_cols].copy()
    for col in ["accuracy", "macro_f1", "elapsed_seconds"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "")

    return table.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment result artifacts into benchmark tables.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "experiments"),
        help="Directory containing experiment run folders with metrics.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="benchmark_summary",
        help="Output filename prefix (without extension)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    frame = aggregate(results_root)

    if frame.empty:
        raise SystemExit(f"No experiment results found under {results_root}")

    out_csv = results_root / f"{args.output_prefix}.csv"
    out_md = results_root / f"{args.output_prefix}.md"

    frame.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write("# Benchmark Summary\n\n")
        handle.write(to_markdown_table(frame))
        handle.write("\n")

    print(f"Aggregated {len(frame)} rows")
    print(f"CSV: {out_csv}")
    print(f"MD : {out_md}")


if __name__ == "__main__":
    main()
