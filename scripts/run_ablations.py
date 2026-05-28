from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"


def _run_command(command: list[str]):
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"CMD: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def _load_metrics(run_dir: Path):
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(run_dir / "classification_report.json", "r", encoding="utf-8") as handle:
        reports = json.load(handle)

    rows = []
    for model_name, model_metrics in metrics.items():
        macro = next((r for r in reports.get(model_name, []) if r.get("label") == "macro_avg"), {})
        rows.append(
            {
                "model": model_name,
                "accuracy": model_metrics.get("accuracy"),
                "macro_f1": macro.get("f1"),
                "macro_precision": macro.get("precision"),
                "macro_recall": macro.get("recall"),
            }
        )
    return rows


def run_ablation(args):
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    settings = []

    for remove_stopwords in [True, False]:
        for ngram_max in [1, 2, 3]:
            for max_features in [1000, 5000, 10000]:
                run_id = f"abl_stop{int(remove_stopwords)}_ng{ngram_max}_mf{max_features}"
                run_dir = results_root / run_id

                command = [
                    sys.executable,
                    str(MAIN_PY),
                    "--dataset",
                    args.dataset,
                    "--limit",
                    str(args.limit),
                    "--ngram-max",
                    str(ngram_max),
                    "--max-features",
                    str(max_features),
                    "--random-state",
                    str(args.random_state),
                    "--test-size",
                    str(args.test_size),
                    "--save-dir",
                    str(run_dir),
                    "--run-id",
                    run_id,
                ]

                if remove_stopwords:
                    command.append("--remove-stopwords")
                if args.keep_numbers:
                    command.append("--keep-numbers")

                _run_command(command)

                metric_rows = _load_metrics(run_dir)
                for metric_row in metric_rows:
                    settings.append(
                        {
                            "run_id": run_id,
                            "dataset": args.dataset,
                            "limit": args.limit,
                            "remove_stopwords": remove_stopwords,
                            "ngram_max": ngram_max,
                            "max_features": max_features,
                            **metric_row,
                        }
                    )

    frame = pd.DataFrame(settings)
    frame = frame.sort_values(by=["model", "remove_stopwords", "ngram_max", "max_features"])

    out_csv = results_root / "ablation_summary.csv"
    out_md = results_root / "ablation_summary.md"

    frame.to_csv(out_csv, index=False)

    md_cols = [
        "run_id",
        "model",
        "accuracy",
        "macro_f1",
        "remove_stopwords",
        "ngram_max",
        "max_features",
    ]
    md_frame = frame[md_cols].copy()
    for col in ["accuracy", "macro_f1"]:
        md_frame[col] = md_frame[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "")

    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write("# Ablation Summary\n\n")
        handle.write(md_frame.to_markdown(index=False))
        handle.write("\n")

    print(f"Ablation runs completed: {len(frame)} model rows")
    print(f"CSV: {out_csv}")
    print(f"MD : {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Run controlled ablations for sentiment baseline.")
    parser.add_argument("--dataset", type=str, choices=["sst", "imdb"], default="sst")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--keep-numbers", action="store_true")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "ablations"),
    )
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
