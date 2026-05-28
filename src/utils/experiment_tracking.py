from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_getattr(obj, name, default=None):
    return getattr(obj, name, default)


def _git_commit(project_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def _environment_text(metadata: dict) -> str:
    lines = [
        f"timestamp_utc={metadata['timestamp_utc']}",
        f"git_commit={metadata['git_commit']}",
        f"python_version={metadata['python_version']}",
        f"platform={metadata['platform']}",
        f"executable={metadata['python_executable']}",
        f"dataset={metadata['dataset']}",
        f"dataset_path={metadata['dataset_path']}",
        f"dataset_checksum_sha256={metadata['dataset_checksum_sha256']}",
        f"random_seed={metadata['random_seed']}",
        f"ngram_max={metadata['ngram_max']}",
        f"max_features={metadata['max_features']}",
        f"remove_stopwords={metadata['remove_stopwords']}",
        f"keep_numbers={metadata['keep_numbers']}",
        f"test_size={metadata['test_size']}",
        f"rows_used={metadata['rows_used']}",
        f"run_id={metadata['run_id']}",
        f"elapsed_seconds={metadata['elapsed_seconds']}",
    ]
    return "\n".join(lines) + "\n"


def resolve_experiment_save_dir(base_dir: Path, run_id: str | None = None) -> Path:
    base_dir = base_dir.resolve()
    if run_id:
        return base_dir / run_id

    base_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.startswith("exp_"):
            suffix = child.name.split("exp_", 1)[1]
            if suffix.isdigit():
                existing_ids.append(int(suffix))

    next_id = max(existing_ids, default=0) + 1
    return base_dir / f"exp_{next_id:03d}"


def write_experiment_tracking(save_dir: Path, args, dataset_bundle: dict, model_results: dict, elapsed_seconds: float):
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_bundle["path"]).resolve()
    project_root = Path(__file__).resolve().parents[2]
    git_commit = _git_commit(project_root)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "dataset": _safe_getattr(args, "dataset", "custom"),
        "dataset_path": str(dataset_path),
        "dataset_checksum_sha256": _sha256_file(dataset_path),
        "random_seed": _safe_getattr(args, "random_state", 42),
        "ngram_max": _safe_getattr(args, "ngram_max", 1),
        "max_features": _safe_getattr(args, "max_features", 5000),
        "remove_stopwords": bool(_safe_getattr(args, "remove_stopwords", False)),
        "keep_numbers": bool(_safe_getattr(args, "keep_numbers", False)),
        "test_size": float(_safe_getattr(args, "test_size", 0.2)),
        "rows_used": int(len(dataset_bundle["frame"])),
        "text_column": dataset_bundle["text_col"],
        "label_column": dataset_bundle["label_col"],
        "run_id": _safe_getattr(args, "run_id", None),
        "config_path": _safe_getattr(args, "config_path", None),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "models": sorted(list(model_results.keys())),
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    with open(save_dir / "git_commit.txt", "w", encoding="utf-8") as handle:
        handle.write(f"{git_commit}\n")

    with open(save_dir / "environment.txt", "w", encoding="utf-8") as handle:
        handle.write(_environment_text(metadata))
