from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import yaml
except Exception:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline
from src.utils.experiment_tracking import resolve_experiment_save_dir


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read experiments/config.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "experiments" / "config.yaml"

    cfg = load_config(config_path)
    ds_cfg = cfg.get("dataset", {})
    preprocess_cfg = cfg.get("preprocessing", {})
    vec_cfg = cfg.get("vectorizer", {})
    eval_cfg = cfg.get("evaluation", {})
    out_cfg = cfg.get("output", {})
    save_root = PROJECT_ROOT / out_cfg.get("results_dir", "results/experiments")
    run_id = out_cfg.get("run_id")
    resolved_save_dir = resolve_experiment_save_dir(save_root, run_id=run_id)

    args = SimpleNamespace(
        dataset_path=ds_cfg.get("path"),
        dataset=ds_cfg.get("name", "sst"),
        text_col=ds_cfg.get("text_col"),
        label_col=ds_cfg.get("label_col"),
        limit=ds_cfg.get("max_rows"),
        remove_stopwords=preprocess_cfg.get("remove_stopwords", True),
        keep_numbers=not preprocess_cfg.get("remove_numbers", True),
        ngram_max=vec_cfg.get("ngram_range", [1, 1])[1],
        max_features=vec_cfg.get("max_features", 5000),
        test_size=eval_cfg.get("test_size", 0.2),
        random_state=eval_cfg.get("random_seed", 42),
        save_dir=str(resolved_save_dir),
        run_id=resolved_save_dir.name,
        config_path=str(Path(config_path).resolve()),
    )

    run_pipeline(args)


if __name__ == "__main__":
    main()
