# Engineering Review — ml-models-sentiment

This review highlights architectural weaknesses, duplication, and recommended refactors to improve maintainability, scalability and research quality.

Findings:

- Weak architecture:
  - `main.py` still mixes orchestration, preprocessing, training and visualization; the reusable experiment logic now lives in `experiments/run_baseline.py`, but the entry points remain somewhat duplicated.
  - `experiments/run_baseline.py` is now implemented and saves metrics, predictions, and figures; the remaining gap is tighter integration with the report workflow.

- Code duplication and poor modularization:
  - The text preprocessing path is now cleaner, but the project still has overlapping documentation and experiment notes that should be consolidated.
  - Vectorizer and transformer are modular, but model training still converts sparse→dense inside the custom `NaiveBayes` and `DecisionTree` implementations; this should be centralized or rewritten for sparse inputs.

- Missing abstractions:
  - No dedicated `Experiment` class or CLI wrapper yet; reproducible runs still depend on config-driven script execution.
  - No structured logger; prints are still used instead of `logging`.

- Missing experiment tracking:
  - Run metadata is saved to JSON, but there is still no git commit hash capture or environment snapshot in the default workflow.

- Reproducibility issues:
  - The current split helper is deterministic with `random_state=42`, but reproducibility is still constrained by manual script execution order and external data availability.

- Data leakage risks:
  - The current baseline follows train/test separation correctly, but any future notebook or manual experiment must keep fit/transform boundaries explicit.

- Evaluation flaws:
  - The baseline currently reports accuracy and confusion matrix; per-class precision/recall/F1 would make the report more useful.

- Scalability concerns:
  - Converting sparse matrices to dense still limits scaling on realistic vocabulary sizes.
  - The decision tree implementation iterates over all features and thresholds, which remains expensive for NLP features.

Recommendations:

1. Move dense conversion out of model internals and prefer sparse-aware training for Naive Bayes and Decision Tree.
2. Add a lightweight Experiment runner or CLI wrapper that standardizes preprocess -> split -> fit -> evaluate -> save.
3. Add structured logging and save git commit hash / environment metadata with each run.
4. Expand metrics beyond accuracy to include macro/micro precision, recall, and F1.
5. Keep the Vietnamese academic report synchronized with implementation changes and generated figures.

Short-term tasks to improve repository:
- Keep `TextProcessor` and vocabulary handling aligned with the current pipeline.
- Continue maintaining `src/preprocessing/pipeline.py` and `experiments/run_baseline.py` as the canonical baseline workflow.
- Keep `report/BaoCaoBaiTap.md` and `report/IMAGES_GUIDE.md` aligned with current outputs.
- Remove or merge low-value note files in `docs/` to reduce duplication.

Next steps (medium-term):
- Implement sparse-aware NB and optimize DecisionTree or switch to ensemble methods
- Add automatic saving of git metadata and environment (pip freeze)
- Replace print statements with `logging`
- Generate report figures automatically into `report/images/`

These changes will improve reproducibility, research rigor, and make the project maintainable for undergraduate research and engineering purposes.
