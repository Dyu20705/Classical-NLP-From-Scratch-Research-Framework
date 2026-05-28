# Image and Figure Workflow

This document specifies how to generate, name, organize and integrate figures into `BaoCaoBaiTap.md`.

Directory:
- Store all figures in `report/images/`.
- Use subfolders per experiment if needed (e.g., `report/images/exp_20260522/`).

Naming conventions:
- Use descriptive, lowercase, underscore-separated names, include model and metric, and date prefix when appropriate.
  - Examples:
    - `2026-05-22_pipeline_diagram.png`
    - `2026-05-22_class_distribution.png`
    - `2026-05-22_confusion_naive_bayes.png`

Recommended essential figures and their purpose:
- `pipeline_diagram`: visualizes preprocessing → vectorization → modeling → evaluation.
- `class_distribution`: shows counts and percentages of each label.
- `accuracy_comparison`: bar chart comparing model accuracies.
- `f1_comparison`: bar chart for per-class F1.
- `confusion_<model>`: confusion matrix heatmap for each model.
- `decision_tree_structure`: visual snapshot of a small tree (if printable) or textual export.
- `preprocessing_examples`: before/after cleaning examples.
- `feature_sparsity`: visualization of sparsity (heatmap or histogram of nnz per doc).

Integration into markdown:
- Place images in `report/images/` and reference from `BaoCaoBaiTap.md` using relative links.
- Provide academic captions below each figure (one sentence + interpretation).

Screenshot and export guidance:
- Save matplotlib figures with `fig.savefig(report/images/<name>.png, dpi=150, bbox_inches='tight')`.
- For notebook screenshots, use the notebook export toolbar or `nbconvert`.

Automation:
- The `experiments/run_baseline.py` script writes outputs into `results/experiments/`; augment the script to save plots into `report/images/` after running visualizations.

This guide standardizes figure creation and integration for reproducible reporting.
