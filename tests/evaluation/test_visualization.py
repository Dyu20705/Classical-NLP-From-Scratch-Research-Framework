import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.evaluation.visualization import MetricsVisualizer


def test_plot_confusion_matrix_returns_fig_ax():
    cm = np.array([[2, 1], [1, 2]])
    classes = np.array([0, 1])
    fig, ax = MetricsVisualizer.plot_confusion_matrix(cm, classes, normalize=True)
    assert fig is not None
    assert ax is not None


def test_plot_accuracy_comparison_returns_fig_ax():
    fig, ax = MetricsVisualizer.plot_accuracy_comparison({'A': 0.8, 'B': 0.9})
    assert fig is not None
    assert ax is not None
