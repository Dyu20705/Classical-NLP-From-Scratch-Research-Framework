"""
Visualization utilities for model evaluation and analysis.

This module provides functions to visualize:
- Confusion matrices
- Accuracy metrics
- Feature importance
- Model performance comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


class MetricsVisualizer:
    """Visualize classification metrics and model performance."""
    
    def __init__(self, figsize=(12, 5), style='darkgrid'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 5)
            Figure size for plots
        style : str, default='darkgrid'
            Seaborn style
        """
        self.figsize = figsize
        sns.set_style(style)
    
    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', 
                            normalize=False, cmap=plt.cm.Blues, 
                            figsize=(8, 6)):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : np.ndarray, shape (n_classes, n_classes)
            Confusion matrix from confusion_matrix()
        classes : np.ndarray
            Class labels from confusion_matrix()
        title : str, default='Confusion Matrix'
            Plot title
        normalize : bool, default=False
            If True, normalize by row (recall) or column (precision)
        cmap : matplotlib colormap
            Color map for heatmap
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        plt.figure(figsize=figsize)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                       xticklabels=classes, yticklabels=classes,
                       cbar_kws={'label': 'Normalized Count'})
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                       xticklabels=classes, yticklabels=classes,
                       cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        return plt.gcf(), plt.gca()
    
    @staticmethod
    def plot_accuracy_comparison(models_dict, figsize=(10, 6)):
        """
        Plot accuracy comparison across models.
        
        Parameters:
        -----------
        models_dict : dict
            {model_name: accuracy_score}
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        names = list(models_dict.keys())
        scores = list(models_dict.values())
        colors = sns.color_palette("husl", len(names))
        
        bars = ax.bar(names, scores, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_roc_style_metrics(y_true, y_pred, figsize=(10, 6)):
        """
        Plot precision, recall, and F1-score comparison.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        from src.evaluation.accuracy import accuracy
        from src.evaluation.confusion_matrix import confusion_matrix
        
        # Compute metrics
        acc = accuracy(y_true, y_pred)
        cm, classes = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.axhline(y=acc, color='r', linestyle='--', label=f'Accuracy: {acc:.4f}')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {c}' for c in classes])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_training_history(train_acc_history, test_acc_history=None, 
                             figsize=(10, 6)):
        """
        Plot training history (accuracy over epochs/iterations).
        
        Parameters:
        -----------
        train_acc_history : list
            Training accuracy per epoch
        test_acc_history : list, optional
            Test accuracy per epoch
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = np.arange(1, len(train_acc_history) + 1)
        
        ax.plot(epochs, train_acc_history, 'o-', label='Training Accuracy',
               linewidth=2, markersize=6, color='#2ecc71')
        
        if test_acc_history is not None:
            ax.plot(epochs, test_acc_history, 's-', label='Test Accuracy',
                   linewidth=2, markersize=6, color='#e74c3c')
        
        ax.set_xlabel('Epoch / Iteration', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_class_distribution(y, labels=None, figsize=(10, 6)):
        """
        Plot class distribution (histogram).
        
        Parameters:
        -----------
        y : array-like
            Class labels
        labels : list, optional
            Class names (e.g., ['Negative', 'Positive'])
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        classes, counts = np.unique(y, return_counts=True)
        colors = sns.color_palette("Set2", len(classes))
        
        bars = ax.bar([f'Class {c}' if labels is None else labels[int(c)] 
                      for c in classes], 
                      counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/len(y)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_prediction_distribution(y_true, y_pred, figsize=(12, 5)):
        """
        Plot side-by-side distribution of true vs predicted labels.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        classes_true, counts_true = np.unique(y_true, return_counts=True)
        classes_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        colors = sns.color_palette("Set2", max(len(classes_true), len(classes_pred)))
        
        # True distribution
        axes[0].bar([f'Class {c}' for c in classes_true], counts_true,
                   color=colors[:len(classes_true)], alpha=0.8, edgecolor='black')
        axes[0].set_title('True Label Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Predicted distribution
        axes[1].bar([f'Class {c}' for c in classes_pred], counts_pred,
                   color=colors[:len(classes_pred)], alpha=0.8, edgecolor='black')
        axes[1].set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, axes


def plot_all_metrics(y_true, y_pred, figsize=(15, 10)):
    """
    Create comprehensive visualization dashboard.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    figsize : tuple
        Overall figure size
    
    Returns:
    --------
    figs : list of matplotlib figures
    """
    from src.evaluation.accuracy import accuracy
    from src.evaluation.confusion_matrix import confusion_matrix
    
    figs = []
    
    # 1. Confusion matrix
    cm, classes = confusion_matrix(y_true, y_pred)
    fig1, _ = MetricsVisualizer.plot_confusion_matrix(cm, classes)
    figs.append(fig1)
    
    # 2. Metrics comparison
    fig2, _ = MetricsVisualizer.plot_roc_style_metrics(y_true, y_pred)
    figs.append(fig2)
    
    # 3. Class distribution
    fig3, _ = MetricsVisualizer.plot_class_distribution(y_true)
    figs.append(fig3)
    
    # 4. Prediction distribution
    fig4, _ = MetricsVisualizer.plot_prediction_distribution(y_true, y_pred)
    figs.append(fig4)
    
    return figs
