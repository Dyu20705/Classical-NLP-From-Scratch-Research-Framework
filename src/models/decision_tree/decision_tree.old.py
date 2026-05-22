from collections import Counter
import numpy as np
from src.models.base import BaseModel


class Node:
    def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.info_gain = info_gain  # Information gain from the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label for leaf nodes


class DecisionTree(BaseModel):
    """
    Decision Tree classifier using Information Gain (entropy-based) splitting.
    
    Supports:
    - Dense numpy arrays (converts sparse to dense if needed)
    - Max depth and min samples per split constraints
    - Entropy-based information gain for split selection
    
    Parameters:
    -----------
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    max_depth : int, default=2
        Maximum depth of the tree.
    """
    
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _build_tree(self, dataset, curr_depth=0):
        """Recursively build decision tree."""
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape

        if num_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split = self._get_best_split(dataset, num_features)
            if best_split["info_gain"] > 0:
                left_node = self._build_tree(best_split["left_dataset"], curr_depth + 1)
                right_node = self._build_tree(best_split["right_dataset"], curr_depth + 1)
                return Node(
                    feature_idx=best_split["feature_idx"],
                    threshold=best_split["threshold"],
                    info_gain=best_split["info_gain"],
                    left=left_node,
                    right=right_node
                )
        
        # Leaf node: most common class
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    def _get_best_split(self, dataset, num_features):
        """Find the best split for a dataset."""
        best_split = {
            "feature_idx": None,
            "threshold": None,
            "info_gain": -1,
            "left_dataset": None,
            "right_dataset": None
        }

        for feature_idx in range(num_features):
            feature_values = dataset[:, feature_idx]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                
                if len(left_indices) > 0 and len(right_indices) > 0:
                    parent_y = dataset[:, -1]
                    left_y = dataset[left_indices, -1]
                    right_y = dataset[right_indices, -1]
                    
                    info_gain = self._calculate_info_gain(parent_y, left_y, right_y)
                    
                    if info_gain > best_split["info_gain"]:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = info_gain
                        best_split["left_dataset"] = dataset[left_indices]
                        best_split["right_dataset"] = dataset[right_indices]
        
        return best_split

    def _calculate_info_gain(self, parent_y, left_y, right_y):
        """Calculate information gain from a split."""
        if len(parent_y) == 0:
            return 0
        
        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)

        parent_entropy = self._calculate_entropy(parent_y)
        left_entropy = self._calculate_entropy(left_y) if len(left_y) > 0 else 0
        right_entropy = self._calculate_entropy(right_y) if len(right_y) > 0 else 0
        
        child_entropy = weight_left * left_entropy + weight_right * right_entropy
        info_gain = parent_entropy - child_entropy
        
        return info_gain

    def _calculate_entropy(self, y):
        """Calculate entropy of a label distribution."""
        if len(y) == 0:
            return 0
        
        entropy = 0.0
        class_labels = np.unique(y)
        
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            if p_cls > 0:
                entropy -= p_cls * np.log2(p_cls)
        
        return entropy

    def fit(self, X, y):
        """
        Train the decision tree.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        
        Returns:
        --------
        self : DecisionTree
            Fitted tree instance.
        """
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Combine features and target
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        
        # Build tree
        self.root = self._build_tree(dataset)
        
        return self

    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Features to predict on.
        
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X)
        
        predictions = [self._predict_single(row, self.root) for row in X]
        return np.array(predictions)

    def _predict_single(self, row, node):
        """Predict class for a single sample by traversing the tree."""
        if node.value is not None:
            return node.value
        
        feature_value = row[node.feature_idx]
        
        if feature_value <= node.threshold:
            return self._predict_single(row, node.left)
        else:
            return self._predict_single(row, node.right)

    def print_tree(self, node=None, depth=0, indent="|    "):
        """Print tree structure (for debugging)."""
        prefix = indent * depth

        if node is None:
            node = self.root
        
        if node.value is not None:
            print(f"{prefix}|--- class: {node.value}")
            return
        
        print(f"{prefix}|--- feature_{node.feature_idx} <= {node.threshold}")
        print(f"{prefix}|   (info_gain: {node.info_gain:.4f})")
        
        if node.left:
            print(f"{prefix}|   LEFT:")
            self.print_tree(node.left, depth + 2, indent)
        
        if node.right:
            print(f"{prefix}|   RIGHT:")
            self.print_tree(node.right, depth + 2, indent)
        
        feature_label = f"Feature {node.feature_idx}"

        print(f"{prefix}|--- {feature_label} <= {node.threshold:.4f}")
        print(f"{prefix}|--- {feature_label} > {node.threshold:.4f}")
        self.print_tree(node.right, depth + 1, indent)
        self.print_tree(node.left, depth + 1, indent)
        