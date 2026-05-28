import numpy as np
from collections import Counter

class node:
    def __init__(self, feature=None, threshold=None, info_gain=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right
        self.value = value

class decisiontree:
    def __init__(self, max_depth=30, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def build(self, dataset, curr_depth=0):
        X, y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = X.shape

        if num_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split = self.get_best_split(dataset, num_features)
            if best_split["info_gain"] > 0:
                left_node = self.build(best_split["left_dataset"], curr_depth + 1)
                right_node = self.build(best_split["right_dataset"], curr_depth + 1)
                return node(feature=best_split["feature_idx"],
                             threshold=best_split["threshold"],
                               info_gain=best_split["info_gain"],
                                 left=left_node,
                                   right=right_node)
        
        leaf_value = Counter(y).most_common(1)[0][0]
        return node(value=leaf_value)
    
    def get_best_split(self, dataset, num_features):
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
                    
                    info_gain = self.info_gain(parent_y, left_y, right_y)
                    
                    if info_gain > best_split["info_gain"]:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = info_gain
                        best_split["left_dataset"] = dataset[left_indices]
                        best_split["right_dataset"] = dataset[right_indices]

        return best_split

    def info_gain(self, parent_y, left_y, right_y):
        if len(parent_y) == 0:
            return 0
        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)
        parent_entropy = self.entropy(parent_y)
        left_entropy = self.entropy(left_y) if len(left_y) > 0 else 0
        right_entropy = self.entropy(right_y) if len(right_y) > 0 else 0
        info_gain = parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)
        return info_gain

    def entropy(self, y):
        entropy = 0.0
        if len(y) == 0:
            return entropy
        
        lbs = np.unique(y)
        for lb in lbs:
            p = np.sum(y == lb) / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build(dataset)

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X)
        predictions = [self._predict_single(x, self.root) for x in X]
        return np.array(predictions)
    
    def predict_single(self, x, node):
        if node.value is not None:
            return node.value
        
        feature_value = x[node.feature]
        if feature_value <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)
    
    def print_tree(self, node=None, depth=0, indent="|    "):
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